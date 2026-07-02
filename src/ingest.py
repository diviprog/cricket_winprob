"""M0 -- Ingestion and state reduction (spec 01).

Turns the raw Cricsheet-derived IPL ball-by-ball file into the shared data
contract consumed by 02 (outcome model), 05 (validation) and 06 (baseline):
one row per legal second-innings ball, carrying the PRE-ball state (b, w, r),
the outcome that followed, and the realized match label y.

Run:  uv run python -m src.ingest

Modeling decisions recorded here (spec 07 requires these be explicit):

* Extras -- Option A. Only legal deliveries (`valid_ball == 1`) are emitted as
  outcome rows, so `b` decrements strictly one-per-row. Runs from wides/no-balls
  are already folded into the running score, so they are reflected in the
  PRE-ball `r` of subsequent legal balls, but are NOT modeled as their own DP
  transition. Known, small, documented downward bias in modeled scoring rate.

* Wicket-ball runs -- a legal ball on which a wicket falls is classified `W`;
  any runs on that ball are not separately modeled (slight run under-count).

* Wickets in hand -- derived from the dataset's cumulative `team_wicket`, so
  non-dismissal exits (retired hurt) are handled by the source's own
  convention and stay consistent with the `W` outcome flag.

* Ties -- kept (not dropped). Terminal value 0.5 lives in wp_markov; the
  realized label y for a tied-then-super-over match is the actual match winner.

* Dropped matches -- DLS/D-L affected, "no result", and any innings not played
  to the standard 20-over (120-ball) budget, because a revised or non-standard
  target breaks the fixed-target assumption. Counts are reported.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config

# Raw columns we rely on, mapped to their canonical role. Schema is verified
# (not assumed) against this before any logic runs.
REQUIRED_COLUMNS = {
    "match_id": "join key / innings grouping",
    "innings": "innings number (keep == 2)",
    "over": "over number (0-indexed in this dump)",
    "date": "match date, for chronological season ordering",
    "season": "season label",
    "batting_team": "second-innings batting side",
    "bowling_team": "bowling side",
    "valid_ball": "1 = legal delivery, 0 = wide/no-ball",
    "runs_total": "runs on the ball (bat + extras)",
    "team_runs": "cumulative team runs, POST-ball, per innings",
    "team_balls": "cumulative legal balls, POST-ball, per innings",
    "team_wicket": "cumulative wickets, POST-ball, per innings",
    "runs_target": "second-innings target (= first-innings total + 1)",
    "match_won_by": "winning team name (realized label source)",
    "result_type": "tie / no result / NaN",
    "method": "D/L when DLS-affected",
    "overs": "scheduled overs for the innings (20 for a full match)",
    "batter_balls": "cumulative balls faced by striker (for M5 striker bucket)",
}


def _verify_schema(df: pd.DataFrame) -> None:
    """Fail loudly if any required field is missing (spec 01)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            "FATAL: raw file is missing required columns: "
            + ", ".join(missing)
            + "\nPresent columns:\n  "
            + ", ".join(df.columns)
        )


def _phase_from_over_index(over_idx: np.ndarray) -> np.ndarray:
    """Coarse innings stage from the 0-indexed over of a legal ball (spec 02).

    powerplay = overs 1-6, middle = 7-15, death = 16-20.
    """
    phase = np.where(over_idx < 6, "powerplay", np.where(over_idx < 15, "middle", "death"))
    return phase


def _resolve_split(df: pd.DataFrame) -> pd.Series:
    """Assign every row a 'train'/'holdout' split label (spec 02/05/06).

    Seasons are ordered by their first match date; the most recent
    N_HELDOUT_SEASONS become the held-out set. Fixed here so it is identical
    everywhere downstream.
    """
    first_date = df.groupby("season")["date"].min().sort_values()
    ordered_seasons = list(first_date.index)
    holdout = set(ordered_seasons[-config.N_HELDOUT_SEASONS :])
    return df["season"].map(lambda s: "holdout" if s in holdout else "train"), ordered_seasons, holdout


def build_ball_table(raw_csv=config.RAW_CSV) -> pd.DataFrame:
    """Load raw CSV -> processed second-innings ball table (the data contract)."""
    df = pd.read_csv(raw_csv, low_memory=False)
    _verify_schema(df)

    n_matches_total = df["match_id"].nunique()

    # Preserve raw file order (== delivery order within a match) so we can diff
    # cumulative columns over the true ball sequence, including illegal balls.
    df = df.reset_index(names="_seq")

    # --- second innings only (spec 00 scope). Drops super-over innings (3+). --
    df = df[df["innings"] == 2].copy()
    df = df.sort_values(["match_id", "_seq"]).reset_index(drop=True)

    # --- drop matches whose fixed-target assumption is broken -----------------
    dls_matches = df.loc[df["method"].astype(str).eq("D/L"), "match_id"].unique()
    noresult_matches = df.loc[df["result_type"].astype(str).eq("no result"), "match_id"].unique()
    # Non-standard innings length (rain-reduced but not flagged D/L, etc.)
    nonstd_matches = df.loc[df["overs"] != 20, "match_id"].unique()
    drop_ids = set(dls_matches) | set(noresult_matches) | set(nonstd_matches)

    df = df[~df["match_id"].isin(drop_ids)].copy()

    # --- wicket-on-this-ball from the cumulative team_wicket delta -------------
    # Computed over the FULL ball sequence (incl. illegal balls) so a wicket on a
    # no-ball is attributed to that no-ball, not smeared onto the next legal ball.
    grp = df.groupby("match_id", sort=False)["team_wicket"]
    prev_wkts = grp.shift(1).fillna(0).astype(int)
    delta = df["team_wicket"].astype(int) - prev_wkts
    if delta.max() > 1:
        raise SystemExit("FATAL: team_wicket jumped by >1 on a single ball.")
    df["_is_wicket"] = delta.clip(lower=0)

    # --- fold wide/no-ball runs into the FOLLOWING legal ball (spec 01 Opt A) --
    # Extras on illegal deliveries add runs but consume no legal ball. To keep the
    # DP on integer legal balls while still crediting the batting side those runs,
    # attribute each illegal ball's runs to the next legal ball's outcome ("absorbed
    # into the run distribution of the following legal ball"). Byes/legbyes fall on
    # legal balls (valid_ball==1) and are already inside runs_total, so only
    # wides/no-balls (valid_ball==0) are folded here.
    legal = df["valid_ball"].eq(1)
    df["_legal_rank"] = legal.groupby(df["match_id"]).cumsum().astype(int)
    illegal = df[~legal]
    # key absorbed extras by (match, rank of the FOLLOWING legal ball = last legal + 1)
    absorbed = illegal.groupby([illegal["match_id"], illegal["_legal_rank"] + 1])["runs_total"].sum()

    # --- keep only legal deliveries -------------------------------------------
    df = df[legal].copy().reset_index(drop=True)
    is_wicket = df["_is_wicket"].astype(int)
    key = pd.MultiIndex.from_arrays([df["match_id"], df["_legal_rank"]])
    df["_absorbed"] = absorbed.reindex(key).fillna(0).to_numpy().astype(int)
    extras_runs_dropped = int(absorbed.sum() - df["_absorbed"].sum())  # trailing extras, no next legal ball

    # --- PRE-ball state (b, w, r) ---------------------------------------------
    # team_* are POST-ball cumulative; back them out to before this ball.
    team_balls = df["team_balls"].astype(int)
    balls_before = team_balls - 1  # legal balls bowled before this ball, 0..119
    b = config.BALLS_PER_INNINGS - balls_before  # balls remaining, 1..120

    wickets_before = df["team_wicket"].astype(int) - is_wicket
    w = config.WICKETS - wickets_before  # wickets in hand, 1..10

    runs_before = df["team_runs"].astype(int) - df["runs_total"].astype(int)
    target = df["runs_target"].astype(int)
    r = target - runs_before  # runs required before this ball

    # --- outcome category ------------------------------------------------------
    # effective runs = legal-ball runs + extras folded in from preceding illegal
    # balls, capped at 6 to fit the {0..6} vocabulary (a boundary preceded by a
    # wide is the only common >6 case; the cap loses a rare run, counted below).
    runs = df["runs_total"].astype(int)
    if runs.max() > 6 or runs.min() < 0:
        raise SystemExit(f"FATAL: unexpected runs_total on a legal ball: {runs.min()}..{runs.max()}")
    effective = runs + df["_absorbed"]
    n_capped = int((effective > 6).sum())
    effective = effective.clip(upper=6)
    outcome = np.where(is_wicket == 1, "W", effective.astype(int).astype(str))

    # --- realized label y (from the summary winner, not reconstructed) --------
    y = (df["match_won_by"] == df["batting_team"]).astype(int)

    # --- derived features carried in the contract -----------------------------
    over_idx = balls_before // 6
    phase = _phase_from_over_index(over_idx.to_numpy())
    rrr = 6.0 * r / b  # b >= 1 here, always defined
    striker_balls_before = (df["batter_balls"].astype(int) - 1).clip(lower=0)  # for M5

    out = pd.DataFrame(
        {
            "match_id": df["match_id"].to_numpy(),
            "season": df["season"].to_numpy(),
            "date": df["date"].to_numpy(),
            "ball_index": balls_before.to_numpy(),  # 0..119 within innings
            "over": over_idx.to_numpy(),
            "b": b.to_numpy(),
            "w": w.to_numpy(),
            "r": r.to_numpy(),
            "outcome": outcome,
            "rrr": rrr.to_numpy(),
            "phase": phase,
            "striker_balls": striker_balls_before.to_numpy(),
            "y": y.to_numpy(),
        }
    )

    split, ordered_seasons, holdout = _resolve_split(out)
    out["split"] = split.to_numpy()

    out.attrs["dropped"] = {
        "dls": len(dls_matches),
        "no_result": len(noresult_matches),
        "non_standard_overs": len(nonstd_matches),
        "total_matches_dropped": len(drop_ids),
        "matches_total_dataset": int(n_matches_total),
    }
    out.attrs["extras"] = {
        "runs_folded_in": int(df["_absorbed"].sum()),
        "runs_dropped_trailing": int(extras_runs_dropped),
        "balls_capped_over6": int(n_capped),
    }
    out.attrs["ordered_seasons"] = ordered_seasons
    out.attrs["holdout_seasons"] = sorted(holdout)
    return out


# --------------------------------------------------------------------------
# Sanity checks (spec 01 / M0 gate)
# --------------------------------------------------------------------------
def run_sanity_checks(out: pd.DataFrame) -> list[str]:
    """Return a list of failure messages; empty list means all gates pass."""
    fails: list[str] = []

    # monotonicity within an innings
    g = out.sort_values(["match_id", "ball_index"]).groupby("match_id")
    if not g["b"].apply(lambda s: s.is_monotonic_decreasing).all():
        fails.append("b is not monotonically decreasing within every innings")
    if not g["w"].apply(lambda s: s.is_monotonic_decreasing).all():
        fails.append("w is not non-increasing within every innings")
    if not g["r"].apply(lambda s: s.is_monotonic_decreasing).all():
        fails.append("r is not non-increasing within every innings")

    # impossible states
    if not out["b"].between(1, 120).all():
        fails.append("b outside [1, 120]")
    if not out["w"].between(1, 10).all():
        fails.append("w outside [1, 10]")

    # y constant per innings
    if (g["y"].nunique() != 1).any():
        fails.append("y is not constant within every innings")

    # plausible row count
    n = len(out)
    if not (100_000 <= n <= 160_000):
        fails.append(f"row count {n} outside plausible ~120k-150k range")

    # outcome distribution: dots + singles dominate, wickets a few %
    dist = out["outcome"].value_counts(normalize=True)
    if dist.get("0", 0) + dist.get("1", 0) < 0.5:
        fails.append("dots+singles do not dominate outcome distribution")
    if not (0.01 < dist.get("W", 0) < 0.10):
        fails.append(f"wicket rate {dist.get('W', 0):.3f} implausible")

    return fails


def main() -> int:
    print(f"Reading {config.RAW_CSV} ...")
    out = build_ball_table()

    d = out.attrs["dropped"]
    print("\n=== drops (fixed-target assumption) ===")
    print(f"  matches in dataset (all innings): {d['matches_total_dataset']}")
    print(f"  dropped DLS/D-L:                  {d['dls']}")
    print(f"  dropped no-result:                {d['no_result']}")
    print(f"  dropped non-20-over innings:      {d['non_standard_overs']}")
    print(f"  total matches dropped:            {d['total_matches_dropped']}")

    e = out.attrs["extras"]
    print("\n=== extras fold-in (spec 01 Option A) ===")
    print(f"  wide/no-ball runs folded into next legal ball: {e['runs_folded_in']}")
    print(f"  trailing extras dropped (no next legal ball):  {e['runs_dropped_trailing']}")
    print(f"  legal balls capped at 6 after fold-in:         {e['balls_capped_over6']}")

    print("\n=== split ===")
    print(f"  seasons (chronological): {out.attrs['ordered_seasons']}")
    print(f"  held-out seasons:        {out.attrs['holdout_seasons']}")
    print(out.groupby("split")["match_id"].nunique().to_string())

    print("\n=== outcome distribution ===")
    print(out["outcome"].value_counts(normalize=True).sort_index().round(4).to_string())

    print("\n=== M0 sanity gate ===")
    fails = run_sanity_checks(out)
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
        return 1
    print(f"  PASS  ({len(out):,} legal second-innings balls, "
          f"{out['match_id'].nunique()} matches)")

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.BALLS_PARQUET, index=False)
    print(f"\nWrote {config.BALLS_PARQUET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
