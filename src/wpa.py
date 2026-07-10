"""M3+ -- Win Probability Added (WPA): clutch attribution.

Leverage (spec 04) measures how much a ball COULD swing WP -- the stakes. WPA is
the realized SIGNED swing a player actually produced:

    WPA_ball = WP(after ball) - WP(before ball)     # batting-team perspective
      batter credit = +WPA_ball  (they moved WP toward a chase win)
      bowler credit = -WPA_ball  (they moved it toward a defence)

Everything needed is already on the ball table: `wp` (the era-adjusted RRR Markov
WP leverage is defined on) and `y`. Within an innings WP(after ball_t) is just
WP(before ball_{t+1}); the final ball transitions to the realized terminal `y`
(1 for a match-winning hit, 0 for the last wicket). That is the exact same path
`leverage.martingale_drift` walks, so WPA telescopes per innings to `y - wp_1`.

Why the martingale property makes this a real clutch measure: WP is an exact
martingale under the outcome model, so E[WPA | state] = 0. A player accumulates
positive WPA ONLY by beating the model's state-conditioned expectation (a six
when the spot expected ~1.3 runs; a dot+wicket when the batting side was
cruising). WPA is therefore "beat expectation, weighted by how much the moment
could swing" -- not "good things happened nearby." It is automatically
leverage-weighted, which is exactly the "high LI and it went their way" idea.

Caveats carried into every number here:
* Inherits the Markov WP's residual lag-1 autocorrelation (+0.037, see
  leverage_validation.md): biases everyone the same way, so RANKINGS are robust,
  absolute WPA less so.
* A wicket's WPA goes entirely to the bowler; run-outs are folded into `W` and
  not separately attributed (spec 01).
* WPA is zero-sum per ball (bowler = -batter), so it scores the batter-vs-bowler
  MATCHUP outcome, not a context-free skill.

Run:  .venv/bin/python -m src.wpa   (after src.leverage has written `wp`,`li`)
Writes reports/wpa.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

HIGH_LI = 2.0    # "high leverage" = a ball worth >= 2x the average ball's swing
MIN_BALLS = 500  # career balls for a stable total/rate

# WP-bin width for the state-conditional drift estimate (see per_ball_wpa). The
# martingale drift is a WP-CALIBRATION error, so it is estimated and removed along
# the WP axis. 0.02 gives 50 bins; the contested band [0.3,0.5] where drift peaks
# (leverage_validation.md) gets ~10 bins of several thousand balls each -- fine
# resolution where it matters, and dead-state point masses at 0/1 (drift ~0, WPA
# ~0) fall in harmless edge bins.
DRIFT_BIN_WIDTH = 0.02


def per_ball_wpa(df: pd.DataFrame) -> pd.DataFrame:
    """Attach signed per-ball WPA, raw and de-drifted, to the table.

    Raw WPA. WP(after ball) = WP(before next ball) within the innings; on the last
    ball it is the realized match label `y` (the decisive transition to a terminal
    1/0). One innings per match_id (spec 01 keeps only the 2nd), so the innings
    grouping IS match_id.

    De-drifted WPA. A calibrated martingale has E[WPA | state] = 0, but the model
    is not perfectly calibrated: batting sides collectively finish ~+62 wins above
    the model's start-of-chase expectation (validation_report.md's tail-thinning
    plus the modern scoring-era shift). That surplus is credited entirely to
    batters and debited from bowlers -- a per-ball baseline of ~+0.00048 WP that
    has nothing to do with skill and makes raw batter vs bowler totals
    incomparable. It is NOT a flat constant: the drift ranges from ~0.00007 in
    decided states to ~0.00185 in contested ones (leverage_validation.md), so it
    is removed STATE-CONDITIONALLY -- subtract the mean batting WPA among balls
    starting at a similar WP:

        wpa_bat_dd = wpa_bat - E[wpa_bat | WP-bin of the pre-ball state]

    Any one player is a tiny fraction of each bin, so this removes the POPULATION
    baseline (the model's calibration error), not the player's own edge -- the
    standard "runs above replacement" logic. Removing it state-conditionally, not
    flatly, is what keeps it fair across players with different state mixes (a
    death bowler lives in low-drift extremes; a middle-overs bowler in the
    high-drift band). Zero-sum survives (wpa_bowl_dd = -wpa_bat_dd), and the grand
    total telescopes to ~0 instead of +62, redistributed across players by where
    they actually operated.

    Caveat: WP is the dominant but not the only axis of the calibration gap, so a
    second-order residual drift survives WP-binning. This is a diagnostic
    de-bias, not a re-calibration of the WP surface itself (that is the outcome
    model's job -- see baseline_comparison.md).
    """
    b = df.sort_values(["match_id", "ball_index"]).copy()
    wp_after = b.groupby("match_id", sort=False)["wp"].shift(-1)
    wp_after = wp_after.where(wp_after.notna(), b["y"].astype(float))  # last ball -> y
    b["wpa_bat"] = wp_after - b["wp"]
    b["wpa_bowl"] = -b["wpa_bat"]

    # state-conditional drift on the PRE-ball WP, subtracted per ball
    bin_id = np.floor(np.clip(b["wp"].to_numpy(), 0.0, 1.0) / DRIFT_BIN_WIDTH).astype(int)
    b["wpa_drift"] = b.groupby(bin_id)["wpa_bat"].transform("mean")
    b["wpa_bat_dd"] = b["wpa_bat"] - b["wpa_drift"]
    b["wpa_bowl_dd"] = -b["wpa_bat_dd"]
    return b


def _agg(b: pd.DataFrame, who: str, side: str) -> pd.DataFrame:
    """Per-player WPA aggregate with a high- vs low-leverage clutch split.

    `side` is 'bat' or 'bowl'. Both raw (`total_wpa`) and de-drifted
    (`total_wpa_dd`) career totals are reported; the clutch rate split uses the
    de-drifted column, because high-LI balls carry the most drift and the raw
    split is the most drift-inflated part of the whole report.
    """
    raw, dd = f"wpa_{side}", f"wpa_{side}_dd"
    g = b.groupby(who)
    out = pd.DataFrame({
        "balls": g.size(),
        "total_wpa": g[raw].sum(),            # raw, shown for transparency
        "total_wpa_dd": g[dd].sum(),          # de-drifted -- the fair career rank
        "wpa_dd_per100": 100 * g[dd].mean(),
        "mean_li": g["li"].mean(),
    })
    hi = b[b["li"] >= HIGH_LI].groupby(who)
    lo = b[b["li"] < HIGH_LI].groupby(who)
    out["n_high"] = hi.size()
    out["wpa_high_dd"] = hi[dd].mean()   # de-drifted per-ball WPA in clutch moments
    out["wpa_low_dd"] = lo[dd].mean()    # de-drifted per-ball WPA in ordinary moments
    out["n_high"] = out["n_high"].fillna(0).astype(int)
    return out[out["balls"] >= MIN_BALLS]


def batter_wpa(b: pd.DataFrame) -> pd.DataFrame:
    return _agg(b, "batter", "bat")


def bowler_wpa(b: pd.DataFrame) -> pd.DataFrame:
    return _agg(b, "bowler", "bowl")


def _fmt(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> int:
    df = pd.read_parquet(config.BALLS_PARQUET)
    if "wp" not in df.columns or "li" not in df.columns:
        raise SystemExit("FATAL: need `wp` and `li` -- run `.venv/bin/python -m src.leverage` first.")

    b = per_ball_wpa(df)

    # --- sanity: zero-sum per ball, telescoping per innings, de-drift removes it
    zero_sum = float((b["wpa_bat"] + b["wpa_bowl"]).abs().max())
    zero_sum_dd = float((b["wpa_bat_dd"] + b["wpa_bowl_dd"]).abs().max())
    total_bat = float(b["wpa_bat"].sum())
    total_bat_dd = float(b["wpa_bat_dd"].sum())
    drift_per_ball = total_bat / len(b)
    first = b.groupby("match_id", sort=False).first()
    telescope = float((first["y"] - first["wp"]).sum())  # == sum of batting WPA

    bats = batter_wpa(b).round(4)
    bowls = bowler_wpa(b).round(4)

    # a concrete cross-role inversion caused by de-drifting, taken from the data:
    # the batter who loses the most and the bowler who gains the most.
    bat_drop = (bats["total_wpa"] - bats["total_wpa_dd"]).idxmax()
    bowl_gain = (bowls["total_wpa_dd"] - bowls["total_wpa"]).idxmax()
    ex_bat, ex_bowl = bats.loc[bat_drop], bowls.loc[bowl_gain]

    bcols = ["player", "balls", "total_wpa", "total_wpa_dd", "wpa_dd_per100",
             "mean_li", "n_high", "wpa_high_dd", "wpa_low_dd"]
    top_bat_total = bats.sort_values("total_wpa_dd", ascending=False).head(20).reset_index().rename(columns={"batter": "player"})
    top_bowl_total = bowls.sort_values("total_wpa_dd", ascending=False).head(20).reset_index().rename(columns={"bowler": "player"})
    # clutch rate: de-drifted per-ball WPA on high-LI balls, min sample of clutch balls
    clutch_cols = ["player", "balls", "n_high", "wpa_high_dd", "wpa_low_dd", "total_wpa_dd"]
    clutch_bat = bats[bats["n_high"] >= 100].sort_values("wpa_high_dd", ascending=False).head(20).reset_index().rename(columns={"batter": "player"})
    clutch_bowl = bowls[bowls["n_high"] >= 100].sort_values("wpa_high_dd", ascending=False).head(20).reset_index().rename(columns={"bowler": "player"})

    L = [
        "# Win Probability Added (WPA) -- clutch attribution",
        "",
        f"WPA = realized signed one-ball WP change, credited +to the batter and "
        f"-to the bowler, on the era-adjusted RRR Markov WP. Because that WP is an "
        f"exact martingale (E[WPA|state]=0), positive WPA means BEATING the model's "
        f"expectation for the state, automatically weighted by the leverage of the "
        f"moment. Career threshold >= {MIN_BALLS} balls; 'high leverage' = LI >= "
        f"{HIGH_LI} (>= {HIGH_LI}x the average ball). **All tables rank by de-drifted "
        f"WPA (`_dd`); see below.**",
        "",
        f"*Sanity:* per-ball batter+bowler WPA cancels to <= {zero_sum:.2e} (zero-sum); "
        f"raw total batting WPA {total_bat:+.2f} equals the telescoped innings sum "
        f"y - wp(first ball) = {telescope:+.2f}. De-drifting preserves the zero-sum "
        f"(<= {zero_sum_dd:.2e}) and drives the grand batting total to "
        f"{total_bat_dd:+.2e} (the +{total_bat:.0f} model surplus removed).",
        "",
        "## Why de-drifted WPA",
        "",
        f"The WP surface is not perfectly calibrated: batting sides collectively finish "
        f"**{total_bat:+.1f} wins** above the model's start-of-chase expectation "
        f"(the memoryless tail-thinning of validation_report.md plus the modern "
        f"scoring-era shift). In a zero-sum matchup that surplus cannot be real skill -- "
        f"it is a **~{drift_per_ball:+.5f} WP-per-ball baseline** handed to every batter "
        f"and taken from every bowler, which makes raw batter-vs-bowler totals "
        f"incomparable. It is removed state-conditionally (the drift ranges "
        f"~0.00007-0.00185 across WP regimes, so a flat subtraction would misprice "
        f"players by where they operate): subtract the mean batting WPA among balls "
        f"starting at a similar WP. Any player is a sliver of each WP bin, so this "
        f"strips the population baseline, not individual edge.",
        "",
        f"Concretely, the largest single shifts here: **{bat_drop}** (batting) falls "
        f"{ex_bat['total_wpa']:+.2f} -> {ex_bat['total_wpa_dd']:+.2f} once the free "
        f"baseline is removed, while **{bowl_gain}** (bowling) rises "
        f"{ex_bowl['total_wpa']:+.2f} -> {ex_bowl['total_wpa_dd']:+.2f}. Never compare "
        f"a raw batter total to a raw bowler total; compare `total_wpa_dd`.",
        "",
        "## Batsmen -- career WPA (de-drifted; total win probability added when chasing)",
        "",
        "`total_wpa` = raw (shown for transparency); `total_wpa_dd` = de-drifted, the "
        "fair rank; `wpa_dd_per100` = de-drifted per 100 balls; `wpa_high_dd` / "
        "`wpa_low_dd` = de-drifted per-ball WPA in high- vs ordinary-leverage moments. "
        "A genuine clutch player has `wpa_high_dd` > 0 and ideally > `wpa_low_dd`.",
        "",
        _fmt(top_bat_total, bcols),
        "",
        "## Batsmen -- pure clutch rate (de-drifted per-ball WPA in high-leverage moments)",
        "",
        f"Ranked by `wpa_high_dd` among batters with >= 100 high-LI balls faced -- who "
        f"actually converts the big moments, rate-adjusted so volume doesn't dominate. "
        f"De-drifting matters most here: high-LI balls carry the heaviest drift, so the "
        f"raw clutch rate was the most inflated number in the report.",
        "",
        _fmt(clutch_bat[clutch_cols], clutch_cols),
        "",
        "## Bowlers -- career WPA (de-drifted; win probability added by defending)",
        "",
        _fmt(top_bowl_total, bcols),
        "",
        "## Bowlers -- pure clutch rate (de-drifted per-ball WPA in high-leverage moments)",
        "",
        _fmt(clutch_bowl[clutch_cols], clutch_cols),
        "",
        "## Reading these tables",
        "",
        "- **total_wpa_dd** is the career clutch-value rank: it rewards being trusted "
        "with the moments AND delivering, with the model's calibration surplus removed "
        "so batters and bowlers sit on one scale.",
        "- **wpa_high_dd vs wpa_low_dd** is the honest clutch signal: a player whose "
        "per-ball WPA is higher when leverage is high genuinely raises their game; one "
        "positive only in low-LI moments padded stats in dead situations.",
        "- WPA scores the batter-vs-bowler matchup and inherits the model's +0.037 "
        "autocorrelation bound; de-drifting removes the first-order calibration bias "
        "but a second-order residual survives WP-binning, so treat the ranking as "
        "robust and the absolute decimals as approximate.",
        "",
    ]
    out = config.REPORTS / "wpa.md"
    out.write_text("\n".join(L))
    print(f"zero_sum_max={zero_sum:.2e} (dd {zero_sum_dd:.2e})  "
          f"total_bat_wpa raw={total_bat:+.3f} dd={total_bat_dd:+.2e}  telescope={telescope:+.3f}")
    print(f"drift/ball={drift_per_ball:+.5f}  example: {bat_drop} bat "
          f"{ex_bat['total_wpa']:+.2f}->{ex_bat['total_wpa_dd']:+.2f}, "
          f"{bowl_gain} bowl {ex_bowl['total_wpa']:+.2f}->{ex_bowl['total_wpa_dd']:+.2f}")
    print(f"batters ranked: {len(bats)}  bowlers ranked: {len(bowls)}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
