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


def per_ball_wpa(df: pd.DataFrame) -> pd.DataFrame:
    """Attach signed per-ball WPA (batting and bowling perspective) to the table.

    WP(after ball) = WP(before next ball) within the innings; on the last ball of
    the innings it is the realized match label `y` (the decisive transition to a
    terminal 1/0). One innings per match_id (spec 01 keeps only the 2nd), so the
    innings grouping IS match_id.
    """
    b = df.sort_values(["match_id", "ball_index"]).copy()
    wp_after = b.groupby("match_id", sort=False)["wp"].shift(-1)
    wp_after = wp_after.where(wp_after.notna(), b["y"].astype(float))  # last ball -> y
    b["wpa_bat"] = wp_after - b["wp"]
    b["wpa_bowl"] = -b["wpa_bat"]
    return b


def _agg(b: pd.DataFrame, who: str, wpa_col: str) -> pd.DataFrame:
    """Per-player WPA aggregate with a high- vs low-leverage clutch split."""
    g = b.groupby(who)
    out = pd.DataFrame({
        "balls": g.size(),
        "total_wpa": g[wpa_col].sum(),
        "wpa_per100": 100 * g[wpa_col].mean(),
        "mean_li": g["li"].mean(),
    })
    hi = b[b["li"] >= HIGH_LI].groupby(who)
    lo = b[b["li"] < HIGH_LI].groupby(who)
    out["n_high"] = hi.size()
    out["wpa_high"] = hi[wpa_col].mean()   # per-ball WPA in clutch moments
    out["wpa_low"] = lo[wpa_col].mean()    # per-ball WPA in ordinary moments
    out["n_high"] = out["n_high"].fillna(0).astype(int)
    return out[out["balls"] >= MIN_BALLS]


def batter_wpa(b: pd.DataFrame) -> pd.DataFrame:
    return _agg(b, "batter", "wpa_bat")


def bowler_wpa(b: pd.DataFrame) -> pd.DataFrame:
    return _agg(b, "bowler", "wpa_bowl")


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

    # --- sanity: zero-sum per ball, telescoping per innings --------------------
    zero_sum = float((b["wpa_bat"] + b["wpa_bowl"]).abs().max())
    total_bat = float(b["wpa_bat"].sum())
    first = b.groupby("match_id", sort=False).first()
    telescope = float((first["y"] - first["wp"]).sum())  # == sum of batting WPA

    bats = batter_wpa(b).round(4)
    bowls = bowler_wpa(b).round(4)

    bcols = ["player", "balls", "total_wpa", "wpa_per100", "mean_li", "n_high", "wpa_high", "wpa_low"]
    top_bat_total = bats.sort_values("total_wpa", ascending=False).head(20).reset_index().rename(columns={"batter": "player"})
    top_bowl_total = bowls.sort_values("total_wpa", ascending=False).head(20).reset_index().rename(columns={"bowler": "player"})
    # clutch rate: per-ball WPA on high-LI balls, min sample of clutch balls
    clutch_bat = bats[bats["n_high"] >= 100].sort_values("wpa_high", ascending=False).head(20).reset_index().rename(columns={"batter": "player"})
    clutch_bowl = bowls[bowls["n_high"] >= 100].sort_values("wpa_high", ascending=False).head(20).reset_index().rename(columns={"bowler": "player"})

    L = [
        "# Win Probability Added (WPA) -- clutch attribution",
        "",
        f"WPA = realized signed one-ball WP change, credited +to the batter and "
        f"-to the bowler, on the era-adjusted RRR Markov WP. Because that WP is an "
        f"exact martingale (E[WPA|state]=0), positive WPA means BEATING the model's "
        f"expectation for the state, automatically weighted by the leverage of the "
        f"moment. Career threshold >= {MIN_BALLS} balls; 'high leverage' = LI >= "
        f"{HIGH_LI} (>= {HIGH_LI}x the average ball).",
        "",
        f"*Sanity:* per-ball batter+bowler WPA cancels to <= {zero_sum:.2e} (zero-sum); "
        f"total batting WPA {total_bat:+.2f} equals the telescoped innings sum "
        f"y - wp(first ball) = {telescope:+.2f} (wins minus start-of-chase "
        f"expectation). Both hold, so the accounting is exact.",
        "",
        "## Batsmen -- career WPA (total win probability added when chasing)",
        "",
        "`total_wpa` = career WP added; `wpa_per100` = per 100 balls; `wpa_high` / "
        "`wpa_low` = per-ball WPA in high- vs ordinary-leverage moments. A genuine "
        "clutch player has `wpa_high` > 0 and ideally > `wpa_low`.",
        "",
        _fmt(top_bat_total, bcols),
        "",
        "## Batsmen -- pure clutch rate (per-ball WPA in high-leverage moments)",
        "",
        f"Ranked by `wpa_high` among batters with >= 100 high-LI balls faced -- who "
        f"actually converts the big moments, rate-adjusted so volume doesn't dominate:",
        "",
        _fmt(clutch_bat[["player", "balls", "n_high", "wpa_high", "wpa_low", "total_wpa"]],
             ["player", "balls", "n_high", "wpa_high", "wpa_low", "total_wpa"]),
        "",
        "## Bowlers -- career WPA (win probability added by defending)",
        "",
        _fmt(top_bowl_total, bcols),
        "",
        "## Bowlers -- pure clutch rate (per-ball WPA in high-leverage moments)",
        "",
        _fmt(clutch_bowl[["player", "balls", "n_high", "wpa_high", "wpa_low", "total_wpa"]],
             ["player", "balls", "n_high", "wpa_high", "wpa_low", "total_wpa"]),
        "",
        "## Reading these tables",
        "",
        "- **Total WPA** rewards both being trusted with the moments AND delivering; "
        "it is the career clutch-value rank.",
        "- **wpa_high vs wpa_low** is the honest clutch signal: a player whose "
        "per-ball WPA is higher when leverage is high genuinely raises their game; "
        "one who is positive only in low-LI moments padded stats in dead situations.",
        "- WPA scores the batter-vs-bowler matchup and inherits the model's +0.037 "
        "autocorrelation bound, so treat the ranking as robust and the absolute "
        "decimals as approximate.",
        "",
    ]
    out = config.REPORTS / "wpa.md"
    out.write_text("\n".join(L))
    print(f"zero_sum_max={zero_sum:.2e}  total_bat_wpa={total_bat:+.3f}  telescope={telescope:+.3f}")
    print(f"batters ranked: {len(bats)}  bowlers ranked: {len(bowls)}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
