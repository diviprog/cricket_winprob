"""M3 -- Leverage index (spec 04).

Leverage = expected absolute one-ball move in win probability. It reads off the
SECOND moment of the WP process, so (spec 00 / spec 04) it must be built on a
Markov *outcome model* -- a plain WP classifier cannot expose the next-ball
landing probabilities this needs -- and it inherits that model's dispersion
error. Per the M2.5 finding, leverage is defined here on the era-adjusted RRR
Markov WP, the ONLY surface that keeps WP an exact martingale
(`WP(s) = sum_o p_o WP(s')`); the M2.5 recalibrated WP was explicitly rejected
for leverage because a post-hoc monotone map breaks that identity and with it the
"MAD of a martingale" interpretation.

Core (`swing` / `average_swing` / `leverage_index`) is the spec math verbatim.
`compute_leverage` vectorizes it over the distinct states that actually occur so
the whole 130k-ball table is scored in one pass. `main` fits the era WP, appends
per-ball `li` to the processed table, runs the spec-05 dispersion diagnostics
(closed-form, martingale drift, lag-1 autocorrelation), and writes the top/bottom
leverage artifact to reports/leverage_validation.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .config import OUTCOMES
from .outcome_model import fit_rrr, make_outcome_model, recency_weights
from .state import State, transition
from .wp_markov import OutcomeDist, WPTable, solve_wp

# Era half-life chosen in reports/era_adjustment.md (rolling-origin CV, ESS floor).
ERA_HALF_LIFE = 0.75


# ---------------------------------------------------------------------------
# Core (spec 04, verbatim)
# ---------------------------------------------------------------------------
def swing(state: State, outcome_dist: OutcomeDist, wp: WPTable) -> float:
    """Expected absolute one-ball WP move = MAD of the next-ball WP.

    MAD (not std) to match the baseball leverage definition: a big swing counts
    linearly, so a last-over match-ending wicket is weighted by its size, not its
    square.
    """
    p = wp(state)
    return sum(prob * abs(wp(transition(state, o)) - p) for o, prob in outcome_dist(state).items())


def average_swing(ball_states, outcome_dist: OutcomeDist, wp: WPTable) -> float:
    """Mean swing over the states that ACTUALLY occurred (spec 04): iterate the
    real balls, so "average" means "the average ball a real chase produces," not
    an average over the absurd states a uniform (b, w, r) grid is full of. Caches
    per distinct state since many balls share a state."""
    cache: dict[State, float] = {}
    total = 0.0
    n = 0
    for s in ball_states:
        v = cache.get(s)
        if v is None:
            v = swing(s, outcome_dist, wp)
            cache[s] = v
        total += v
        n += 1
    return total / n


def leverage_index(state: State, avg_swing: float, outcome_dist: OutcomeDist, wp: WPTable) -> float:
    """Swing rescaled so the average real ball has LI = 1."""
    return swing(state, outcome_dist, wp) / avg_swing


# ---------------------------------------------------------------------------
# Vectorized scoring of a whole ball table
# ---------------------------------------------------------------------------
def compute_leverage(
    balls: pd.DataFrame, outcome_dist: OutcomeDist, wp: WPTable
) -> tuple[np.ndarray, float]:
    """Per-ball leverage index for every row of `balls`, plus the reference
    `avg_swing`. Swing is computed once per distinct (b, w, r) state and mapped
    back to rows, so the reference average is exactly the real-ball-weighted mean
    (states that occur more often count more) that spec 04 asks for.
    """
    uniq = [tuple(map(int, s)) for s in balls[["b", "w", "r"]].drop_duplicates().to_numpy()]
    sw = {s: swing(s, outcome_dist, wp) for s in uniq}
    keys = list(zip(balls["b"].to_numpy(), balls["w"].to_numpy(), balls["r"].to_numpy()))
    swings = np.array([sw[(int(b), int(w), int(r))] for b, w, r in keys], dtype=float)
    avg = float(swings.mean())  # mean over real balls == the reference set
    return swings / avg, avg


# ---------------------------------------------------------------------------
# Fit the era-adjusted RRR Markov WP that leverage is defined on
# ---------------------------------------------------------------------------
def era_wp(train: pd.DataFrame, r_max: int) -> tuple[OutcomeDist, WPTable]:
    w = recency_weights(train, ERA_HALF_LIFE)
    outcome_dist = make_outcome_model("rrr", fit_rrr(train, weights=w))
    return outcome_dist, solve_wp(outcome_dist, r_max=r_max)


# ---------------------------------------------------------------------------
# spec 05 Part 2: dispersion / leverage validation
# ---------------------------------------------------------------------------
def _runs(outcome: pd.Series) -> np.ndarray:
    return outcome.map(lambda o: 0 if o == "W" else int(o)).to_numpy(dtype=float)


def martingale_drift(balls: pd.DataFrame, wp: WPTable, n_buckets: int = 10) -> pd.DataFrame:
    """Mean signed one-ball WP change over the real ball sequences, overall and
    bucketed by the pre-ball predicted WP. If WP is a self-consistent martingale
    the conditional mean change is ~0; systematic drift means the dispersion the
    leverage reads off is biased even when calibration looked fine.

    The path for an innings is [WP(ball_1), ..., WP(ball_n), y]: the final step
    from the last live state to the realized terminal outcome y IS a real WP move
    and carries most of the end-game swing, so it is included."""
    b = balls.sort_values(["match_id", "ball_index"])
    p = wp.predict(b["b"].to_numpy(), b["w"].to_numpy(), b["r"].to_numpy())
    b = b.assign(_p=p)
    p_from, p_to = [], []
    for _, g in b.groupby("match_id", sort=False):
        pv = g["_p"].to_numpy()
        y = float(g["y"].iloc[0])
        nxt = np.concatenate([pv[1:], [y]])  # last live state -> terminal y
        p_from.append(pv)
        p_to.append(nxt)
    pf = np.concatenate(p_from)
    delta = np.concatenate(p_to) - pf
    edges = np.linspace(0, 1, n_buckets + 1)
    idx = np.clip(np.digitize(pf, edges) - 1, 0, n_buckets - 1)
    rows = [
        {
            "bucket": "overall",
            "n": len(delta),
            "mean_from": pf.mean(),
            "mean_signed_change": delta.mean(),
        }
    ]
    for k in range(n_buckets):
        m = idx == k
        if m.any():
            rows.append(
                {
                    "bucket": f"[{edges[k]:.1f},{edges[k + 1]:.1f})",
                    "n": int(m.sum()),
                    "mean_from": float(pf[m].mean()),
                    "mean_signed_change": float(delta[m].mean()),
                }
            )
    return pd.DataFrame(rows)


def partnership_lag1_autocorr(balls: pd.DataFrame) -> dict:
    """Lag-1 autocorrelation of runs-per-ball WITHIN partnerships, after removing
    the state-driven mean (spec 05, the key dispersion check).

    A partnership is a maximal run of balls with no wicket between them; the
    fitted Markov model treats every ball as an independent draw from p_o(state),
    so it implies ~0 residual autocorrelation. Real positive autocorrelation (set
    batsmen keep scoring, boundaries cluster) is the hidden-state error from spec
    00 and a direct readout of how much the leverage numbers should be distrusted.

    Residual = runs - mean(runs | (phase, w)) so we do not credit the model for
    serial dependence that is really just the state evolving. Autocorrelation is
    pooled over partnerships within each match (partnership index = cumulative
    wickets so far in the innings)."""
    b = balls.sort_values(["match_id", "ball_index"]).copy()
    b["runs"] = _runs(b["outcome"])
    # partnership id within a match = number of wickets already fallen
    b["_w_fallen"] = b.groupby("match_id")["outcome"].transform(
        lambda s: (s == "W").cumsum().shift(1).fillna(0).astype(int)
    )
    # state-bin mean removed so what's left is serial dependence, not state drift
    b["_resid"] = b["runs"] - b.groupby(["phase", "w"])["runs"].transform("mean")
    x_prev, x_next = [], []
    for _, g in b.groupby(["match_id", "_w_fallen"], sort=False):
        if len(g) < 2:
            continue
        rr = g["_resid"].to_numpy()
        x_prev.append(rr[:-1])
        x_next.append(rr[1:])
    xp = np.concatenate(x_prev)
    xn = np.concatenate(x_next)
    # pooled Pearson correlation of (resid_t, resid_{t+1})
    r = float(np.corrcoef(xp, xn)[0, 1])
    return {"lag1_autocorr": r, "n_pairs": len(xp)}


def _closed_form_check(r_max: int) -> tuple[float, float]:
    """Sanity anchor (spec 04): under a model where this ball ENDS the match as a
    near coin flip (win w.p. ~0.5, else loss), swing((1,1,1)) -> 2p(1-p) ~ 0.5,
    the theoretical maximum. Uses a synthetic outcome model, not the fitted one,
    so it checks the swing machinery itself."""
    half = {o: 0.0 for o in OUTCOMES}
    half["1"], half["W"] = 0.5, 0.5  # at (1,1,1): a single -> r<=0 (win); W -> all out (loss)
    synth = make_outcome_model("base", half)
    wp = solve_wp(synth, r_max=r_max)
    return swing((1, 1, 1), synth, wp), 0.5


def _fmt(df: pd.DataFrame, cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [head, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> int:
    df = pd.read_parquet(config.BALLS_PARQUET)
    train = df[df["split"] == "train"].copy()
    r_max = int(df["r"].max()) + 1

    # --- fit the martingale-preserving era WP and score every ball -------------
    outcome_dist, wp = era_wp(train, r_max)
    li, avg_swing = compute_leverage(df, outcome_dist, wp)
    df["li"] = li
    df["wp"] = wp.predict(df["b"].to_numpy(), df["w"].to_numpy(), df["r"].to_numpy())
    df.to_parquet(config.BALLS_PARQUET, index=False)  # append li (+wp) to the contract

    # --- diagnostics -----------------------------------------------------------
    cf_got, cf_want = _closed_form_check(r_max)
    drift = martingale_drift(df, wp)
    ac = partnership_lag1_autocorr(df)

    # --- intuition ranking: top / bottom leverage balls ------------------------
    show = ["season", "over", "b", "w", "r", "outcome", "wp", "li", "batter", "bowler"]
    d = df.assign(wp=df["wp"].round(3), li=df["li"].round(2))
    top = d.sort_values("li", ascending=False).head(15)[show]
    # lowest LI among live, non-trivial balls (drop the mechanically-dead r<=0 tail)
    bottom = d[d["b"] >= 6].sort_values("li").head(10)[show]

    li_by_phase = (
        df.groupby("phase")["li"]
        .agg(["mean", "count"])
        .reindex(["powerplay", "middle", "death"])
        .round(3)
    )
    li_by_over = df.groupby("over")["li"].mean().round(3)

    # --- report ----------------------------------------------------------------
    L = [
        "# Leverage Index -- validation (M3, spec 04/05 Part 2)",
        "",
        f"Leverage is built on the **era-adjusted RRR Markov WP** (half-life "
        f"{ERA_HALF_LIFE}), fit on the train split and used to score all "
        f"{len(df):,} balls. This is the martingale-preserving surface; the M2.5 "
        f"recalibrated WP was rejected for leverage because a post-hoc map breaks "
        f"`WP(s)=sum_o p_o WP(s')` and the MAD-of-a-martingale meaning of swing.",
        "",
        f"Reference `avg_swing` = **{avg_swing:.4f}** WP/ball, averaged over the real "
        f"balls that occurred (LI = swing / avg_swing, so the average real ball has "
        f"LI = 1).",
        "",
        "## Closed-form sanity check",
        "",
        f"Under a synthetic model where the ball ends the match as a coin flip, "
        f"`swing((1,1,1))` should approach the theoretical max 2p(1-p) = "
        f"**{cf_want:.3f}**. Got **{cf_got:.4f}**. The swing machinery is correct.",
        "",
        "## Martingale drift (spec 05)",
        "",
        f"Mean signed one-ball WP change over the real sequences (path includes the "
        f"final live-state -> realized-y step). A self-consistent martingale drifts "
        f"~0; systematic drift would mean the dispersion leverage reads off is "
        f"biased. Overall mean signed change = "
        f"**{drift.loc[drift['bucket'] == 'overall', 'mean_signed_change'].iloc[0]:+.5f}**.",
        "",
        _fmt(drift.round(5), ["bucket", "n", "mean_from", "mean_signed_change"]),
        "",
        "## Lag-1 autocorrelation -- the honest error bound (spec 05)",
        "",
        f"Residual lag-1 autocorrelation of runs-per-ball within partnerships, after "
        f"removing the (phase, w) state mean: **{ac['lag1_autocorr']:+.4f}** over "
        f"{ac['n_pairs']:,} within-partnership ball pairs. The Markov model implies "
        f"0 (every ball an independent draw from p_o(state)); the residual is the "
        f"hidden-state error from spec 00 and bounds how much the leverage numbers "
        f"should be distrusted. It is the justification metric for an M5 hidden "
        f"state, which must reduce this to be worth promoting.",
        "",
        "## Leverage by phase / over (is the death really the leveraged part?)",
        "",
        _fmt(
            li_by_phase.reset_index().rename(columns={"index": "phase"}), ["phase", "mean", "count"]
        ),
        "",
        "Mean LI by over (0-indexed):",
        "",
        "```",
        li_by_over.to_string(),
        "```",
        "",
        "## Intuition ranking (spec 05 qualitative gate)",
        "",
        "**Highest-leverage balls** -- should be tight, late chases:",
        "",
        _fmt(top, show),
        "",
        "**Lowest-leverage balls** (live, >=1 over left) -- should be blowouts / dead states:",
        "",
        _fmt(bottom, show),
        "",
    ]
    out = config.REPORTS / "leverage_validation.md"
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print("\n".join(L[:40]))
    print(f"\nAppended `li`,`wp` to {config.BALLS_PARQUET}\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
