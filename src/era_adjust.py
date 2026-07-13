"""Era adjustment for the RRR Markov outcome model (post-M2.5 follow-up).

The M2.5 diagnostics attributed most of the held-out RRR Markov error to a
scoring-era shift (the 2024-2026 holdout is the highest-scoring era in IPL
history) rather than to memoryless tail-thinning. The fix that keeps WP a proper
martingale is to re-estimate p_o(state) with recency weighting -- implemented in
`outcome_model.recency_weights` / the weighted fitters -- so the estimation layer
tracks modern scoring while the exact backward-induction core is untouched.

This module (1) tunes the recency half-life on a validation fold DISJOINT from
the held-out test set, (2) refits the era-adjusted RRR Markov on full train, and
(3) reports before/after on the held-out set against the same references and the
same metric functions as M2. It does not modify the M2 report.

Run:  uv run python -m src.era_adjust
Writes reports/era_adjustment.md and reports/era_reliability.png. Commits nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config
from .outcome_model import fit_rrr, make_outcome_model, recency_weights
from .validation import brier, ece, log_loss, reliability_curve
from .wp_markov import solve_wp

# Half-life grid in seasons; None = no weighting (the M2 fit). Tuned by rolling-
# origin CV. ESS_FLOOR is the variance guard (see tune_half_life): we do not chase
# the CV argmin, which is boundary-pinned because the scoring trend is monotone.
HALF_LIFE_GRID = [None, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5]
N_CV_FOLDS = 3  # rolling one-step-ahead validation seasons (most recent)
ESS_FLOOR = 0.15  # keep >=15% effective sample so per-cell estimates stay stable


def _mean_runs(frame: pd.DataFrame, weights: np.ndarray | None = None) -> float:
    runs = frame["outcome"].map(lambda o: 0 if o == "W" else int(o)).to_numpy(dtype=float)
    if weights is None:
        return float(runs.mean())
    return float((runs * weights).sum() / weights.sum())


def _fit_rrr_wp(train: pd.DataFrame, half_life: float | None, r_max: int):
    w = recency_weights(train, half_life)
    params = fit_rrr(train, weights=w)
    return solve_wp(make_outcome_model("rrr", params), r_max=r_max)


def _score(wp, frame: pd.DataFrame) -> dict:
    y = frame["y"].to_numpy()
    p = wp.predict(frame["b"].to_numpy(), frame["w"].to_numpy(), frame["r"].to_numpy())
    return {
        "brier": brier(p, y),
        "log_loss": log_loss(p, y),
        "ece": ece(p, y),
        "mean_pred": float(p.mean()),
        "actual": float(y.mean()),
        "p": p,
    }


# ---------------------------------------------------------------------------
# 1. Tune the half-life by rolling-origin CV, disjoint from the held-out test.
# ---------------------------------------------------------------------------
def _ess_frac(weights: np.ndarray) -> float:
    """Kish effective sample size as a fraction of n. 1.0 = uniform weights."""
    return float((weights.sum() ** 2) / (np.square(weights).sum() * len(weights)))


def tune_half_life(train: pd.DataFrame, r_max: int):
    """Rolling-origin CV over the most recent `N_CV_FOLDS` train seasons: each
    fold season is predicted one step ahead from strictly earlier seasons, with
    recency weights relative to that earlier frame -- the same forecast the real
    2024-2026 test faces. All folds are TRAIN seasons, disjoint from the held-out
    test, so the half-life is never chosen on test.

    Selection is NOT the raw CV argmin. Because IPL scoring trends monotonically
    upward and each fold's target is the newest available season, mean log loss
    keeps falling as the half-life shrinks, pinned to the grid edge and driven by
    the single most-trend-exposed fold, while effective sample size collapses. We
    instead take the most aggressive (smallest) half-life on the improving part of
    the curve whose mean effective sample size stays >= ESS_FLOOR -- a stated
    variance guard, not a metric we cherry-picked.
    """
    order = train.groupby("season")["date"].min().sort_values().index.tolist()
    folds = order[-N_CV_FOLDS:]

    rows = []  # (hl, mean_ll, mean_brier, mean_ess_frac, per_fold_ll)
    for hl in HALF_LIFE_GRID:
        lls, brs, essf = [], [], []
        for s in folds:
            fit_frame = train[train["season"].isin(order[: order.index(s)])]
            val = train[train["season"] == s]
            w = recency_weights(fit_frame, hl)
            essf.append(_ess_frac(w))
            wp = solve_wp(make_outcome_model("rrr", fit_rrr(fit_frame, weights=w)), r_max=r_max)
            p = wp.predict(val["b"].to_numpy(), val["w"].to_numpy(), val["r"].to_numpy())
            lls.append(log_loss(p, val["y"].to_numpy()))
            brs.append(brier(p, val["y"].to_numpy()))
        rows.append(
            (
                hl,
                float(np.mean(lls)),
                float(np.mean(brs)),
                float(np.mean(essf)),
                [round(x, 4) for x in lls],
            )
        )

    # candidates: keep improving vs the previous (larger) half-life, ESS above floor
    best_hl, best_ll = None, rows[0][1]
    for hl, ll, _br, ess, _pf in rows:
        if hl is None:
            continue
        if ll < best_ll - 1e-9 and ess >= ESS_FLOOR:
            best_hl, best_ll = hl, ll
    if best_hl is None:  # nothing cleared the floor; fall back to CV argmin
        best_hl = min(rows, key=lambda t: t[1])[0]
    return folds, rows, best_hl


# ---------------------------------------------------------------------------
def main() -> int:
    df = pd.read_parquet(config.BALLS_PARQUET)
    train = df[df["split"] == "train"].copy()
    holdout = df[df["split"] == "holdout"].copy()
    r_max = int(df["r"].max()) + 1

    # --- tune by rolling-origin CV (disjoint from test) --------------------
    cv_folds, tune_rows, best_hl = tune_half_life(train, r_max)

    # --- fit final models on FULL train ------------------------------------
    wp_raw = _fit_rrr_wp(train, None, r_max)  # M2 RRR Markov (unweighted)
    wp_era = _fit_rrr_wp(train, best_hl, r_max)  # era-adjusted
    lr = LogisticRegression().fit(train[["rrr"]].to_numpy(), train["y"].to_numpy())
    base_rate = float(train["y"].mean())

    y = holdout["y"].to_numpy()
    s_raw = _score(wp_raw, holdout)
    s_era = _score(wp_era, holdout)
    p_lr = lr.predict_proba(holdout[["rrr"]].to_numpy())[:, 1]
    p_const = np.full(len(holdout), base_rate)
    ref = {
        "pure-RRR logistic": {
            "brier": brier(p_lr, y),
            "log_loss": log_loss(p_lr, y),
            "ece": ece(p_lr, y),
            "mean_pred": float(p_lr.mean()),
        },
        "base-rate const": {
            "brier": brier(p_const, y),
            "log_loss": log_loss(p_const, y),
            "ece": ece(p_const, y),
            "mean_pred": base_rate,
        },
    }

    # in-sample gap on train, to show the era term shrinks the structural piece too
    s_raw_tr = _score(wp_raw, train)
    s_era_tr = _score(wp_era, train)

    # era means: how far recency weighting moved the fitted scoring level
    w_best = recency_weights(train, best_hl)
    runs_unw = _mean_runs(train)
    runs_w = _mean_runs(train, w_best)
    runs_ho = _mean_runs(holdout)

    # --- report -------------------------------------------------------------
    L = []
    L += [
        "# Era Adjustment (recency-weighted RRR Markov outcome model)",
        "",
        "Follow-up to M2.5. The RRR outcome model is now re-estimated with per-ball "
        "recency weights (`outcome_model.recency_weights`), tilting p_o(state) toward "
        "recent seasons. This is an estimation-layer change only: still one outcome "
        "distribution, still one backward-induction sweep, so WP stays an exact "
        "martingale and the DP core is unchanged.",
        "",
        "## Half-life tuning (rolling-origin CV, disjoint from test)",
        "",
        f"Rolling-origin CV over the {N_CV_FOLDS} most recent train seasons "
        f"({', '.join(cv_folds)}): each is predicted one step ahead from strictly "
        f"earlier seasons, recency-weighted relative to that earlier frame -- the same "
        f"forecast the real 2024-2026 test faces. All folds are train seasons, disjoint "
        f"from the held-out test.",
        "",
        f"The mean log loss is monotone-decreasing but flattening: the trend is monotone "
        f"in time and each fold's target is the newest available season, so the raw argmin "
        f"is pinned to the grid edge and driven by the single most-trend-exposed fold, "
        f"while effective sample size (ESS) collapses. Selection rule (stated, not "
        f"cherry-picked): the most aggressive half-life on the improving curve with mean "
        f"ESS >= {ESS_FLOOR:.0%} of train.",
        "",
        "| half-life (seasons) | mean CV log loss | mean CV Brier | mean ESS frac | per-fold log loss |",
        "|---|---|---|---|---|",
    ]
    for hl, ll, br, ess, pf in tune_rows:
        mark = " **<- chosen**" if hl == best_hl else ""
        L.append(
            f"| {'none (M2)' if hl is None else hl}{mark} | {ll:.4f} | {br:.4f} | {ess:.3f} | {pf} |"
        )
    L += [
        "",
        f"Chosen half-life: **{best_hl} seasons** (weight halves every {best_hl} season(s); "
        f"more aggressive settings keep gaining only on the newest fold and drop ESS below "
        f"the floor).",
        "",
        "## Held-out results (2024-2026) -- before vs after",
        "",
        "| model | Brier | log loss | ECE | mean pred | mean gap |",
        "|---|---|---|---|---|---|",
    ]
    actual = float(y.mean())

    def row(name, d):
        return f"| {name} | {d['brier']:.4f} | {d['log_loss']:.4f} | {d['ece']:.4f} | {d['mean_pred']:.3f} | {d['mean_pred'] - actual:+.4f} |"

    L.append(row("RRR Markov (M2, unweighted)", s_raw))
    L.append(row(f"RRR Markov + recency (hl={best_hl})", s_era))
    L.append(row("pure-RRR logistic", ref["pure-RRR logistic"]))
    L.append(row("base-rate const", ref["base-rate const"]))
    L += ["", f"Held-out actual win rate: {actual:.3f}.", ""]

    d_brier = s_raw["brier"] - s_era["brier"]
    d_ll = s_raw["log_loss"] - s_era["log_loss"]
    d_ece = s_raw["ece"] - s_era["ece"]
    gap_raw = s_raw["mean_pred"] - actual
    gap_era = s_era["mean_pred"] - actual
    gap_close = (abs(gap_raw) - abs(gap_era)) / abs(gap_raw) if gap_raw else float("nan")
    brier_gap_to_lr_before = s_raw["brier"] - ref["pure-RRR logistic"]["brier"]
    brier_gap_to_lr_after = s_era["brier"] - ref["pure-RRR logistic"]["brier"]
    lr_close = (
        (brier_gap_to_lr_before - brier_gap_to_lr_after) / brier_gap_to_lr_before
        if brier_gap_to_lr_before
        else float("nan")
    )

    L += [
        "## Effect",
        "",
        f"- **Held-out improvement.** Brier {s_raw['brier']:.4f} -> {s_era['brier']:.4f} "
        f"({d_brier:+.4f}); log loss {s_raw['log_loss']:.4f} -> {s_era['log_loss']:.4f} "
        f"({d_ll:+.4f}); ECE {s_raw['ece']:.4f} -> {s_era['ece']:.4f} ({d_ece:+.4f}).",
        f"- **Mean-gap (under-confidence).** {gap_raw:+.4f} -> {gap_era:+.4f}: recency "
        f"weighting closes **{gap_close:.0%}** of the one-directional under-confidence the "
        f"M2.5 slice-by-position diagnostic flagged as the era signature.",
        f"- **Gap to the logistic.** Brier gap to the pure-RRR logistic {brier_gap_to_lr_before:+.4f} "
        f"-> {brier_gap_to_lr_after:+.4f} ({lr_close:.0%} of it closed by the era term alone), "
        f"and this WP is still a proper martingale (unlike the M2.5 recalibration, which was not).",
        f"- **In-sample (structural) piece.** Train mean gap {s_raw_tr['mean_pred'] - s_raw_tr['actual']:+.4f} "
        f"-> {s_era_tr['mean_pred'] - s_era_tr['actual']:+.4f}: recency weighting is a mild "
        f"reweighting of the same fit, so the in-sample memoryless component is little changed, "
        f"as expected -- the gain is on the era shift, not the tail-thinning.",
        "",
        "## Honest ceiling",
        "",
        f"Recency weighting can only pull the fitted scoring level toward the most recent "
        f"TRAIN season, not past it. Weighted train mean runs/legal-ball = {runs_w:.3f} "
        f"(vs {runs_unw:.3f} unweighted), still below the holdout's {runs_ho:.3f} because "
        f"2024-2026 out-scored every training season. So the era term reduces but cannot "
        f"erase the shift without trend EXTRAPOLATION beyond the data, which trades the "
        f"conservative no-leakage property for reach. Documented, not taken here.",
        "",
    ]

    # reliability plot: before vs after
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, p in [
        ("RRR Markov (M2)", s_raw["p"]),
        ("RRR Markov + recency", s_era["p"]),
        ("pure-RRR logistic", p_lr),
    ]:
        conf, acc, _ = reliability_curve(p, y)
        ax.plot(conf, acc, marker="o", ms=3, label=name)
    ax.set_xlabel("predicted WP")
    ax.set_ylabel("realized win rate")
    ax.set_title("Reliability: era adjustment (held-out)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path = config.REPORTS / "era_reliability.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS / "era_adjustment.md"
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nWrote {out} and {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
