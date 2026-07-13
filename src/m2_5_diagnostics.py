"""M2.5 -- Pre-M3 diagnostics (analysis only).

Reads existing model outputs and adds analysis. Does NOT modify the exact core
(state.transition / wp_markov.terminal_value / solve_wp) or the outcome model.
It only *calls* them, and reuses the metric functions and the on-disk train /
held-out split from M2 (src.config / src.validation / src.outcome_model).

Five diagnostics (spec: M2.5):
  1. Brier (Murphy) decomposition: reliability / resolution / uncertainty.
  2. Post-hoc recalibration of the RRR Markov WP (isotonic + Platt), fit on a
     season fold disjoint from both the outcome-model training data and test.
  3. Calibration sliced by ahead/behind the required run rate (RRR Markov).
  4. In-sample vs held-out attribution (memoryless share vs era-shift share).
  5. Robustness to the split (random match-grouped, and a mid-history season).

Run:  uv run python -m src.m2_5_diagnostics
Writes reports/m2_5_diagnostics.md and reports/m2_5_*.png. Commits nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from . import config
from .outcome_model import fit_base, fit_rrr, make_outcome_model

# Reuse M2 metric functions verbatim -- identical footing to the M2 report.
from .validation import brier, ece, log_loss, reliability_curve
from .wp_markov import solve_wp

MODEL_ORDER = ["base Markov", "RRR Markov", "pure-RRR logistic", "base-rate const"]


# ---------------------------------------------------------------------------
# Shared model grid: fit the four models on an arbitrary train frame, predict
# on an arbitrary frame. This is the M2 build_predictions logic factored so it
# can run on any split (used by diagnostics 4 and 5). Same estimators, same
# features -- nothing new is modeled.
# ---------------------------------------------------------------------------
def fit_grid(train: pd.DataFrame, r_max: int) -> dict:
    wp_base = solve_wp(make_outcome_model("base", fit_base(train)), r_max=r_max)
    wp_rrr = solve_wp(make_outcome_model("rrr", fit_rrr(train)), r_max=r_max)
    base_rate = float(train["y"].mean())
    lr = LogisticRegression()
    lr.fit(train[["rrr"]].to_numpy(), train["y"].to_numpy())
    return {"wp_base": wp_base, "wp_rrr": wp_rrr, "base_rate": base_rate, "lr": lr}


def predict_grid(fitted: dict, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    b = frame["b"].to_numpy()
    w = frame["w"].to_numpy()
    r = frame["r"].to_numpy()
    return {
        "base Markov": fitted["wp_base"].predict(b, w, r),
        "RRR Markov": fitted["wp_rrr"].predict(b, w, r),
        "pure-RRR logistic": fitted["lr"].predict_proba(frame[["rrr"]].to_numpy())[:, 1],
        "base-rate const": np.full(len(frame), fitted["base_rate"]),
    }


def score_grid(preds: dict[str, np.ndarray], y: np.ndarray) -> dict[str, dict]:
    return {
        name: {"brier": brier(p, y), "log_loss": log_loss(p, y), "ece": ece(p, y)}
        for name, p in preds.items()
    }


def r_max_for(*frames: pd.DataFrame) -> int:
    return int(max(f["r"].max() for f in frames)) + 1


# ---------------------------------------------------------------------------
# Diagnostic 1 -- Brier (Murphy) decomposition
# ---------------------------------------------------------------------------
def brier_decomposition(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    """Murphy decomposition of the (binned) Brier score:

        Brier ~= Reliability - Resolution + Uncertainty

    Uncertainty = base_rate*(1-base_rate) is shared across models on a given
    eval set. Reliability is calibration error (lower better); Resolution is
    discriminative power (higher better). Binning matches ece()/reliability_curve().
    """
    ybar = float(y.mean())
    uncertainty = ybar * (1.0 - ybar)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    n = len(p)
    reliability = 0.0
    resolution = 0.0
    for bkt in range(n_bins):
        m = idx == bkt
        nb = int(m.sum())
        if nb == 0:
            continue
        pbar_k = float(p[m].mean())
        ybar_k = float(y[m].mean())
        reliability += nb / n * (pbar_k - ybar_k) ** 2
        resolution += nb / n * (ybar_k - ybar) ** 2

    return {
        "brier": brier(p, y),
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_recon": reliability - resolution + uncertainty,
    }


def diag1(preds: dict, y: np.ndarray) -> tuple[list[str], dict]:
    decomp = {name: brier_decomposition(preds[name], y) for name in MODEL_ORDER}
    lines = [
        "## 1. Brier (Murphy) decomposition",
        "",
        "Brier = Reliability - Resolution + Uncertainty (binned, 15 bins). "
        "Reliability = calibration error (lower better). Resolution = discriminative "
        "power (higher better). Uncertainty = base_rate*(1-base_rate), shared.",
        "",
        "| model | Brier | Reliability | Resolution | Uncertainty | Rel-Res+Unc |",
        "|---|---|---|---|---|---|",
    ]
    for name in MODEL_ORDER:
        d = decomp[name]
        lines.append(
            f"| {name} | {d['brier']:.4f} | {d['reliability']:.4f} | "
            f"{d['resolution']:.4f} | {d['uncertainty']:.4f} | {d['brier_recon']:.4f} |"
        )
    lines.append("")

    rrr = decomp["RRR Markov"]
    log = decomp["pure-RRR logistic"]
    res_gap = log["resolution"] - rrr["resolution"]
    rel_gap = rrr["reliability"] - log["reliability"]
    brier_gap = rrr["brier"] - log["brier"]
    rel_share = rel_gap / brier_gap if brier_gap else float("nan")
    lines += [
        f"**Hypothesis test (RRR Markov vs logistic).** Brier gap = {brier_gap:+.4f}. "
        f"Of this, reliability contributes {rel_gap:+.4f} and (lost) resolution "
        f"contributes {res_gap:+.4f}. Reliability accounts for {rel_share:.0%} of the gap. "
        f"RRR Markov resolution {rrr['resolution']:.4f} vs logistic {log['resolution']:.4f} "
        f"(difference {res_gap:+.4f}).",
        "",
    ]
    return lines, decomp


# ---------------------------------------------------------------------------
# Diagnostic 2 -- Post-hoc recalibration of the RRR Markov WP
# ---------------------------------------------------------------------------
def diag2(df: pd.DataFrame, cal_season: str = "2023") -> tuple[list[str], dict]:
    """Fit isotonic + Platt maps on a season fold disjoint from BOTH the
    outcome-model training data and the held-out test set.

    To keep the calibration fold truly disjoint from the WP model's training
    data, the RRR outcome model is REFIT on train-minus-cal_season, WP is
    predicted on cal_season to fit the maps, and the maps are then applied to
    test. `cal_season` (2023) is the most recent pre-holdout season.
    """
    train = df[df["split"] == "train"]
    test = df[df["split"] == "holdout"]
    cal = train[train["season"] == cal_season]
    train_cal = train[train["season"] != cal_season]
    rmax = r_max_for(train, test)

    # WP model + logistic refit WITHOUT the calibration season (no leakage).
    wp = solve_wp(make_outcome_model("rrr", fit_rrr(train_cal)), r_max=rmax)
    lr = LogisticRegression().fit(train_cal[["rrr"]].to_numpy(), train_cal["y"].to_numpy())

    p_cal = wp.predict(cal["b"].to_numpy(), cal["w"].to_numpy(), cal["r"].to_numpy())
    y_cal = cal["y"].to_numpy()

    iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal)
    platt = LogisticRegression().fit(p_cal.reshape(-1, 1), y_cal)

    y_te = test["y"].to_numpy()
    raw = wp.predict(test["b"].to_numpy(), test["w"].to_numpy(), test["r"].to_numpy())
    p_iso = iso.transform(raw)
    p_platt = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
    p_log = lr.predict_proba(test[["rrr"]].to_numpy())[:, 1]

    def m(p):
        return {"brier": brier(p, y_te), "log_loss": log_loss(p, y_te), "ece": ece(p, y_te)}

    rows = {
        "RRR Markov (raw, refit)": m(raw),
        "RRR Markov + isotonic": m(p_iso),
        "RRR Markov + Platt": m(p_platt),
        "pure-RRR logistic (refit)": m(p_log),
    }

    gap_before = (
        rows["RRR Markov (raw, refit)"]["brier"] - rows["pure-RRR logistic (refit)"]["brier"]
    )
    best_after = min(rows["RRR Markov + isotonic"]["brier"], rows["RRR Markov + Platt"]["brier"])
    gap_after = best_after - rows["pure-RRR logistic (refit)"]["brier"]
    closed = (gap_before - gap_after) / gap_before if gap_before else float("nan")

    lines = [
        "## 2. Post-hoc recalibration of the RRR Markov WP",
        "",
        f"Calibration fold: season **{cal_season}** ({len(cal):,} balls), disjoint from "
        f"both the outcome-model training data (refit on the other {len(train_cal):,} "
        f"train balls) and the held-out test set (2024-2026). Maps fit on the fold, "
        f"scored on test.",
        "",
        "| model | Brier | log loss | ECE |",
        "|---|---|---|---|",
    ]
    for name, d in rows.items():
        lines.append(f"| {name} | {d['brier']:.4f} | {d['log_loss']:.4f} | {d['ece']:.4f} |")
    lines += [
        "",
        f"**Gap closure.** Brier gap to the logistic before recalibration = "
        f"{gap_before:+.4f}; after the better map = {gap_after:+.4f}. Recalibration "
        f"alone closes **{closed:.0%}** of the Brier gap. ECE falls from "
        f"{rows['RRR Markov (raw, refit)']['ece']:.4f} (raw) to "
        f"{min(rows['RRR Markov + isotonic']['ece'], rows['RRR Markov + Platt']['ece']):.4f} "
        f"(best map).",
        "",
        "> **Martingale caveat (flag, do not resolve here).** Pointwise recalibration "
        "applies a monotone map to WP that is NOT derived from the outcome model, so "
        "the recalibrated WP no longer satisfies WP(s) = sum_o p_o WP(s') w.r.t. the "
        "outcome distribution. It breaks the martingale consistency between the outcome "
        "model and WP. Therefore leverage in M3 (swing = MAD of the next-ball WP under "
        "the martingale) cannot use the recalibrated WP without losing the clean "
        '"MAD of a martingale" interpretation. This is a real tradeoff for M3, left open.',
        "",
    ]
    return lines, {"rows": rows, "closed": closed, "gap_before": gap_before, "gap_after": gap_after}


# ---------------------------------------------------------------------------
# Diagnostic 3 -- calibration sliced by ahead/behind the required run rate
# ---------------------------------------------------------------------------
def _ahead_of_required(frame: pd.DataFrame) -> np.ndarray:
    """Boolean: is the batting side AHEAD of the required run rate?

    Definition: sign of (current run rate - required run rate).
      CRR = 6 * runs_scored / balls_faced,   RRR = 6 * r / b  (the `rrr` column)
    runs_scored = r_start - r  (r_start = runs required at the first ball of the
    innings = max r within the match_id, since r never increases). balls_faced =
    120 - b. Ahead iff CRR > RRR, cross-multiplied to avoid division:
        runs_scored * b  >  r * balls_faced
    First-ball edge case: balls_faced = 0 => CRR = 0 < RRR, so the opening ball is
    classified BEHIND (the side has scored nothing yet against a positive target).
    """
    r_start = frame.groupby("match_id")["r"].transform("max").to_numpy()
    r = frame["r"].to_numpy()
    b = frame["b"].to_numpy()
    runs_scored = r_start - r
    balls_faced = config.BALLS_PER_INNINGS - b
    return runs_scored * b > r * balls_faced


def diag3(df: pd.DataFrame, fitted_full: dict, plot_path) -> tuple[list[str], dict]:
    test = df[df["split"] == "holdout"].copy()
    p = predict_grid(fitted_full, test)["RRR Markov"]
    y = test["y"].to_numpy()
    ahead = _ahead_of_required(test)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    stats = {}
    for label, mask in [("ahead of RRR", ahead), ("behind RRR", ~ahead)]:
        conf, acc, cnt = reliability_curve(p[mask], y[mask])
        ax.plot(conf, acc, marker="o", ms=4, label=f"{label} (n={int(mask.sum()):,})")
        stats[label] = {
            "n": int(mask.sum()),
            "mean_pred": float(p[mask].mean()),
            "actual": float(y[mask].mean()),
            "ece": ece(p[mask], y[mask]),
            # fraction of occupied bins where realized > predicted (curve above diagonal)
            "frac_above_diag": float(np.mean(acc > conf)) if len(acc) else float("nan"),
        }
    ax.set_xlabel("predicted WP (RRR Markov)")
    ax.set_ylabel("realized win rate")
    ax.set_title("Reliability by position vs required rate (held-out)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    a, bh = stats["ahead of RRR"], stats["behind RRR"]
    # tail-thinning predicts: under-confident when behind (above diagonal), OVER-confident
    # when ahead (below diagonal). If above diagonal in BOTH slices -> one-directional bias.
    both_under = a["actual"] > a["mean_pred"] and bh["actual"] > bh["mean_pred"]
    pattern = (
        "under-confident in BOTH slices (realized > predicted on both sides) -> a "
        "ONE-DIRECTIONAL bias, consistent with an era/scoring shift dominating, NOT the "
        "symmetric signature of memoryless tail-thinning"
        if both_under
        else "directionally split (under when behind, over when ahead) -> consistent with "
        "memoryless variance tail-thinning"
    )
    lines = [
        "## 3. Calibration sliced by ahead/behind the required run rate",
        "",
        "Slice = sign(CRR - RRR), CRR = 6*runs_scored/balls_faced, runs_scored = "
        "r_start - r, r_start = max r in the innings. First ball (balls_faced=0) -> "
        "CRR=0 -> classified behind. Curves for the RRR Markov, held-out set. See "
        f"`{plot_path.name}`.",
        "",
        "| slice | n | mean pred | actual | ECE | bins with realized>pred |",
        "|---|---|---|---|---|---|",
        f"| ahead of RRR | {a['n']:,} | {a['mean_pred']:.3f} | {a['actual']:.3f} | "
        f"{a['ece']:.4f} | {a['frac_above_diag']:.0%} |",
        f"| behind RRR | {bh['n']:,} | {bh['mean_pred']:.3f} | {bh['actual']:.3f} | "
        f"{bh['ece']:.4f} | {bh['frac_above_diag']:.0%} |",
        "",
        f"**Pattern.** The reliability curve is {pattern}. "
        f"Tail-thinning predicts under-confidence when behind (an upper-tail win) and "
        f"OVER-confidence when ahead (a lower-tail collapse). Observed: ahead slice "
        f"realized {a['actual']:.3f} vs predicted {a['mean_pred']:.3f} "
        f"({'under' if a['actual'] > a['mean_pred'] else 'over'}-confident); behind slice "
        f"realized {bh['actual']:.3f} vs predicted {bh['mean_pred']:.3f} "
        f"({'under' if bh['actual'] > bh['mean_pred'] else 'over'}-confident).",
        "",
    ]
    return lines, {"stats": stats, "both_under": both_under}


# ---------------------------------------------------------------------------
# Diagnostic 4 -- in-sample vs held-out attribution
# ---------------------------------------------------------------------------
def diag4(df: pd.DataFrame, fitted_full: dict) -> tuple[list[str], dict]:
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "holdout"].copy()
    p_tr = predict_grid(fitted_full, train)
    p_te = predict_grid(fitted_full, test)
    y_tr, y_te = train["y"].to_numpy(), test["y"].to_numpy()

    rows = {}
    for name in MODEL_ORDER:
        in_gap = float(p_tr[name].mean() - y_tr.mean())  # mean pred - mean actual, TRAIN
        out_gap = float(p_te[name].mean() - y_te.mean())  # ... HELD-OUT
        rows[name] = {
            "in_brier": brier(p_tr[name], y_tr),
            "out_brier": brier(p_te[name], y_te),
            "in_ll": log_loss(p_tr[name], y_tr),
            "out_ll": log_loss(p_te[name], y_te),
            "in_gap": in_gap,
            "out_gap": out_gap,
            "widening": out_gap - in_gap,
        }

    lines = [
        "## 4. In-sample vs held-out attribution",
        "",
        "Gap = mean(predicted WP) - mean(realized y). In-sample gap has no "
        "distribution shift, so it isolates the structural memoryless error "
        "(tail-thinning). Train->test widening isolates the era shift.",
        "",
        "| model | in Brier | out Brier | in gap | out gap | widening (out-in) |",
        "|---|---|---|---|---|---|",
    ]
    for name in MODEL_ORDER:
        d = rows[name]
        lines.append(
            f"| {name} | {d['in_brier']:.4f} | {d['out_brier']:.4f} | "
            f"{d['in_gap']:+.4f} | {d['out_gap']:+.4f} | {d['widening']:+.4f} |"
        )

    rrr = rows["RRR Markov"]
    log = rows["pure-RRR logistic"]
    total = abs(rrr["out_gap"])
    mem_share = abs(rrr["in_gap"]) / total if total else float("nan")
    era_share = abs(rrr["widening"]) / total if total else float("nan")
    robust_claim = abs(log["widening"]) < 0.5 * abs(rrr["widening"])
    lines += [
        "",
        f"**RRR Markov attribution.** Total held-out mean gap {rrr['out_gap']:+.4f} = "
        f"in-sample (memoryless) {rrr['in_gap']:+.4f} + widening (era shift) "
        f"{rrr['widening']:+.4f}. Memoryless share ~= **{mem_share:.0%}**, era-shift "
        f"share ~= **{era_share:.0%}** of the held-out mean miscalibration.",
        "",
        f"**Logistic era-robustness claim.** Logistic train->test widening "
        f"{log['widening']:+.4f} vs RRR Markov {rrr['widening']:+.4f}. "
        + (
            f"The logistic widens far less (|{log['widening']:.4f}| < "
            f"0.5*|{rrr['widening']:.4f}|) -> claim SUPPORTED: the logistic is materially "
            f"more era-robust."
            if robust_claim
            else "The logistic widening is NOT much smaller than the Markov's -> claim "
            "NOT supported; both degrade similarly under the era shift."
        ),
        "",
    ]
    return lines, {
        "rows": rows,
        "mem_share": mem_share,
        "era_share": era_share,
        "robust_claim": robust_claim,
    }


# ---------------------------------------------------------------------------
# Diagnostic 5 -- robustness to the split
# ---------------------------------------------------------------------------
def _grid_gap(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, float]:
    rmax = r_max_for(train, test)
    fitted = fit_grid(train, rmax)
    preds = predict_grid(fitted, test)
    scored = score_grid(preds, test["y"].to_numpy())
    gap = scored["RRR Markov"]["brier"] - scored["pure-RRR logistic"]["brier"]
    return scored, gap


def diag5(df: pd.DataFrame, mid_season: str = "2016") -> tuple[list[str], dict]:
    results = {}

    # Reference: the M2 season holdout (2024-2026).
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "holdout"]
    results["season holdout 2024-26 (M2)"] = _grid_gap(tr, te)

    # (a) random match-grouped 80/20 (GroupKFold by match_id; deterministic first fold).
    gkf = GroupKFold(n_splits=5)
    tr_idx, te_idx = next(gkf.split(df, df["y"], groups=df["match_id"]))
    results["random match-grouped 80/20"] = _grid_gap(df.iloc[tr_idx], df.iloc[te_idx])

    # (b) single mid-history season held out.
    if mid_season in set(df["season"]):
        te_m = df[df["season"] == mid_season]
        tr_m = df[df["season"] != mid_season]
        results[f"mid-history season {mid_season}"] = _grid_gap(tr_m, te_m)

    lines = [
        "## 5. Robustness to the split",
        "",
        "Four-model grid re-scored under alternative splits. The M2 holdout "
        "(2024-2026) is the highest-scoring era in IPL history, which maximally "
        "penalizes a stationary compounding model. If the RRR-Markov-minus-logistic "
        "Brier gap shrinks off that holdout, era shift is the dominant driver.",
        "",
        "| split | RRR Markov Brier | logistic Brier | gap (Markov-logistic) |",
        "|---|---|---|---|",
    ]
    for name, (scored, gap) in results.items():
        lines.append(
            f"| {name} | {scored['RRR Markov']['brier']:.4f} | "
            f"{scored['pure-RRR logistic']['brier']:.4f} | {gap:+.4f} |"
        )

    ref_gap = results["season holdout 2024-26 (M2)"][1]
    alt_gaps = {k: v[1] for k, v in results.items() if k != "season holdout 2024-26 (M2)"}
    mean_alt = float(np.mean(list(alt_gaps.values())))
    shrink = (ref_gap - mean_alt) / ref_gap if ref_gap else float("nan")
    lines += [
        "",
        f"**Verdict.** Gap on the 2024-26 holdout = {ref_gap:+.4f}; mean gap on the "
        f"non-era splits = {mean_alt:+.4f}, i.e. the gap shrinks by **{shrink:.0%}** off "
        f"the high-scoring holdout. "
        + (
            "This independently confirms era shift as a dominant driver of the "
            "RRR-Markov-vs-logistic gap."
            if shrink > 0.25
            else "The gap does not shrink materially off the era holdout, so era shift is "
            "NOT the dominant driver."
        ),
        "",
    ]
    return lines, {"results": {k: v[1] for k, v in results.items()}, "shrink": shrink}


# ---------------------------------------------------------------------------
# Findings + recommendation
# ---------------------------------------------------------------------------
def findings(d1, d2, d3, d4, d5) -> list[str]:
    decomp = d1
    rrr, log = decomp["RRR Markov"], decomp["pure-RRR logistic"]
    rel_gap = rrr["reliability"] - log["reliability"]
    res_gap = log["resolution"] - rrr["resolution"]
    brier_gap = rrr["brier"] - log["brier"]

    lines = [
        "## Findings",
        "",
        "**(a) Reliability vs resolution, per model.** "
        f"RRR Markov: reliability {rrr['reliability']:.4f}, resolution {rrr['resolution']:.4f}. "
        f"Logistic: reliability {log['reliability']:.4f}, resolution {log['resolution']:.4f}. "
        f"base Markov: reliability {decomp['base Markov']['reliability']:.4f}, resolution "
        f"{decomp['base Markov']['resolution']:.4f}. base-rate const: reliability "
        f"{decomp['base-rate const']['reliability']:.4f}, resolution "
        f"{decomp['base-rate const']['resolution']:.4f}. "
        f"The RRR Markov's Brier gap to the logistic ({brier_gap:+.4f}) is "
        f"{rel_gap / brier_gap:.0%} reliability ({rel_gap:+.4f}) and only {res_gap:+.4f} "
        f"lost resolution: it discriminates almost as well as the logistic and is chiefly "
        f"MIScalibrated, not under-powered.",
        "",
        "**(b) How much recalibration alone closes.** "
        f"An isotonic/Platt map fit on a disjoint season fold closes "
        f"**{d2['closed']:.0%}** of the Brier gap to the logistic "
        f"(gap {d2['gap_before']:+.4f} -> {d2['gap_after']:+.4f}), confirming (a): most "
        f"of the deficit is fixable by calibration without changing the model class. But "
        f"recalibration breaks the outcome-model<->WP martingale, so it is not free for M3.",
        "",
        "**(c) Attribution: memoryless tail-thinning vs era shift.** "
        f"In-sample the RRR Markov mean gap is {d4['rows']['RRR Markov']['in_gap']:+.4f} "
        f"(the pure structural/memoryless error, no shift), widening by "
        f"{d4['rows']['RRR Markov']['widening']:+.4f} on the held-out era. That splits the "
        f"held-out mean miscalibration into ~**{d4['mem_share']:.0%} memoryless** and "
        f"~**{d4['era_share']:.0%} era shift**. Diagnostic 3 shows the held-out miscalibration "
        f"is {'one-directional (under-confident whether ahead or behind)' if d3['both_under'] else 'directionally split'}, "
        f"and diagnostic 5 shows the Markov-vs-logistic gap shrinks {d5['shrink']:.0%} off the "
        f"2024-26 holdout. "
        + (
            "Both independently point to era shift as the larger, and the directionally "
            "dominant, driver of the held-out gap."
            if d3["both_under"] and d5["shrink"] > 0.25
            else "The two directional tests are mixed; weight them against the mean-gap split above."
        ),
        "",
    ]

    # Recommendation: single, driven by the numbers.
    era_dominant = d3["both_under"] and d5["shrink"] > 0.25 and d4["era_share"] >= d4["mem_share"]
    recal_sufficient = d2["closed"] >= 0.75
    if era_dominant:
        rec = (
            "**Recommendation: an era-adjustment to the outcome estimation "
            "(recency weighting / an era term), before M3.** The evidence is convergent: "
            f"(1) the gap to the logistic is mostly reliability, not resolution; (2) the "
            f"held-out miscalibration is one-directional under-confidence in both the "
            f"ahead and behind slices -- the signature of a level shift, not the symmetric "
            f"ahead-over / behind-under signature of memoryless tail-thinning; (3) the "
            f"Markov-vs-logistic Brier gap shrinks ~{d5['shrink']:.0%} once the high-scoring "
            f"2024-26 seasons are no longer the entire test set; and (4) the mean-gap "
            f"attribution puts ~{d4['era_share']:.0%} of the held-out error on the shift. "
            "M5 (hidden striker state) targets tail-thinning, which these tests show is the "
            "SMALLER, and wrong-directional, component -- it would not fix the dominant "
            "under-confidence. Cheap recalibration removes most of the metric gap but "
            f"breaks the martingale that M3 leverage depends on, so it is a patch, not the "
            "fix. Re-estimate p_o(state) with recency weighting (or an explicit era term) "
            "so WP stays a proper martingale AND tracks modern scoring; then proceed to M3. "
            "Defer M5 until after the era term, and only if residual tail-clustering remains."
        )
    elif recal_sufficient:
        rec = (
            "**Recommendation: cheap recalibration is sufficient before M3.** Recalibration "
            f"closes {d2['closed']:.0%} of the gap and the residual is small; adopt a "
            "calibrated WP for reporting, but keep the raw martingale WP for M3 leverage and "
            "document the split."
        )
    else:
        rec = (
            "**Recommendation: M5 (hidden striker state).** The dominant component is the "
            "in-sample memoryless error with the directional signature of tail-thinning, "
            "which a hidden-state refit targets directly; era shift is secondary."
        )
    lines += ["## Recommendation", "", rec, ""]
    return lines


# ---------------------------------------------------------------------------
def main() -> int:
    df = pd.read_parquet(config.BALLS_PARQUET)
    test = df[df["split"] == "holdout"].copy()
    train = df[df["split"] == "train"].copy()
    rmax = r_max_for(df)
    fitted_full = fit_grid(train, rmax)  # models fit on full train, as in M2
    preds_test = predict_grid(fitted_full, test)
    y_te = test["y"].to_numpy()

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    d3_plot = config.REPORTS / "m2_5_reliability_by_position.png"

    l1, r1 = diag1(preds_test, y_te)
    l2, r2 = diag2(df)
    l3, r3 = diag3(df, fitted_full, d3_plot)
    l4, r4 = diag4(df, fitted_full)
    l5, r5 = diag5(df)
    lf = findings(r1, r2, r3, r4, r5)

    header = [
        "# M2.5 Pre-M3 Diagnostics",
        "",
        f"Held-out test: {sorted(test['season'].unique())} | {len(test):,} balls | "
        f"realized win rate {y_te.mean():.3f}. Train: {len(train):,} balls.",
        "",
        "Analysis only -- reuses the M2 on-disk split, the M2 metric functions "
        "(`brier`/`log_loss`/`ece`/`reliability_curve`), and the existing "
        "fitters/solver. The exact core and outcome model are unchanged.",
        "",
    ]
    report = "\n".join(header + l1 + l2 + l3 + l4 + l5 + lf)
    out = config.REPORTS / "m2_5_diagnostics.md"
    out.write_text(report)
    print(report)
    print(f"\nWrote {out} and {d3_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
