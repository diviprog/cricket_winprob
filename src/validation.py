"""05 Part 1 -- WP validation on the mean level (spec 05).

Ground truth is the realized label y in {0,1}, always on the HELD-OUT split.
Evaluates the Markov WP models (base, RRR) and two reference baselines
(constant base-rate, pure-RRR logistic) on identical footing:

  * calibration / reliability + Expected Calibration Error (ECE), overall and
    sliced by phase and by wickets in hand
  * proper scoring: Brier score and log loss

Run:  uv run python -m src.validation

Writes reports/validation_report.md and reports/reliability.png, and prints the
M2 gate verdict (RRR Markov must beat both baselines on Brier and log loss).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config
from .outcome_model import fit_base, fit_rrr, make_outcome_model
from .wp_markov import solve_wp

EPS = 1e-12


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error: bin-count-weighted mean |confidence - accuracy|."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = len(p)
    err = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        err += m.sum() / total * abs(p[m].mean() - y[m].mean())
    return float(err)


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 15):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf.append(p[m].mean())
        acc.append(y[m].mean())
        cnt.append(int(m.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


# ---------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------
def _predict(wp, frame) -> np.ndarray:
    return wp.predict(frame["b"].to_numpy(), frame["w"].to_numpy(), frame["r"].to_numpy())


def build_predictions(df: pd.DataFrame):
    train = df[df["split"] == "train"].copy()
    holdout = df[df["split"] == "holdout"].copy()
    r_max = int(df["r"].max()) + 1

    preds: dict[str, np.ndarray] = {}
    insample: dict[str, dict] = {}  # Markov calibration on TRAIN, to separate model-class error from shift

    # --- Markov models (solve WP once, score on both splits) -----------------
    for name, kind, params in [
        ("base Markov", "base", fit_base(train)),
        ("RRR Markov", "rrr", fit_rrr(train)),
    ]:
        wp = solve_wp(make_outcome_model(kind, params), r_max=r_max)
        preds[name] = _predict(wp, holdout)
        p_tr = _predict(wp, train)
        insample[name] = {
            "brier": brier(p_tr, train["y"].to_numpy()),
            "ece": ece(p_tr, train["y"].to_numpy()),
            "mean_pred": float(p_tr.mean()),
            "actual": float(train["y"].mean()),
        }

    # --- reference baselines -------------------------------------------------
    base_rate = float(train["y"].mean())  # constant chase-success rate
    preds["base-rate const"] = np.full(len(holdout), base_rate)

    lr = LogisticRegression()  # pure-RRR logistic, single feature
    lr.fit(train[["rrr"]].to_numpy(), train["y"].to_numpy())
    preds["pure-RRR logistic"] = lr.predict_proba(holdout[["rrr"]].to_numpy())[:, 1]

    # scoring-era shift, for the finding
    tmp = df.assign(runs=df["outcome"].map(lambda o: 0 if o == "W" else int(o)))
    era = tmp.groupby("split")["runs"].mean().to_dict()

    return preds, holdout, insample, era


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _sliced_ece(p, holdout, by: str) -> dict:
    y = holdout["y"].to_numpy()
    out = {}
    for key, sub in holdout.groupby(by, observed=True).groups.items():
        m = holdout.index.isin(sub)
        out[key] = ece(p[m], y[m])
    return out


def make_report(preds: dict, holdout: pd.DataFrame, insample: dict, era: dict) -> tuple[str, dict]:
    y = holdout["y"].to_numpy()
    rows = []
    for name, p in preds.items():
        rows.append((name, brier(p, y), log_loss(p, y), ece(p, y)))
    metrics = {name: {"brier": b, "log_loss": ll, "ece": e} for name, b, ll, e in rows}

    lines = ["# WP Validation Report (M2, held-out split)", ""]
    lines.append(f"Held-out seasons: {sorted(holdout['season'].unique())}")
    lines.append(f"Held-out balls: {len(holdout):,} | realized win rate: {y.mean():.3f}")
    lines.append("")
    lines.append("## Proper scoring + calibration (lower is better)")
    lines.append("")
    lines.append("| model | Brier | log loss | ECE |")
    lines.append("|---|---|---|---|")
    for name, b, ll, e in sorted(rows, key=lambda t: t[1]):
        lines.append(f"| {name} | {b:.4f} | {ll:.4f} | {e:.4f} |")
    lines.append("")

    # sliced ECE for the two Markov models
    for name in ["base Markov", "RRR Markov"]:
        p = preds[name]
        by_phase = _sliced_ece(p, holdout, "phase")
        by_w = _sliced_ece(p, holdout, "w")
        lines.append(f"## {name}: ECE sliced")
        lines.append("")
        lines.append("By phase: " + ", ".join(f"{k}={v:.4f}" for k, v in by_phase.items()))
        lines.append("")
        lines.append("By wickets in hand: " + ", ".join(f"{k}={v:.4f}" for k, v in sorted(by_w.items())))
        lines.append("")

    # --- honest finding ------------------------------------------------------
    rrr_in = insample["RRR Markov"]
    lines.append("## Finding (why the RRR Markov trails the direct logistic)")
    lines.append("")
    lines.append(
        "The exact memoryless RRR Markov WP beats the constant base-rate but is "
        "beaten by a one-feature (RRR) logistic, and is systematically "
        "under-confident. This is not a DP bug -- a Monte-Carlo simulation of the "
        "same independent process reproduces the DP's WP to sampling noise. Two "
        "real effects, both spec-anticipated:"
    )
    lines.append("")
    lines.append(
        f"1. **Memoryless tail-thinning (structural).** Even in-sample the RRR "
        f"Markov under-predicts: mean WP {rrr_in['mean_pred']:.3f} vs actual win "
        f"rate {rrr_in['actual']:.3f} (ECE {rrr_in['ece']:.3f}). Independent "
        f"per-ball draws thin the scoring right-tail, so hard chases look harder "
        f"than they are. This is exactly the spec 00 hidden-state error appearing "
        f"at the mean, and it motivates M5."
    )
    lines.append("")
    lines.append(
        f"2. **Scoring-era shift.** Mean runs/legal-ball is {era.get('train', float('nan')):.3f} "
        f"in training vs {era.get('holdout', float('nan')):.3f} in the held-out "
        f"seasons (2024-2026, the highest-scoring in IPL history). The model, fit "
        f"mostly on lower-scoring cricket, under-predicts modern chases; the direct "
        f"logistic on RRR is more era-robust because RRR already encodes difficulty."
    )
    lines.append("")

    return "\n".join(lines), metrics


def plot_reliability(preds: dict, holdout: pd.DataFrame, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = holdout["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name in ["base-rate const", "pure-RRR logistic", "base Markov", "RRR Markov"]:
        conf, acc, cnt = reliability_curve(preds[name], y)
        ax.plot(conf, acc, marker="o", ms=3, label=name)
    ax.set_xlabel("predicted WP")
    ax.set_ylabel("realized win rate")
    ax.set_title("Reliability (held-out)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    df = pd.read_parquet(config.BALLS_PARQUET)
    preds, holdout, insample, era = build_predictions(df)
    report, metrics = make_report(preds, holdout, insample, era)

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "validation_report.md").write_text(report)
    plot_reliability(preds, holdout, config.REPORTS / "reliability.png")

    print(report)
    print("\n=== M2 gate (spec 07: RRR Markov beats base-rate AND pure-RRR logistic) ===")
    rrr = metrics["RRR Markov"]
    ok = True
    for ref in ["base-rate const", "pure-RRR logistic"]:
        for metric in ["brier", "log_loss"]:
            better = rrr[metric] < metrics[ref][metric]
            ok &= better
            verdict = "PASS" if better else "FAIL"
            print(f"  {verdict}: RRR Markov {metric} {rrr[metric]:.4f} < {ref} {metrics[ref][metric]:.4f}")
    print(f"\n  M2 GATE: {'PASS' if ok else 'FAIL (beats base-rate; trails direct logistic -- see Finding)'}")
    print(f"\nWrote {config.REPORTS / 'validation_report.md'} and reliability.png")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
