"""Render the paper's F3 and F4 figures from the committed reports.

Data is PARSED from reports/dependence_decomposition.md and
reports/correlation_experiment.md rather than recomputed: the reports are the
committed source of truth (regenerating the correlation experiment takes ~35
minutes), and parsing keeps figure and text numbers identical by construction.

Design (print / arXiv, grayscale-safe): near-black observed line with markers,
neutral gray shaded bands, identity carried by linestyle + direct labels, no
legend boxes, recessive grid, no top/right spines. Output: PNG (preview) + PDF
(vector, for the LaTeX pass) into paper/figures/.

Run:  .venv/bin/python paper/make_figures.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUT = ROOT / "paper" / "figures"

INK = "#1a1a1a"
BAND = "#c7c7c7"
BAND_EDGE = "#8a8a8a"
GRID = "#e3e3e3"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.8,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "text.color": INK,
    "axes.labelcolor": INK,
})


def _rows(md: str, section: str, n_cols: int, drop: int = 2) -> list[list[str]]:
    """Cells of the first markdown table after `section`.

    `drop` removes leading header/separator rows -- but beware headers whose
    CELLS contain pipes (e.g. `mean |gap|`): those split into extra columns and
    fail the n_cols filter on their own, so callers with such headers should
    pass drop accordingly and filter rows by content."""
    body = md.split(section, 1)[1]
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if rows and not line.startswith("|"):
            break
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == n_cols:
                rows.append(cells)
    return rows[drop:]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


# ---------------------------------------------------------------------------
# F3 -- lag profile of run residual autocorrelation vs permutation-null band
# ---------------------------------------------------------------------------
def make_f3() -> Path:
    md = (REPORTS / "dependence_decomposition.md").read_text()
    rows = _rows(md, "### state-demeaned (the M3 construction)", 6)
    lag = [int(r[0]) for r in rows]
    obs = [float(r[1]) for r in rows]
    nmean = [float(r[2]) for r in rows]
    nlo = [float(r[3]) for r in rows]
    nhi = [float(r[4]) for r in rows]

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.fill_between(lag, nlo, nhi, color=BAND, alpha=0.55, linewidth=0,
                    zorder=1)
    ax.plot(lag, nmean, color=BAND_EDGE, linewidth=1.1, linestyle="--",
            zorder=2)
    ax.plot(lag, obs, color=INK, linewidth=1.6, marker="o", markersize=4.5,
            zorder=3)
    ax.axhline(0, color="#999999", linewidth=0.7, linestyle=":", zorder=0)

    ax.set_xscale("log")
    ax.set_xticks(lag)
    ax.set_xticklabels([str(v) for v in lag])
    ax.minorticks_off()
    ax.set_xlabel("lag (balls)")
    ax.set_ylabel("autocorrelation of run residuals")

    # direct labels instead of a legend box
    ax.annotate("observed", xy=(lag[1], obs[1]), xytext=(2.3, 0.040),
                fontsize=9, color=INK)
    ax.annotate("permutation null: mean, 95% band\n"
                "(= innings-composition component)",
                xy=(20, nhi[5]), xytext=(9.5, 0.0305), fontsize=8,
                color="#555555",
                arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.7))
    ax.annotate("sequential excess,\n~3–5-ball range",
                xy=(2, 0.030), xytext=(1.5, 0.0185), fontsize=8,
                color="#555555")

    _style(ax)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "F3_lag_profile.png"
    fig.savefig(png, dpi=200)
    fig.savefig(OUT / "F3_lag_profile.pdf")
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# F4 -- mean absolute calibration gap vs block length K (6-seed bands)
# ---------------------------------------------------------------------------
def make_f4() -> Path:
    md = (REPORTS / "correlation_experiment.md").read_text()
    # header cells contain pipes (`mean |gap|`), so the header self-excludes on
    # column count; keep every 4-col row (drop=0) and filter by mode name.
    rows = _rows(md, "## Replicated mean |gap| by mode", 4, drop=0)
    ks, mean, lo, hi = [], [], [], []
    dp_ref = None
    for r in rows:
        if r[0] == "markov":
            dp_ref = float(r[1])
        m = re.fullmatch(r"block_K(\d+)", r[0])
        if not m:
            continue
        ks.append(int(m.group(1)))
        mean.append(float(r[1]))
        a, b = r[2].split("..")
        lo.append(float(a))
        hi.append(float(b))
    assert dp_ref is not None and len(ks) == 7, "report table parse failed"

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.fill_between(ks, lo, hi, color=BAND, alpha=0.55, linewidth=0, zorder=1)
    ax.plot(ks, mean, color=INK, linewidth=1.6, marker="o", markersize=4.5,
            zorder=3)
    ax.axhline(dp_ref, color="#777777", linewidth=1.0, linestyle="--",
               zorder=2)
    ax.annotate("exact DP (independent balls)", xy=(25, dp_ref),
                xytext=(22, dp_ref + 0.0008), fontsize=8, color="#555555")

    # highlight the replicated minimum
    i_min = mean.index(min(mean))
    ax.plot([ks[i_min]], [mean[i_min]], marker="o", markersize=8,
            markerfacecolor="white", markeredgecolor=INK,
            markeredgewidth=1.4, zorder=4)
    ax.annotate(f"K={ks[i_min]}: −26% vs independence\n(minimum in 6/6 seeds)",
                xy=(ks[i_min], mean[i_min]),
                xytext=(ks[i_min] * 1.35, mean[i_min] - 0.0022), fontsize=8,
                color="#555555",
                arrowprops=dict(arrowstyle="-", color="#999999",
                                linewidth=0.7))
    ax.annotate("min–max over 6 seeds", xy=(60, hi[5]),
                xytext=(34, 0.0545), fontsize=8, color="#555555")

    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(v) for v in ks])
    ax.minorticks_off()
    ax.set_xlabel("block length K (balls of real dependence honoured)")
    ax.set_ylabel("mean |calibration gap|")

    _style(ax)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "F4_gap_vs_K.png"
    fig.savefig(png, dpi=200)
    fig.savefig(OUT / "F4_gap_vs_K.pdf")
    plt.close(fig)
    return png


if __name__ == "__main__":
    p3 = make_f3()
    p4 = make_f4()
    print(f"Wrote {p3}\nWrote {p4}")
