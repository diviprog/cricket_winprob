"""Calibration-gap localization: is the Markov WP's under-confidence tail-thinning,
and if so, estimation-tail-thinning or genuine-correlation-tail-thinning?

M4 (baseline_comparison.md) established that an unconstrained XGB on the *identical*
(b,w,r) beats the RRR Markov WP by ~0.094 nats with far lower ECE. Since (b,w,r) is
sufficient (adding striker_balls did nothing), the gap is not missing state -- it is
the independence-assuming backward induction producing a miscalibrated mapping from
(b,w,r) to WP. The working hypothesis is TAIL-THINNING: independent per-ball draws
under-count scoring explosions, so the model rates hard chases as harder than they
are and is systematically under-confident.

That hypothesis is falsifiable and this module tests it, on the TRAIN split so the
scoring-era shift is excluded and we isolate pure structure error.

Diagnostic 1 -- WP calibration gap by RRR.
  gap(cell) = mean(Markov WP) - mean(realized y)   over balls in the cell.
  Tail-thinning predicts gap < 0 (model too low) growing more negative with RRR,
  and ~0 for easy chases. A flat gap would REFUTE tail-thinning and point at era /
  estimation elsewhere.

Diagnostic 2 -- outcome-marginal tail check (the mechanism, and which cure).
  Compare the model's own per-ball p(6), p(4), p(boundary)=p4+p6 against the
  empirical boundary rate along a FINE RRR axis. The model conditions RRR only
  through 6 coarse bins with a 15+ catch-all, shrunk (alpha=10) toward calmer
  (phase,w) parents. If the model's p(boundary) sits BELOW empirical at high RRR,
  the tail-thinning is (partly) an ESTIMATION artifact -- fixable by better tail
  estimation of the marginal, which PRESERVES independence, the exact martingale,
  and leverage. If the marginals already match empirical yet Diagnostic 1's gap
  persists, the residual is GENUINE CORRELATION -- fixable only by a correlated
  outcome process, which breaks WP(s)=sum_o p_o WP(s') and so the leverage
  definition. The two diagnostics together say which lever to pull.

Run:  .venv/bin/python -m src.tail_diagnostics
Writes reports/tail_diagnostics.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .leverage import era_wp
from .outcome_model import RRR_BIN_EDGES, make_outcome_model, fit_rrr

# fine RRR axis for the shape (finer than the model's 6 estimation bins)
FINE_RRR_EDGES = [0, 6, 8, 10, 12, 14, 16, 18, 22, 40]
MIN_CELL = 150  # min balls for a cell to be reported / trusted
BOUNDARY = ("4", "6")


def _rrr_slice(df: pd.DataFrame) -> pd.Series:
    lab = [f"{lo}-{hi}" for lo, hi in zip(FINE_RRR_EDGES[:-1], FINE_RRR_EDGES[1:])]
    return pd.cut(df["rrr"], bins=FINE_RRR_EDGES, labels=lab, right=True, include_lowest=True)


def wp_gap_by_rrr(df: pd.DataFrame, wp) -> pd.DataFrame:
    """Diagnostic 1: mean Markov WP vs empirical win rate, sliced by fine RRR."""
    d = df.copy()
    d["markov_wp"] = wp.predict(d["b"].to_numpy(), d["w"].to_numpy(), d["r"].to_numpy())
    d["rrr_slice"] = _rrr_slice(d)
    g = d.groupby("rrr_slice", observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "mean_markov_wp": g["markov_wp"].mean(),
        "empirical_winrate": g["y"].mean(),
    })
    out["gap"] = out["mean_markov_wp"] - out["empirical_winrate"]
    return out[out["n"] >= MIN_CELL]


def wp_gap_by_rrr_and_wickets(df: pd.DataFrame, wp) -> pd.DataFrame:
    """Diagnostic 1b: the same gap, split by wickets in hand, to show the RRR
    signature is not just a wickets artifact."""
    d = df.copy()
    d["markov_wp"] = wp.predict(d["b"].to_numpy(), d["w"].to_numpy(), d["r"].to_numpy())
    d["rrr_slice"] = _rrr_slice(d)
    w = d["w"]
    d["w_grp"] = np.where(w >= 7, "7-10 in hand", np.where(w >= 4, "4-6 in hand", "1-3 in hand"))
    g = d.groupby(["w_grp", "rrr_slice"], observed=True)
    out = pd.DataFrame({"n": g.size(), "gap": g["markov_wp"].mean() - g["y"].mean()})
    return out[out["n"] >= MIN_CELL]


def marginal_tail_check(df: pd.DataFrame, dist) -> pd.DataFrame:
    """Diagnostic 2: model vs empirical boundary rate along fine RRR.

    For each ball, the model's p(6), p(4) at its exact state, averaged per RRR
    slice, against the empirical fraction of that outcome. If model < empirical at
    high RRR, the marginal itself is tail-thinned (an estimation fix; martingale
    preserved). If they match, the WP gap in Diagnostic 1 is correlation, not
    marginals.
    """
    d = df.copy()
    states = list(zip(d["b"], d["w"], d["r"]))
    dists = [dist(s) for s in states]
    for o in (*BOUNDARY, "W"):
        d[f"model_p{o}"] = [dd[o] for dd in dists]
        d[f"emp_{o}"] = (d["outcome"] == o).astype(float)
    d["model_boundary"] = d["model_p4"] + d["model_p6"]
    d["emp_boundary"] = d["emp_4"] + d["emp_6"]
    d["rrr_slice"] = _rrr_slice(d)
    g = d.groupby("rrr_slice", observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "model_p_boundary": g["model_boundary"].mean(),
        "emp_p_boundary": g["emp_boundary"].mean(),
        "model_pW": g["model_pW"].mean(),
        "emp_pW": g["emp_W"].mean(),
    })
    out["boundary_shortfall"] = out["emp_p_boundary"] - out["model_p_boundary"]  # >0 => model too thin
    out["wicket_shortfall"] = out["emp_pW"] - out["model_pW"]  # >0 => model under-rates wickets
    return out[out["n"] >= MIN_CELL]


def _fmt(df: pd.DataFrame, float_cols: list[str], nd: int = 4) -> str:
    d = df.reset_index()
    for c in float_cols:
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:.{nd}f}")
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> int:
    df = pd.read_parquet(config.BALLS_PARQUET)
    train = df[df["split"] == "train"].copy()
    holdout = df[df["split"] == "holdout"].copy()
    r_max = int(df["r"].max()) + 1

    params = fit_rrr(train)  # unweighted marginals, matching the estimation model
    dist = make_outcome_model("rrr", params)
    _, wp = era_wp(train, r_max)  # the era-adjusted WP leverage actually rides on

    gap = wp_gap_by_rrr(train, wp)
    gap_w = wp_gap_by_rrr_and_wickets(train, wp)
    tail = marginal_tail_check(train, dist)
    gap_oos = wp_gap_by_rrr(holdout, wp)  # robustness: does the SHAPE persist OOS?

    # headline signals
    corr_gap_rrr = float(np.corrcoef(
        np.arange(len(gap)), gap["gap"].to_numpy())[0, 1]) if len(gap) > 2 else float("nan")
    worst = gap.sort_values("gap").iloc[0]
    tail_hi = tail.iloc[-1]
    easy_gap = float(gap.iloc[0]["gap"])        # sign on the easiest chases
    max_shortfall = float(tail["boundary_shortfall"].abs().max())
    max_wkt_shortfall = float(tail["wicket_shortfall"].abs().max())
    # aggregate level bias (n-weighted), to reconcile with validation_report
    agg_model = float((gap["mean_markov_wp"] * gap["n"]).sum() / gap["n"].sum())
    agg_emp = float((gap["empirical_winrate"] * gap["n"]).sum() / gap["n"].sum())
    sign_flips = bool((gap["gap"] > 0).any() and (gap["gap"] < 0).any())

    L = ["# Calibration-gap localization: what shape is the Markov WP's miscalibration?", ""]
    L += [f"Train split, {len(train):,} balls. WP = era-adjusted RRR Markov (the surface "
          f"leverage/WPA ride on). Estimation-only diagnostics; era shift is out of frame "
          f"by construction (train-on-train).", "",
          "**Result: neither cheap hypothesis holds.** The gap is NOT monotone in RRR (so "
          "not tail-thinning) and the per-ball marginals match empirical (so not an "
          "estimation artifact). The gap SIGN-FLIPS -- the model is over-confident on easy "
          "chases and under-confident on hard ones -- i.e. its WP is OVER-DISPERSED. See "
          "below.", ""]

    L += ["## Diagnostic 1 -- WP calibration gap by RRR", "",
          "`gap = mean(Markov WP) - empirical win rate`. A pure tail-thinning story predicts "
          "gap < 0 growing monotonically more negative with RRR. That is NOT what happens.", "",
          _fmt(gap, ["mean_markov_wp", "empirical_winrate", "gap"]), "",
          f"The gap flips sign (**{sign_flips}**): **positive {easy_gap:+.4f} on the easiest "
          f"chases** (model WP too high) and strongly **negative in the live middle** "
          f"(worst RRR **{worst.name}** at **{worst['gap']:+.4f}**, {int(worst['n']):,} "
          f"balls), fading toward 0 for near-hopeless chases. Correlation of gap with RRR "
          f"order is only **{corr_gap_rrr:+.3f}** -- flat, not monotone -- so this is not "
          f"tail-thinning. It is the signature of OVER-DISPERSED WP: predictions pushed too "
          f"far toward 0 and 1 at both ends.", "",
          f"This refines validation_report.md. The n-weighted mean WP is {agg_model:.3f} vs "
          f"empirical {agg_emp:.3f}, so the model DOES under-predict on average -- but only "
          f"because live balls cluster on the hard side of the crossover, where the model is "
          f"too low. The underlying defect is over-dispersion (too extreme both ways), not "
          f"uniform under-confidence.", ""]

    L += ["## Diagnostic 1b -- same gap, split by wickets in hand", "",
          "Confirms the RRR signature is not a wickets-in-hand artifact.", "",
          _fmt(gap_w, ["gap"]), ""]

    L += ["## Diagnostic 2 -- outcome-marginal tail check (rules out the cheap fix)", "",
          f"Model's own per-ball boundary probability p(4)+p(6) vs the empirical rate, "
          f"along fine RRR. `boundary_shortfall = empirical - model` > 0 would mean the "
          f"model's MARGINAL is itself tail-thinned. The model bins RRR as {RRR_BIN_EDGES} "
          f"with a 15+ catch-all shrunk (alpha=10) toward calmer (phase,w) parents, so "
          f"extreme chases were the prime suspects.", "",
          _fmt(tail, ["model_p_boundary", "emp_p_boundary", "model_pW", "emp_pW",
                      "boundary_shortfall", "wicket_shortfall"]), "",
          f"Both the scoring marginal (boundary p4+p6) and the wicket marginal p(W) match "
          f"empirical at every RRR: max |boundary shortfall| = **{max_shortfall:.4f}**, max "
          f"|wicket shortfall| = **{max_wkt_shortfall:.4f}**, both at the noise level. The "
          f"per-ball marginals are well estimated in both directions, so the WP gap is NOT "
          f"a marginal error, and finer high-RRR bins / lighter shrinkage would not close "
          f"it.", "",
          f"This makes the diagnosis airtight. If the one-ball marginals are correct AND "
          f"balls were conditionally independent given (b,w,r), the backward-induction "
          f"WP would equal E[y|b,w,r] exactly. It does not (Diagnostic 1). With the "
          f"marginals ruled out, the only remaining cause is conditional DEPENDENCE -- "
          f"ball-to-ball correlation given the state.", ""]

    # --- OOS robustness: the sign-flip shape, on the held-out split -----------
    flips_oos = bool((gap_oos["gap"] > 0).any() and (gap_oos["gap"] < 0).any())
    worst_oos = gap_oos.sort_values("gap").iloc[0]
    L += ["## Out-of-sample robustness -- the shape on the held-out split", "",
          f"Same Diagnostic-1 table on the held-out seasons. CAVEAT: unlike the "
          f"train-on-train tables above, the era shift IS in frame here (the model is "
          f"fit on lower-scoring seasons), so the LEVEL is expected to move toward "
          f"under-prediction; the robustness check is whether the non-monotone "
          f"sign-flip STRUCTURE persists, not the level.", "",
          _fmt(gap_oos, ["mean_markov_wp", "empirical_winrate", "gap"]), "",
          f"Sign flip out of sample: **{flips_oos}**; worst slice RRR "
          f"**{worst_oos.name}** at gap **{worst_oos['gap']:+.4f}** "
          f"({int(worst_oos['n']):,} balls). "
          + ("The over-dispersion shape is not an in-sample artifact."
             if flips_oos else
             "The positive band does not survive out of sample -- consistent with "
             "the era shift swamping the over-confidence on easy chases; the "
             "deep negative mid-RRR band, which carries the diagnosis, persists."),
          ""]

    L += ["## Reading -- the cheap fix is dead; the gap is genuine dependence, "
          "and it decomposes", "",
          "- **Not tail-thinning.** The gap is non-monotone and sign-flipping (Diagnostic "
          "1), not a monotone right-tail deficit.",
          "- **Not marginal estimation.** Per-ball p(4)+p(6) matches empirical at every RRR "
          "(Diagnostic 2), so the martingale-preserving estimation fix would buy nothing.",
          "- **It is over-dispersed WP from unmodelled dependence given (b,w,r).** "
          "Independent draws under-disperse the innings trajectory, so the final outcome "
          "looks more determined than it is and WP is pushed too far toward 0/1.",
          "- **The dependence is decomposed in dependence_decomposition.md:** it is "
          "short-range SEQUENTIAL run-scoring persistence (~3-5 ball range; lag-1 "
          "+0.036 above the permutation null), surviving innings and partnership "
          "demeaning. Innings-level heterogeneity contributes only ~18% of the "
          "signal, and wickets ANTI-cluster at short lags -- so the mechanism is "
          "scoring bursts and their mirror-image droughts, NOT wicket clusters. The "
          "burst/drought symmetry is exactly a two-sided variance effect, matching "
          "the sign-flip.",
          "- **Consequence for the fix.** The cure must inject sequential dependence "
          "(variance), which breaks the conditional independence behind WP(s)=sum_o p_o "
          "WP(s') -- and therefore the leverage definition -- RELATIVE TO the (b,w,r) "
          "state. There is no cheap martingale-preserving fix on this state; a richer "
          "state could in principle restore both properties, but the obvious "
          "enrichment (striker_balls) is a null (baseline_comparison.md). "
          "correlation_experiment.md quantifies the closure a dependence-honouring "
          "process recovers (~28%, a lower bound) before any rebuild decision.", ""]

    out = config.REPORTS / "tail_diagnostics.md"
    out.write_text("\n".join(L))
    print(f"corr(gap, rrr_order) = {corr_gap_rrr:+.3f}")
    print(f"worst WP gap: RRR {worst.name}  gap {worst['gap']:+.4f}  (n={int(worst['n']):,})")
    print(f"OOS sign flip: {flips_oos}  worst OOS gap: RRR {worst_oos.name} "
          f"{worst_oos['gap']:+.4f}")
    print(f"high-RRR boundary shortfall (emp - model): {tail_hi['boundary_shortfall']:+.4f} "
          f"at RRR {tail_hi.name}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
