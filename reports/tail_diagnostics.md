# Calibration-gap localization: what shape is the Markov WP's miscalibration?

Train split, 112,019 balls. WP = era-adjusted RRR Markov (the surface leverage/WPA ride on). Estimation-only diagnostics; era shift is out of frame by construction (train-on-train).

**Result: neither cheap hypothesis holds.** The gap is NOT monotone in RRR (so not tail-thinning) and the per-ball marginals match empirical (so not an estimation artifact). The gap SIGN-FLIPS -- the model is over-confident on easy chases and under-confident on hard ones -- i.e. its WP is OVER-DISPERSED. See below.

## Diagnostic 1 -- WP calibration gap by RRR

`gap = mean(Markov WP) - empirical win rate`. A pure tail-thinning story predicts gap < 0 growing monotonically more negative with RRR. That is NOT what happens.

| rrr_slice | n | mean_markov_wp | empirical_winrate | gap |
|---|---|---|---|---|
| 0-6 | 15210 | 0.9770 | 0.9628 | 0.0142 |
| 6-8 | 24190 | 0.7852 | 0.7529 | 0.0323 |
| 8-10 | 33801 | 0.4466 | 0.5177 | -0.0711 |
| 10-12 | 18194 | 0.1827 | 0.2944 | -0.1117 |
| 12-14 | 8034 | 0.0796 | 0.1646 | -0.0849 |
| 14-16 | 4021 | 0.0467 | 0.1002 | -0.0535 |
| 16-18 | 2263 | 0.0284 | 0.0619 | -0.0334 |
| 18-22 | 2133 | 0.0114 | 0.0220 | -0.0106 |
| 22-40 | 2652 | 0.0061 | 0.0136 | -0.0075 |

The gap flips sign (**True**): **positive +0.0142 on the easiest chases** (model WP too high) and strongly **negative in the live middle** (worst RRR **10-12** at **-0.1117**, 18,194 balls), fading toward 0 for near-hopeless chases. Correlation of gap with RRR order is only **-0.078** -- flat, not monotone -- so this is not tail-thinning. It is the signature of OVER-DISPERSED WP: predictions pushed too far toward 0 and 1 at both ends.

This refines validation_report.md. The n-weighted mean WP is 0.482 vs empirical 0.522, so the model DOES under-predict on average -- but only because live balls cluster on the hard side of the crossover, where the model is too low. The underlying defect is over-dispersion (too extreme both ways), not uniform under-confidence.

## Diagnostic 1b -- same gap, split by wickets in hand

Confirms the RRR signature is not a wickets-in-hand artifact.

| w_grp | rrr_slice | n | gap |
|---|---|---|---|
| 1-3 in hand | 8-10 | 243 | 0.0687 |
| 1-3 in hand | 10-12 | 474 | -0.0276 |
| 1-3 in hand | 12-14 | 483 | -0.0238 |
| 1-3 in hand | 14-16 | 490 | -0.0117 |
| 1-3 in hand | 16-18 | 591 | -0.0016 |
| 1-3 in hand | 18-22 | 848 | -0.0064 |
| 1-3 in hand | 22-40 | 1662 | -0.0067 |
| 4-6 in hand | 0-6 | 2374 | -0.0139 |
| 4-6 in hand | 6-8 | 2276 | 0.0223 |
| 4-6 in hand | 8-10 | 4226 | -0.0213 |
| 4-6 in hand | 10-12 | 4695 | -0.1070 |
| 4-6 in hand | 12-14 | 3943 | -0.0823 |
| 4-6 in hand | 14-16 | 2630 | -0.0516 |
| 4-6 in hand | 16-18 | 1480 | -0.0367 |
| 4-6 in hand | 18-22 | 1222 | -0.0116 |
| 4-6 in hand | 22-40 | 969 | -0.0074 |
| 7-10 in hand | 0-6 | 12711 | 0.0199 |
| 7-10 in hand | 6-8 | 21829 | 0.0334 |
| 7-10 in hand | 8-10 | 29332 | -0.0795 |
| 7-10 in hand | 10-12 | 13025 | -0.1164 |
| 7-10 in hand | 12-14 | 3608 | -0.0959 |
| 7-10 in hand | 14-16 | 901 | -0.0817 |
| 7-10 in hand | 16-18 | 192 | -0.1065 |

## Diagnostic 2 -- outcome-marginal tail check (rules out the cheap fix)

Model's own per-ball boundary probability p(4)+p(6) vs the empirical rate, along fine RRR. `boundary_shortfall = empirical - model` > 0 would mean the model's MARGINAL is itself tail-thinned. The model bins RRR as [6.0, 8.0, 10.0, 12.0, 15.0] with a 15+ catch-all shrunk (alpha=10) toward calmer (phase,w) parents, so extreme chases were the prime suspects.

| rrr_slice | n | model_p_boundary | emp_p_boundary | model_pW | emp_pW | boundary_shortfall | wicket_shortfall |
|---|---|---|---|---|---|---|---|
| 0-6 | 15210 | 0.1599 | 0.1593 | 0.0414 | 0.0409 | -0.0006 | -0.0005 |
| 6-8 | 24190 | 0.1563 | 0.1563 | 0.0394 | 0.0396 | 0.0000 | 0.0002 |
| 8-10 | 33801 | 0.1643 | 0.1648 | 0.0431 | 0.0433 | 0.0005 | 0.0001 |
| 10-12 | 18194 | 0.1693 | 0.1681 | 0.0544 | 0.0547 | -0.0012 | 0.0003 |
| 12-14 | 8034 | 0.1772 | 0.1750 | 0.0630 | 0.0635 | -0.0022 | 0.0005 |
| 14-16 | 4021 | 0.1882 | 0.1937 | 0.0763 | 0.0741 | 0.0055 | -0.0022 |
| 16-18 | 2263 | 0.1936 | 0.1878 | 0.0920 | 0.0840 | -0.0058 | -0.0081 |
| 18-22 | 2133 | 0.1878 | 0.1889 | 0.0982 | 0.0956 | 0.0011 | -0.0026 |
| 22-40 | 2652 | 0.1742 | 0.1761 | 0.1041 | 0.1090 | 0.0019 | 0.0048 |

Both the scoring marginal (boundary p4+p6) and the wicket marginal p(W) match empirical at every RRR: max |boundary shortfall| = **0.0058**, max |wicket shortfall| = **0.0081**, both at the noise level. The per-ball marginals are well estimated in both directions, so the WP gap is NOT a marginal error, and finer high-RRR bins / lighter shrinkage would not close it.

This makes the diagnosis airtight. If the one-ball marginals are correct AND balls were conditionally independent given (b,w,r), the backward-induction WP would equal E[y|b,w,r] exactly. It does not (Diagnostic 1). With the marginals ruled out, the only remaining cause is conditional DEPENDENCE -- ball-to-ball correlation given the state.

## Out-of-sample robustness -- the shape on the held-out split

Same Diagnostic-1 table on the held-out seasons. CAVEAT: unlike the train-on-train tables above, the era shift IS in frame here (the model is fit on lower-scoring seasons), so the LEVEL is expected to move toward under-prediction; the robustness check is whether the non-monotone sign-flip STRUCTURE persists, not the level.

| rrr_slice | n | mean_markov_wp | empirical_winrate | gap |
|---|---|---|---|---|
| 0-6 | 1920 | 0.9604 | 0.9344 | 0.0260 |
| 6-8 | 2675 | 0.7984 | 0.9372 | -0.1388 |
| 8-10 | 4106 | 0.4465 | 0.7136 | -0.2671 |
| 10-12 | 4515 | 0.1480 | 0.3003 | -0.1523 |
| 12-14 | 1948 | 0.0495 | 0.1407 | -0.0912 |
| 14-16 | 1077 | 0.0240 | 0.0752 | -0.0512 |
| 16-18 | 476 | 0.0175 | 0.0462 | -0.0287 |
| 18-22 | 479 | 0.0061 | 0.0021 | 0.0041 |
| 22-40 | 558 | 0.0030 | 0.0000 | 0.0030 |

Sign flip out of sample: **True**; worst slice RRR **8-10** at gap **-0.2671** (4,106 balls). The over-dispersion shape is not an in-sample artifact.

## Reading -- the cheap fix is dead; the gap is genuine dependence, and it decomposes

- **Not tail-thinning.** The gap is non-monotone and sign-flipping (Diagnostic 1), not a monotone right-tail deficit.
- **Not marginal estimation.** Per-ball p(4)+p(6) matches empirical at every RRR (Diagnostic 2), so the martingale-preserving estimation fix would buy nothing.
- **It is over-dispersed WP from unmodelled dependence given (b,w,r).** Independent draws under-disperse the innings trajectory, so the final outcome looks more determined than it is and WP is pushed too far toward 0/1.
- **The dependence is decomposed in dependence_decomposition.md:** it is short-range SEQUENTIAL run-scoring persistence (~3-5 ball range; lag-1 +0.036 above the permutation null), surviving innings and partnership demeaning. Innings-level heterogeneity contributes only ~18% of the signal, and wickets ANTI-cluster at short lags -- so the mechanism is scoring bursts and their mirror-image droughts, NOT wicket clusters. The burst/drought symmetry is exactly a two-sided variance effect, matching the sign-flip.
- **Consequence for the fix.** The cure must inject sequential dependence (variance), which breaks the conditional independence behind WP(s)=sum_o p_o WP(s') -- and therefore the leverage definition -- RELATIVE TO the (b,w,r) state. There is no cheap martingale-preserving fix on this state; a richer state could in principle restore both properties, but the obvious enrichment (striker_balls) is a null (baseline_comparison.md). correlation_experiment.md quantifies the closure a dependence-honouring process recovers (~28%, a lower bound) before any rebuild decision.
