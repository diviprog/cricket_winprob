# Does honouring ball-to-ball dependence close the calibration gap?

Train split, 112,019 balls. Constructive test of the tail_diagnostics conclusion. Uses the PLAIN (non-era) RRR Markov model so every mode shares the same flat empirical marginals as the block resampler -- era adjustment is an orthogonal level correction that would otherwise confound the marginal match. 400 sims/state, 220/slice, replicated over 6 seeds (each seed redraws both the evaluated states and the MC paths).

## Part A -- the full one-ball marginal matches empirical

Total variation distance between the model's per-ball distribution over {0..6,W} and empirical, by RRR slice. Completes the elimination (tail_diagnostics only checked boundary + wicket).

| rrr_slice | n | tv | max_abs_dev |
|---|---|---|---|
| 0-6 | 14069 | 0.0006 | 0.0005 |
| 6-8 | 24793 | 0.0003 | 0.0002 |
| 8-10 | 33823 | 0.0002 | 0.0001 |
| 10-12 | 17747 | 0.0004 | 0.0003 |
| 12-14 | 8784 | 0.0038 | 0.0022 |
| 14-16 | 4087 | 0.0116 | 0.0090 |
| 16-18 | 2101 | 0.0141 | 0.0128 |
| 18-22 | 2370 | 0.0181 | 0.0128 |
| 22-40 | 4245 | 0.0117 | 0.0095 |

Max TV across slices = **0.0181** (0 = identical distributions). The full marginal is well estimated everywhere, so a WP gap cannot be a marginal error.

## Part B -- gap vs dependence (block length K), seed-0 detail

Mean WP by mode vs empirical win rate, by RRR slice. `model_iid` draws each ball independently from the fitted model (independence reference, should track `markov`). `block_Kk` resamples real k-ball consecutive runs matched to the current cell -- same marginals, real dependence. `scattered_K20` was designed as a heterogeneity-only control (cell-matched draws from random positions of one donor innings) but turned out DEGENERATE -- see the reading below; do not interpret it as a mechanism probe.

| rrr_slice | n | emp | markov | model_iid | block_K1 | block_K3 | block_K8 | block_K20 | block_K40 | block_K80 | block_K120 | scattered_K20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-6 | 220 | 0.9667 | 0.9849 | 0.9829 | 0.9875 | 0.9846 | 0.9813 | 0.9764 | 0.9671 | 0.9606 | 0.9589 | 0.9811 |
| 6-8 | 220 | 0.7622 | 0.7741 | 0.7741 | 0.7789 | 0.7726 | 0.7633 | 0.7500 | 0.7040 | 0.6857 | 0.6839 | 0.8099 |
| 8-10 | 220 | 0.5209 | 0.3927 | 0.4016 | 0.4077 | 0.4145 | 0.4173 | 0.4243 | 0.4029 | 0.4176 | 0.4267 | 0.5165 |
| 10-12 | 220 | 0.2995 | 0.1411 | 0.1272 | 0.1337 | 0.1397 | 0.1505 | 0.1679 | 0.1622 | 0.1812 | 0.1925 | 0.2408 |
| 12-14 | 220 | 0.1777 | 0.0685 | 0.0635 | 0.0661 | 0.0755 | 0.0895 | 0.1064 | 0.1196 | 0.1252 | 0.1235 | 0.1482 |
| 14-16 | 220 | 0.1006 | 0.0365 | 0.0377 | 0.0375 | 0.0427 | 0.0510 | 0.0601 | 0.0622 | 0.0629 | 0.0617 | 0.1042 |
| 16-18 | 220 | 0.0662 | 0.0189 | 0.0178 | 0.0178 | 0.0224 | 0.0303 | 0.0392 | 0.0390 | 0.0390 | 0.0406 | 0.0624 |
| 18-22 | 220 | 0.0325 | 0.0140 | 0.0098 | 0.0108 | 0.0127 | 0.0158 | 0.0177 | 0.0176 | 0.0182 | 0.0182 | 0.0417 |
| 22-40 | 220 | 0.0090 | 0.0032 | 0.0032 | 0.0032 | 0.0041 | 0.0044 | 0.0046 | 0.0043 | 0.0042 | 0.0042 | 0.0088 |

## Replicated mean |gap| by mode (6 seeds; lower = better calibrated)

`marginal_TV_vs_K1` = max across seeds of the total variation between the mode's realized per-ball outcome distribution and the independent K=1 mode's -- near 0 means the mode adds dependence WITHOUT changing the marginals, so gap movement is attributable to dependence alone.

| mode | mean |gap| | min..max over seeds | marginal_TV_vs_K1 |
|---|---|---|---|
| markov | 0.0624 | 0.0624..0.0624 | -- |
| model_iid | 0.0618 | 0.0587..0.0667 | 0.0007 |
| block_K1 | 0.0613 | 0.0582..0.0663 | 0.0000 |
| block_K3 | 0.0565 | 0.0532..0.0611 | 0.0010 |
| block_K8 | 0.0504 | 0.0477..0.0549 | 0.0017 |
| block_K20 | 0.0451 | 0.0432..0.0490 | 0.0026 |
| block_K40 | 0.0502 | 0.0481..0.0545 | 0.0033 |
| block_K80 | 0.0480 | 0.0455..0.0529 | 0.0032 |
| block_K120 | 0.0456 | 0.0433..0.0498 | 0.0032 |
| scattered_K20 | 0.0187 | 0.0163..0.0218 | 0.0180 |

Scattered-arm fallback rate (donor innings lacked the needed cell, global pool substituted): **18.0%** of draws.

## Reading, with the cross-seed spread

Independence references agree at every seed: markov 0.0624 ~ model_iid 0.0618 ~ block_K1 0.0613 (validates the simulator and the flat-marginal match). Injecting real dependence shrinks the mean |gap| to **0.0451 at K=20** -- a **26% reduction** versus independence, with marginal fidelity <= TV 0.0033 across all block modes and seeds. The gap closes **constructively**.

Observations:
- **It closes in both directions**, as over-dispersion predicts: dependence adds outcome variance, pulling WP toward 0.5 -- UP in hard chases (RRR>=8, where the model was too low) and DOWN in easy ones (RRR 0-6, where it was too high). Per dependence_decomposition.md the dependence is run-scoring persistence (wickets ANTI-cluster), whose burst/drought symmetry is exactly a two-sided variance effect.
- **Saturation.** The minimum at **K=20** is stable across seeds: the K=20 < K=40 ordering holds in 6/6 seeds, and the K=20 mean sits below every seed's K=1 value. The saturation scale is consistent with the **~3-5-ball sequential correlation range** measured in dependence_decomposition.md: a block's extra cumulative variance grows with K only until K is several times the correlation range, so a ~5-ball range saturates near K~20.
- **The scattered arm.** It closed **263%** of the consecutive-block closure -- MORE than replaying whole real trajectories (mean |gap| 0.0187 vs block_K120 0.0456), which no heterogeneity injection could do, and it broke its own marginal control (realized-marginal TV 0.0180 vs <= 0.0033 for every block mode). The arm is DEGENERATE as designed: re-matching every ball to the sim's evolving cell within one donor makes it a nearest-neighbour replay of empirical outcomes -- an E[y|state] estimator in disguise, not a dependence injection. It is reported as a negative methodological result; the mechanism question is settled by dependence_decomposition.md (sequential, ~18% heterogeneity), whose permutation-null design does not have this failure mode.
- **26% is a lower bound, and the block bootstrap cannot measure more.** Even whole-innings replays (K=120) splice segments that diverge from the sim's evolving state and break dependence at block starts, so they extract no further clean signal. Pinning down the full contribution needs a generative dependent model (e.g. Markov-switching outcomes) -- the natural next step, and the one that would forfeit the exact martingale.
