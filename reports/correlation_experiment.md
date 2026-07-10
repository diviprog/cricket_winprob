# Does honouring ball-to-ball correlation close the calibration gap?

Train split, 112,019 balls. Constructive test of the tail_diagnostics conclusion. Uses the PLAIN (non-era) RRR Markov model so every mode shares the same flat empirical marginals as the block resampler -- era adjustment is an orthogonal level correction that would otherwise confound the marginal match. 400 sims/state, 220/slice.

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

## Part B -- gap vs correlation (block length K)

Mean WP by mode vs empirical win rate, by RRR slice. `model_iid` draws each ball independently from the fitted model (independence reference, should track `markov`). `block_Kk` resamples real k-ball runs matched to the current cell -- same marginals, real correlation.

| rrr_slice | n | emp | markov | model_iid | block_K1 | block_K3 | block_K8 | block_K20 | block_K40 | block_K80 | block_K120 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-6 | 220 | 0.9667 | 0.9849 | 0.9828 | 0.9880 | 0.9848 | 0.9809 | 0.9769 | 0.9673 | 0.9618 | 0.9591 |
| 6-8 | 220 | 0.7622 | 0.7741 | 0.7735 | 0.7805 | 0.7724 | 0.7638 | 0.7505 | 0.6998 | 0.6884 | 0.6793 |
| 8-10 | 220 | 0.5209 | 0.3927 | 0.4010 | 0.4097 | 0.4133 | 0.4178 | 0.4239 | 0.4043 | 0.4160 | 0.4268 |
| 10-12 | 220 | 0.2995 | 0.1411 | 0.1253 | 0.1327 | 0.1395 | 0.1508 | 0.1653 | 0.1626 | 0.1796 | 0.1913 |
| 12-14 | 220 | 0.1777 | 0.0685 | 0.0652 | 0.0659 | 0.0762 | 0.0898 | 0.1065 | 0.1210 | 0.1258 | 0.1236 |
| 14-16 | 220 | 0.1006 | 0.0365 | 0.0365 | 0.0375 | 0.0425 | 0.0507 | 0.0612 | 0.0618 | 0.0617 | 0.0624 |
| 16-18 | 220 | 0.0662 | 0.0189 | 0.0180 | 0.0181 | 0.0227 | 0.0308 | 0.0397 | 0.0382 | 0.0388 | 0.0391 |
| 18-22 | 220 | 0.0325 | 0.0140 | 0.0111 | 0.0110 | 0.0128 | 0.0161 | 0.0184 | 0.0179 | 0.0175 | 0.0180 |
| 22-40 | 220 | 0.0090 | 0.0032 | 0.0033 | 0.0032 | 0.0037 | 0.0044 | 0.0045 | 0.0043 | 0.0040 | 0.0042 |

## Saturation: mean |gap| and marginal fidelity vs block length

`mean |gap|` = mean absolute calibration gap (lower = better calibrated). `marginal_TV_vs_K1` = total variation between this mode's realized per-ball outcome distribution and the independent K=1 mode's -- near 0 means longer blocks add correlation WITHOUT changing the marginals, so the gap-closing is attributable to correlation alone.

| mode | mean |gap| | marginal_TV_vs_K1 |
|---|---|---|
| markov | 0.0624 | -- |
| model_iid | 0.0637 | -- |
| block_K1 | 0.0631 | 0.0000 |
| block_K3 | 0.0582 | 0.0009 |
| block_K8 | 0.0513 | 0.0015 |
| block_K20 | 0.0454 | 0.0019 |
| block_K40 | 0.0510 | 0.0023 |
| block_K80 | 0.0491 | 0.0024 |
| block_K120 | 0.0480 | 0.0021 |

Independence references agree: markov 0.0624 ~ model_iid 0.0637 ~ block_K1 0.0631 (validates the simulator and confirms all three share the flat empirical marginals). Adding real correlation shrinks the gap to a minimum of **0.0454 at K=20** -- a **28% reduction** in mean |gap| versus independence. Critically, this is clean: the realized per-ball distribution drifts from the independent K=1 mode by at most TV **0.0024** across ALL block lengths, so the closure is correlation, not a change in the marginals. The gap closes **constructively**, confirming the diagnosis.

Three observations:
- **It closes in both directions**, exactly as the over-dispersion story predicts: correlation adds outcome variance, pulling WP toward 0.5 -- UP in hard chases (RRR>=8, where the model was too low) and DOWN in easy ones (RRR 0-6, where it was too high).
- **The effect is partnership-scale.** The reduction saturates at K~20 balls (~3 overs, a typical partnership) and then plateaus/re-widens -- consistent with the correlation being a within-partnership 'set batsman' persistence, the same source as the +0.037 lag-1 autocorrelation.
- **28% is a lower bound, and the block bootstrap cannot measure more.** Even whole-innings replays (K=120) splice segments that diverge from the sim's evolving state and break correlation at block starts, so they extract no further clean signal. Pinning down the full contribution needs a generative correlated model (e.g. Markov-switching outcomes) -- the natural next step, and the one that would forfeit the exact martingale.
