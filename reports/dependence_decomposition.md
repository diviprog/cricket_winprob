# Dependence decomposition: serial correlation vs shared latents

Train split, 112,019 balls, 200 within-innings permutations. Residual = value minus its (phase, w) state mean. The permutation null PRESERVES innings-level composition (shuffling within an innings keeps the innings' shared latent in every pair) while destroying ordering and partnership structure -- so `observed above the null band` means dependence BEYOND innings-level heterogeneity, and the null centre estimates the heterogeneity component itself.

## Runs residuals -- within-innings lag profile

### state-demeaned (the M3 construction)

| lag | observed | null mean | null 2.5% | null 97.5% | above band |
|---|---|---|---|---|---|
| 1 | +0.0436 | +0.0079 | +0.0013 | +0.0140 | **yes** |
| 2 | +0.0376 | +0.0081 | +0.0023 | +0.0139 | **yes** |
| 3 | +0.0287 | +0.0079 | +0.0024 | +0.0136 | **yes** |
| 5 | +0.0155 | +0.0076 | +0.0013 | +0.0131 | **yes** |
| 10 | +0.0127 | +0.0081 | +0.0018 | +0.0148 | no |
| 20 | +0.0058 | +0.0076 | +0.0013 | +0.0137 | no |
| 30 | +0.0074 | +0.0081 | +0.0013 | +0.0139 | no |

### + innings-demeaned

| lag | observed | null mean | null 2.5% | null 97.5% | above band |
|---|---|---|---|---|---|
| 1 | +0.0274 | -0.0092 | -0.0158 | -0.0029 | **yes** |
| 2 | +0.0212 | -0.0089 | -0.0150 | -0.0030 | **yes** |
| 3 | +0.0122 | -0.0091 | -0.0148 | -0.0033 | **yes** |
| 5 | -0.0012 | -0.0094 | -0.0157 | -0.0039 | **yes** |
| 10 | -0.0043 | -0.0089 | -0.0151 | -0.0022 | no |
| 20 | -0.0115 | -0.0092 | -0.0155 | -0.0032 | no |
| 30 | -0.0096 | -0.0086 | -0.0154 | -0.0026 | no |

### + partnership-demeaned, within-partnership pairs

| lag | observed | null mean | null 2.5% | null 97.5% | above band |
|---|---|---|---|---|---|
| 1 | -0.0359 | -0.0530 | -0.0592 | -0.0479 | **yes** |
| 2 | -0.0353 | -0.0480 | -0.0548 | -0.0418 | **yes** |
| 3 | -0.0381 | -0.0448 | -0.0510 | -0.0379 | no |
| 5 | -0.0456 | -0.0396 | -0.0469 | -0.0329 | no |
| 10 | -0.0324 | -0.0308 | -0.0386 | -0.0230 | no |

## Continuity with the M3 headline number

Within-partnership lag-1 autocorrelation of state-demeaned runs residuals (the construction behind the +0.037 in leverage_validation.md, here on train only): **+0.0354**, permutation null [+0.0013, +0.0139] centred at +0.0079.

## Wicket residuals -- within-innings lag profile

### state-demeaned

| lag | observed | null mean | null 2.5% | null 97.5% | above band |
|---|---|---|---|---|---|
| 1 | -0.0087 | -0.0008 | -0.0064 | +0.0046 | below |
| 2 | -0.0119 | -0.0008 | -0.0068 | +0.0049 | below |
| 3 | -0.0101 | -0.0011 | -0.0074 | +0.0060 | below |
| 5 | -0.0039 | -0.0011 | -0.0071 | +0.0060 | no |
| 10 | +0.0016 | -0.0004 | -0.0065 | +0.0069 | no |
| 20 | +0.0022 | -0.0008 | -0.0070 | +0.0051 | no |
| 30 | +0.0097 | -0.0013 | -0.0076 | +0.0049 | **yes** |

### + innings-demeaned

| lag | observed | null mean | null 2.5% | null 97.5% | above band |
|---|---|---|---|---|---|
| 1 | -0.0162 | -0.0090 | -0.0145 | -0.0035 | below |
| 2 | -0.0193 | -0.0090 | -0.0150 | -0.0032 | below |
| 3 | -0.0174 | -0.0092 | -0.0156 | -0.0021 | below |
| 5 | -0.0109 | -0.0092 | -0.0153 | -0.0021 | no |
| 10 | -0.0054 | -0.0084 | -0.0147 | -0.0010 | no |
| 20 | -0.0045 | -0.0088 | -0.0149 | -0.0028 | no |
| 30 | +0.0029 | -0.0090 | -0.0152 | -0.0029 | **yes** |

## Variance decomposition (ICC)

| residual | innings-level ICC | partnership-level ICC |
|---|---|---|
| runs | 0.0081 | 0.0173 |
| wicket | -0.0008 | 0.0680 |

Cross-check: the state-variant null-band centre at lag 1 (+0.0079) should sit near the innings ICC (+0.0081) -- both estimate the innings-composition component.

Caveat: the wicket *partnership* ICC is mechanically confounded and should not be read as a latent -- every partnership ends in exactly one wicket, so its per-partnership wicket rate is 1/length by construction and the group-mean variance is a length artifact.

## Interpretation matrix

| observation | verdict |
|---|---|
| profile decays, survives innings demeaning | genuine serial correlation |
| flat profile, collapses under innings demeaning, innings ICC > 0 | match-level heterogeneity |
| collapses under partnership demeaning, partnership ICC >> innings ICC | partnership-scale latent |
| mixed | report the shares |

## Verdict (computed, runs residuals)

- dependence beyond innings composition at lag 1: **yes**
- survives innings demeaning: **yes**
- survives partnership demeaning (within-partnership pairs): **yes**
- innings-composition share of the lag-1 signal: **18%**
- excess over null by lag (decaying): lag 1: +0.0357, lag 2: +0.0295, lag 3: +0.0208, lag 5: +0.0078, lag 10: +0.0046, lag 20: -0.0018, lag 30: -0.0007

**sequential** -- Genuine ball-adjacent sequential dependence survives even after removing partnership means; momentum-like correlation is real.

## Verdict (wicket residuals)

Short-lag wicket dependence is **NEGATIVE**: lag-1 observed -0.0087 vs null [-0.0064, +0.0046]. Wickets ANTI-cluster ball-adjacent -- immediately after state-mean adjustment, a wicket makes another wicket in the next few balls LESS likely than independence predicts (consolidation). The 'wicket clusters' half of the original mechanism story is not supported; the dependence that closes the calibration gap is run-scoring persistence. Note positive run-rate persistence alone explains the two-sided gap closure: bursts make hard chases more winnable, and the mirror-image droughts make easy chases more losable.
