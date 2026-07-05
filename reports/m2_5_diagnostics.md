# M2.5 Pre-M3 Diagnostics

Held-out test: ['2024', '2025', '2026'] | 18,010 balls | realized win rate 0.498. Train: 112,019 balls.

Analysis only -- reuses the M2 on-disk split, the M2 metric functions (`brier`/`log_loss`/`ece`/`reliability_curve`), and the existing fitters/solver. The exact core and outcome model are unchanged.

## 1. Brier (Murphy) decomposition

Brier = Reliability - Resolution + Uncertainty (binned, 15 bins). Reliability = calibration error (lower better). Resolution = discriminative power (higher better). Uncertainty = base_rate*(1-base_rate), shared.

| model | Brier | Reliability | Resolution | Uncertainty | Rel-Res+Unc |
|---|---|---|---|---|---|
| base Markov | 0.2040 | 0.0685 | 0.1126 | 0.2500 | 0.2059 |
| RRR Markov | 0.1637 | 0.0386 | 0.1237 | 0.2500 | 0.1650 |
| pure-RRR logistic | 0.1404 | 0.0129 | 0.1219 | 0.2500 | 0.1410 |
| base-rate const | 0.2503 | 0.0003 | 0.0000 | 0.2500 | 0.2503 |

**Hypothesis test (RRR Markov vs logistic).** Brier gap = +0.0234. Of this, reliability contributes +0.0257 and (lost) resolution contributes -0.0017. Reliability accounts for 110% of the gap. RRR Markov resolution 0.1237 vs logistic 0.1219 (difference -0.0017).

## 2. Post-hoc recalibration of the RRR Markov WP

Calibration fold: season **2023** (8,090 balls), disjoint from both the outcome-model training data (refit on the other 103,929 train balls) and the held-out test set (2024-2026). Maps fit on the fold, scored on test.

| model | Brier | log loss | ECE |
|---|---|---|---|
| RRR Markov (raw, refit) | 0.1660 | 0.5406 | 0.1672 |
| RRR Markov + isotonic | 0.1356 | 0.4254 | 0.0722 |
| RRR Markov + Platt | 0.1429 | 0.4512 | 0.1117 |
| pure-RRR logistic (refit) | 0.1403 | 0.4379 | 0.0784 |

**Gap closure.** Brier gap to the logistic before recalibration = +0.0257; after the better map = -0.0047. Recalibration alone closes **118%** of the Brier gap. ECE falls from 0.1672 (raw) to 0.0722 (best map).

> **Martingale caveat (flag, do not resolve here).** Pointwise recalibration applies a monotone map to WP that is NOT derived from the outcome model, so the recalibrated WP no longer satisfies WP(s) = sum_o p_o WP(s') w.r.t. the outcome distribution. It breaks the martingale consistency between the outcome model and WP. Therefore leverage in M3 (swing = MAD of the next-ball WP under the martingale) cannot use the recalibrated WP without losing the clean "MAD of a martingale" interpretation. This is a real tradeoff for M3, left open.

## 3. Calibration sliced by ahead/behind the required run rate

Slice = sign(CRR - RRR), CRR = 6*runs_scored/balls_faced, runs_scored = r_start - r, r_start = max r in the innings. First ball (balls_faced=0) -> CRR=0 -> classified behind. Curves for the RRR Markov, held-out set. See `m2_5_reliability_by_position.png`.

| slice | n | mean pred | actual | ECE | bins with realized>pred |
|---|---|---|---|---|---|
| ahead of RRR | 7,282 | 0.637 | 0.793 | 0.1605 | 93% |
| behind RRR | 10,728 | 0.134 | 0.298 | 0.1644 | 87% |

**Pattern.** The reliability curve is under-confident in BOTH slices (realized > predicted on both sides) -> a ONE-DIRECTIONAL bias, consistent with an era/scoring shift dominating, NOT the symmetric signature of memoryless tail-thinning. Tail-thinning predicts under-confidence when behind (an upper-tail win) and OVER-confidence when ahead (a lower-tail collapse). Observed: ahead slice realized 0.793 vs predicted 0.637 (under-confident); behind slice realized 0.298 vs predicted 0.134 (under-confident).

## 4. In-sample vs held-out attribution

Gap = mean(predicted WP) - mean(realized y). In-sample gap has no distribution shift, so it isolates the structural memoryless error (tail-thinning). Train->test widening isolates the era shift.

| model | in Brier | out Brier | in gap | out gap | widening (out-in) |
|---|---|---|---|---|---|
| base Markov | 0.1984 | 0.2040 | -0.1346 | -0.2191 | -0.0845 |
| RRR Markov | 0.1630 | 0.1637 | -0.0713 | -0.1605 | -0.0892 |
| pure-RRR logistic | 0.1671 | 0.1404 | -0.0000 | -0.0666 | -0.0666 |
| base-rate const | 0.2498 | 0.2503 | +0.0000 | +0.0169 | +0.0169 |

**RRR Markov attribution.** Total held-out mean gap -0.1605 = in-sample (memoryless) -0.0713 + widening (era shift) -0.0892. Memoryless share ~= **44%**, era-shift share ~= **56%** of the held-out mean miscalibration.

**Logistic era-robustness claim.** Logistic train->test widening -0.0666 vs RRR Markov -0.0892. The logistic widening is NOT much smaller than the Markov's -> claim NOT supported; both degrade similarly under the era shift.

## 5. Robustness to the split

Four-model grid re-scored under alternative splits. The M2 holdout (2024-2026) is the highest-scoring era in IPL history, which maximally penalizes a stationary compounding model. If the RRR-Markov-minus-logistic Brier gap shrinks off that holdout, era shift is the dominant driver.

| split | RRR Markov Brier | logistic Brier | gap (Markov-logistic) |
|---|---|---|---|
| season holdout 2024-26 (M2) | 0.1637 | 0.1404 | +0.0234 |
| random match-grouped 80/20 | 0.1448 | 0.1553 | -0.0105 |
| mid-history season 2016 | 0.1591 | 0.1514 | +0.0077 |

**Verdict.** Gap on the 2024-26 holdout = +0.0234; mean gap on the non-era splits = -0.0014, i.e. the gap shrinks by **106%** off the high-scoring holdout. This independently confirms era shift as a dominant driver of the RRR-Markov-vs-logistic gap.

## Findings

**(a) Reliability vs resolution, per model.** RRR Markov: reliability 0.0386, resolution 0.1237. Logistic: reliability 0.0129, resolution 0.1219. base Markov: reliability 0.0685, resolution 0.1126. base-rate const: reliability 0.0003, resolution 0.0000. The RRR Markov's Brier gap to the logistic (+0.0234) is 110% reliability (+0.0257) and only -0.0017 lost resolution: it discriminates almost as well as the logistic and is chiefly MIScalibrated, not under-powered.

**(b) How much recalibration alone closes.** An isotonic/Platt map fit on a disjoint season fold closes **118%** of the Brier gap to the logistic (gap +0.0257 -> -0.0047), confirming (a): most of the deficit is fixable by calibration without changing the model class. But recalibration breaks the outcome-model<->WP martingale, so it is not free for M3.

**(c) Attribution: memoryless tail-thinning vs era shift.** In-sample the RRR Markov mean gap is -0.0713 (the pure structural/memoryless error, no shift), widening by -0.0892 on the held-out era. That splits the held-out mean miscalibration into ~**44% memoryless** and ~**56% era shift**. Diagnostic 3 shows the held-out miscalibration is one-directional (under-confident whether ahead or behind), and diagnostic 5 shows the Markov-vs-logistic gap shrinks 106% off the 2024-26 holdout. Both independently point to era shift as the larger, and the directionally dominant, driver of the held-out gap.

## Recommendation

**Recommendation: an era-adjustment to the outcome estimation (recency weighting / an era term), before M3.** The evidence is convergent: (1) the gap to the logistic is mostly reliability, not resolution; (2) the held-out miscalibration is one-directional under-confidence in both the ahead and behind slices -- the signature of a level shift, not the symmetric ahead-over / behind-under signature of memoryless tail-thinning; (3) the Markov-vs-logistic Brier gap shrinks ~106% once the high-scoring 2024-26 seasons are no longer the entire test set; and (4) the mean-gap attribution puts ~56% of the held-out error on the shift. M5 (hidden striker state) targets tail-thinning, which these tests show is the SMALLER, and wrong-directional, component -- it would not fix the dominant under-confidence. Cheap recalibration removes most of the metric gap but breaks the martingale that M3 leverage depends on, so it is a patch, not the fix. Re-estimate p_o(state) with recency weighting (or an explicit era term) so WP stays a proper martingale AND tracks modern scoring; then proceed to M3. Defer M5 until after the era term, and only if residual tail-clustering remains.
