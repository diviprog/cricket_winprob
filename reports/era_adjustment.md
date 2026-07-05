# Era Adjustment (recency-weighted RRR Markov outcome model)

Follow-up to M2.5. The RRR outcome model is now re-estimated with per-ball recency weights (`outcome_model.recency_weights`), tilting p_o(state) toward recent seasons. This is an estimation-layer change only: still one outcome distribution, still one backward-induction sweep, so WP stays an exact martingale and the DP core is unchanged.

## Half-life tuning (rolling-origin CV, disjoint from test)

Rolling-origin CV over the 3 most recent train seasons (2021, 2022, 2023): each is predicted one step ahead from strictly earlier seasons, recency-weighted relative to that earlier frame -- the same forecast the real 2024-2026 test faces. All folds are train seasons, disjoint from the held-out test.

The mean log loss is monotone-decreasing but flattening: the trend is monotone in time and each fold's target is the newest available season, so the raw argmin is pinned to the grid edge and driven by the single most-trend-exposed fold, while effective sample size (ESS) collapses. Selection rule (stated, not cherry-picked): the most aggressive half-life on the improving curve with mean ESS >= 15% of train.

| half-life (seasons) | mean CV log loss | mean CV Brier | mean ESS frac | per-fold log loss |
|---|---|---|---|---|
| none (M2) | 0.5356 | 0.1714 | 1.000 | [0.4753, 0.479, 0.6525] |
| 8.0 | 0.5259 | 0.1691 | 0.893 | [0.4705, 0.4696, 0.6374] |
| 6.0 | 0.5230 | 0.1685 | 0.827 | [0.4691, 0.4671, 0.6329] |
| 4.0 | 0.5182 | 0.1673 | 0.689 | [0.4667, 0.4631, 0.6247] |
| 3.0 | 0.5145 | 0.1665 | 0.568 | [0.4649, 0.4605, 0.618] |
| 2.0 | 0.5097 | 0.1655 | 0.403 | [0.4626, 0.4582, 0.6083] |
| 1.5 | 0.5072 | 0.1650 | 0.309 | [0.4616, 0.4584, 0.6018] |
| 1.0 | 0.5049 | 0.1646 | 0.213 | [0.461, 0.4607, 0.593] |
| 0.75 **<- chosen** | 0.5036 | 0.1644 | 0.166 | [0.461, 0.4628, 0.587] |
| 0.5 | 0.5023 | 0.1641 | 0.122 | [0.4617, 0.4651, 0.58] |

Chosen half-life: **0.75 seasons** (weight halves every 0.75 season(s); more aggressive settings keep gaining only on the newest fold and drop ESS below the floor).

## Held-out results (2024-2026) -- before vs after

| model | Brier | log loss | ECE | mean pred | mean gap |
|---|---|---|---|---|---|
| RRR Markov (M2, unweighted) | 0.1637 | 0.5313 | 0.1627 | 0.337 | -0.1605 |
| RRR Markov + recency (hl=0.75) | 0.1489 | 0.4741 | 0.1310 | 0.367 | -0.1304 |
| pure-RRR logistic | 0.1404 | 0.4379 | 0.0773 | 0.431 | -0.0666 |
| base-rate const | 0.2503 | 0.6937 | 0.0169 | 0.515 | +0.0169 |

Held-out actual win rate: 0.498.

## Effect

- **Held-out improvement.** Brier 0.1637 -> 0.1489 (+0.0148); log loss 0.5313 -> 0.4741 (+0.0572); ECE 0.1627 -> 0.1310 (+0.0316).
- **Mean-gap (under-confidence).** -0.1605 -> -0.1304: recency weighting closes **19%** of the one-directional under-confidence the M2.5 slice-by-position diagnostic flagged as the era signature.
- **Gap to the logistic.** Brier gap to the pure-RRR logistic +0.0234 -> +0.0085 (63% of it closed by the era term alone), and this WP is still a proper martingale (unlike the M2.5 recalibration, which was not).
- **In-sample (structural) piece.** Train mean gap -0.0713 -> -0.0397: recency weighting is a mild reweighting of the same fit, so the in-sample memoryless component is little changed, as expected -- the gain is on the era shift, not the tail-thinning.

## Honest ceiling

Recency weighting can only pull the fitted scoring level toward the most recent TRAIN season, not past it. Weighted train mean runs/legal-ball = 1.424 (vs 1.344 unweighted), still below the holdout's 1.598 because 2024-2026 out-scored every training season. So the era term reduces but cannot erase the shift without trend EXTRAPOLATION beyond the data, which trades the conservative no-leakage property for reach. Documented, not taken here.
