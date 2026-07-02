# WP Validation Report (M2, held-out split)

Held-out seasons: ['2024', '2025', '2026']
Held-out balls: 18,010 | realized win rate: 0.498

## Proper scoring + calibration (lower is better)

| model | Brier | log loss | ECE |
|---|---|---|---|
| pure-RRR logistic | 0.1404 | 0.4379 | 0.0773 |
| RRR Markov | 0.1637 | 0.5313 | 0.1627 |
| base Markov | 0.2040 | 0.8011 | 0.2235 |
| base-rate const | 0.2503 | 0.6937 | 0.0169 |

## base Markov: ECE sliced

By phase: death=0.1208, middle=0.2215, powerplay=0.2894

By wickets in hand: 1=0.0336, 2=0.0817, 3=0.0922, 4=0.1247, 5=0.1064, 6=0.1780, 7=0.2204, 8=0.1785, 9=0.3116, 10=0.3538

## RRR Markov: ECE sliced

By phase: death=0.0626, middle=0.1485, powerplay=0.2471

By wickets in hand: 1=0.0232, 2=0.0609, 3=0.0787, 4=0.1035, 5=0.0883, 6=0.1304, 7=0.1658, 8=0.1099, 9=0.2311, 10=0.2614

## Finding (why the RRR Markov trails the direct logistic)

The exact memoryless RRR Markov WP beats the constant base-rate but is beaten by a one-feature (RRR) logistic, and is systematically under-confident. This is not a DP bug -- a Monte-Carlo simulation of the same independent process reproduces the DP's WP to sampling noise. Two real effects, both spec-anticipated:

1. **Memoryless tail-thinning (structural).** Even in-sample the RRR Markov under-predicts: mean WP 0.443 vs actual win rate 0.515 (ECE 0.082). Independent per-ball draws thin the scoring right-tail, so hard chases look harder than they are. This is exactly the spec 00 hidden-state error appearing at the mean, and it motivates M5.

2. **Scoring-era shift.** Mean runs/legal-ball is 1.344 in training vs 1.598 in the held-out seasons (2024-2026, the highest-scoring in IPL history). The model, fit mostly on lower-scoring cricket, under-predicts modern chases; the direct logistic on RRR is more era-robust because RRR already encodes difficulty.
