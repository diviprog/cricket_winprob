# M4 -- Fitted baselines vs the structured Markov WP (spec 06)

Held-out seasons: ['2024', '2025', '2026'] | 18,010 balls across 164 matches | realized win rate 0.498

XGBoost early-stopping validation = training season `2023` held out whole (8,090 balls); models fit on 103,929. Markov models and the logistic use the full train split, as in M2/M3.

## M4 gate table (held-out; lower is better)

| model | Brier | log loss | ECE |
|---|---|---|---|
| logistic (spec06) | 0.1350 | 0.4157 | 0.0700 |
| XGB (b,w,r) | 0.1414 | 0.4374 | 0.0771 |
| XGB (b,w,r)+striker | 0.1412 | 0.4378 | 0.0818 |
| pure-RRR logistic | 0.1404 | 0.4379 | 0.0773 |
| XGB (full) | 0.1490 | 0.4607 | 0.0847 |
| RRR Markov (era 0.75) | 0.1489 | 0.4741 | 0.1310 |
| RRR Markov | 0.1637 | 0.5313 | 0.1627 |
| base-rate const | 0.2503 | 0.6937 | 0.0169 |
| base Markov | 0.2040 | 0.8011 | 0.2235 |

## Paired match-clustered bootstrap on log loss

Delta = mean log loss(B) - mean log loss(A); negative means **B is better**. 1000 resamples over whole matches (balls within a match share a label, so per-ball CIs would be ~sqrt(120) too narrow). 'sig' = 95% CI excludes 0.

| question | A | B | delta | 95% CI | sig |
|---|---|---|---|---|---|
| Structure loss: does an unconstrained fit on the SAME state beat the Markov WP? | RRR Markov | XGB (b,w,r) | -0.09388 | [-0.17375, -0.02557] | yes |
| M5 licensing: does `striker_balls` add signal beyond (b,w,r)? | XGB (b,w,r) | XGB (b,w,r)+striker | +0.00041 | [-0.00322, +0.00345] | no |
| Extra features: does target/crr/over/phase add signal beyond state+striker? | XGB (b,w,r)+striker | XGB (full) | +0.02291 | [-0.02533, +0.07824] | no |
| Spec 06 headline: unstructured vs structured, unrestricted features. | RRR Markov | XGB (full) | -0.07056 | [-0.13987, -0.00469] | yes |
| Does the era adjustment help on held-out seasons? | RRR Markov | RRR Markov (era 0.75) | -0.05719 | [-0.07918, -0.03542] | yes |

## Why the richest XGBoost loses: era overfitting, not a bug

Mean chase `target` is **166.9** in training vs **194.0** in the held-out seasons -- the same era shift M2.5 was built for. `XGB (full)` is the only fit that sees `target`/`crr`, and it spends its capacity carving up 2008-2023 scoring conditions that no longer exist. Its TRAINING log loss is far the best and its held-out log loss far the worst. Early stopping cannot catch this: the validation season (2023) is on the training side of the shift.

| XGBoost fit | best iter | train LL | val LL | held-out LL |
|---|---|---|---|---|
| XGB (b,w,r) | 90 | 0.4547 | 0.4975 | 0.4374 |
| XGB (b,w,r)+striker | 73 | 0.4553 | 0.4969 | 0.4378 |
| XGB (full) | 93 | 0.4096 | 0.5020 | 0.4607 |

Note the XGBoost fits are *handicapped* relative to the Markov and logistic models: they lose the 2023 season to early stopping, and 2023 is the training season closest to the held-out era. The structure-loss result below is therefore conservative -- XGB wins despite the handicap.

## Where the structured model lags (log loss by phase)

| model | powerplay | middle | death |
|---|---|---|---|
| RRR Markov | 0.7237 | 0.5023 | 0.2773 |
| logistic (spec06) | 0.5287 | 0.4025 | 0.2572 |
| XGB (b,w,r) | 0.5336 | 0.4237 | 0.3087 |
| XGB (full) | 0.5555 | 0.4627 | 0.2956 |

## Findings

**0. The best model here is the plain logistic** (log loss 0.4157, beating every XGBoost variant). Spec 06 predicted 'XGBoost will likely win on raw Brier/log loss.' It does not, and the reason is the same era shift: under a covariate shift this large, the rigid model is the robust one. Capacity is a liability here, not an asset.

**1. The gap is structure/calibration, not missing state.** Given the *identical* `(b,w,r)`, an unconstrained fit beats the RRR Markov WP by 0.094 nats (95% CI [0.026, 0.174]). The Markov WP's ECE is 0.163 against 0.077: it is not missing information, it is *under-confident* with what it has. That is the memoryless tail-thinning already documented in validation_report.md -- independent per-ball draws thin the scoring right tail, so hard chases look harder than they are.

**2. M5 is NOT licensed.** Adding `striker_balls` moves held-out log loss by +0.00041 nats, CI [-0.00322, +0.00345] -- a CI straddling zero and two orders of magnitude smaller than the structure gap. A direct label fit is the most flexible possible use of the feature; if the signal were there, XGBoost would find it. Spec 07 gates M5 on reducing held-out log loss AND residual autocorrelation. It fails the first half outright.

This does not say the +0.037 lag-1 autocorrelation is illusory. It says the autocorrelation does not translate into *win-probability* signal: knowing the striker is set shifts the next-ball outcome distribution slightly, but that edge washes out across the ~60 balls left in a chase. The Markov approximation is adequate at this granularity -- spec 07: 'a negative result here is a real finding, not a failure.'

**3. Where to spend effort instead.** The era adjustment (half-life 0.75) is the single largest structured-model win in this table (0.057 nats, CI excludes zero), and it closes only part of the calibration gap. Recovering the rest means attacking the tail-thinning in the *outcome model* -- correlated per-ball draws or an over-dispersed outcome distribution -- not widening the state.

## Caveat that survives any result here

A fitted WP model does not give leverage for free (spec 06). It predicts WP but not the next-ball outcome distribution, so it cannot produce `swing`, and it is not a martingale -- WPA on top of it would not satisfy E[WPA|state]=0. Leverage and WPA stay on the Markov outcome model regardless of which row wins the table above. What this table decides is whether the WP *surface* those metrics ride on should be widened (M5).
