# Leverage Index -- validation (M3, spec 04/05 Part 2)

Leverage is built on the **era-adjusted RRR Markov WP** (half-life 0.75), fit on the train split and used to score all 130,029 balls. This is the martingale-preserving surface; the M2.5 recalibrated WP was rejected for leverage because a post-hoc map breaks `WP(s)=sum_o p_o WP(s')` and the MAD-of-a-martingale meaning of swing.

Reference `avg_swing` = **0.0170** WP/ball, averaged over the real balls that occurred (LI = swing / avg_swing, so the average real ball has LI = 1).

## Closed-form sanity check

Under a synthetic model where the ball ends the match as a coin flip, `swing((1,1,1))` should approach the theoretical max 2p(1-p) = **0.500**. Got **0.5000**. The swing machinery is correct.

## Martingale drift (spec 05)

Mean signed one-ball WP change over the real sequences (path includes the final live-state -> realized-y step). A self-consistent martingale drifts ~0; systematic drift would mean the dispersion leverage reads off is biased. Overall mean signed change = **+0.00048**.

| bucket | n | mean_from | mean_signed_change |
|---|---|---|---|
| overall | 130029 | 0.46008 | 0.00048 |
| [0.0,0.1) | 31205 | 0.02856 | 0.00015 |
| [0.1,0.2) | 12171 | 0.14626 | 0.00016 |
| [0.2,0.3) | 9828 | 0.24999 | 0.00031 |
| [0.3,0.4) | 9382 | 0.34931 | 0.00123 |
| [0.4,0.5) | 8431 | 0.44996 | 0.00185 |
| [0.5,0.6) | 8752 | 0.54966 | 0.00057 |
| [0.6,0.7) | 8757 | 0.65185 | 0.00136 |
| [0.7,0.8) | 8918 | 0.7502 | 0.00033 |
| [0.8,0.9) | 10091 | 0.85122 | 0.0004 |
| [0.9,1.0) | 22494 | 0.97026 | 7e-05 |

## Lag-1 autocorrelation -- the honest error bound (spec 05)

Residual lag-1 autocorrelation of runs-per-ball within partnerships, after removing the (phase, w) state mean: **+0.0368** over 122,403 within-partnership ball pairs. The Markov model implies 0 (every ball an independent draw from p_o(state)); the residual is the hidden-state error from spec 00 and bounds how much the leverage numbers should be distrusted. It is the justification metric for an M5 hidden state, which must reduce this to be worth promoting.

## Leverage by phase / over (is the death really the leveraged part?)

| phase | mean | count |
|---|---|---|
| powerplay | 0.935 | 41829 |
| middle | 0.788 | 61580 |
| death | 1.592 | 26620 |

Mean LI by over (0-indexed):

```
over
0     0.978
1     0.958
2     0.932
3     0.920
4     0.911
5     0.909
6     0.723
7     0.729
8     0.745
9     0.761
10    0.774
11    0.780
12    0.807
13    0.853
14    0.933
15    1.268
16    1.313
17    1.455
18    1.756
19    2.672
```

## Intuition ranking (spec 05 qualitative gate)

**Highest-leverage balls** -- should be tight, late chases:

| season | over | b | w | r | outcome | wp | li | batter | bowler |
|---|---|---|---|---|---|---|---|---|---|
| 2012 | 19 | 1 | 6 | 3 | 6 | 0.331 | 24.02 | SS Tiwary | A Nehra |
| 2021 | 19 | 1 | 6 | 3 | 0 | 0.331 | 24.02 | FA Allen | Kartik Tyagi |
| 2022 | 19 | 1 | 5 | 3 | 6 | 0.306 | 23.38 | Rashid Khan | M Jansen |
| 2012 | 19 | 1 | 5 | 3 | 6 | 0.306 | 23.38 | RG Sharma | DT Christian |
| 2016 | 19 | 1 | 6 | 4 | 2 | 0.275 | 23.34 | MP Stoinis | CJ Jordan |
| 2012 | 19 | 1 | 8 | 2 | W | 0.588 | 23.09 | AM Rahane | UT Yadav |
| 2019 | 19 | 1 | 4 | 3 | 6 | 0.306 | 22.77 | MJ Santner | BA Stokes |
| 2023 | 19 | 1 | 4 | 3 | 3 | 0.306 | 22.77 | Sikandar Raza | M Pathirana |
| 2014 | 19 | 1 | 5 | 4 | 1 | 0.259 | 22.47 | JA Morkel | R Vinay Kumar |
| 2011 | 19 | 1 | 5 | 4 | 6 | 0.259 | 22.47 | AT Rayudu | L Balaji |
| 2016 | 19 | 1 | 5 | 4 | 2 | 0.259 | 22.47 | CH Morris | DJ Bravo |
| 2017 | 19 | 1 | 5 | 4 | W | 0.259 | 22.47 | DT Christian | MG Johnson |
| 2025 | 19 | 1 | 5 | 4 | 1 | 0.259 | 22.47 | S Dube | Yash Dayal |
| 2013 | 19 | 1 | 5 | 4 | 1 | 0.259 | 22.47 | KA Pollard | R Vinay Kumar |
| 2025 | 19 | 1 | 5 | 4 | 1 | 0.259 | 22.47 | SB Dubey | Avesh Khan |

**Lowest-leverage balls** (live, >=1 over left) -- should be blowouts / dead states:

| season | over | b | w | r | outcome | wp | li | batter | bowler |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | 16 | 22 | 7 | 2 | 5 | 1.0 | 0.0 | MP Stoinis | DL Chahar |
| 2019 | 18 | 11 | 5 | 53 | W | 0.0 | 0.0 | KL Rahul | KK Ahmed |
| 2019 | 18 | 12 | 5 | 56 | 2 | 0.0 | 0.0 | KL Rahul | KK Ahmed |
| 2019 | 17 | 13 | 5 | 62 | 6 | 0.0 | 0.0 | P Simran Singh | B Kumar |
| 2019 | 17 | 14 | 5 | 62 | 0 | 0.0 | 0.0 | P Simran Singh | B Kumar |
| 2019 | 17 | 15 | 5 | 64 | 3 | 0.0 | 0.0 | P Simran Singh | B Kumar |
| 2019 | 17 | 16 | 5 | 66 | 1 | 0.0 | 0.0 | KL Rahul | B Kumar |
| 2024 | 14 | 34 | 5 | 106 | 0 | 0.0 | 0.0 | Abdul Samad | SN Thakur |
| 2024 | 14 | 33 | 5 | 106 | 1 | 0.0 | 0.0 | Abdul Samad | SN Thakur |
| 2024 | 14 | 32 | 5 | 105 | 1 | 0.0 | 0.0 | H Klaasen | SN Thakur |
