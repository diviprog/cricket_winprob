# Player-level uncertainty -- match-clustered bootstrap CIs

95% percentile intervals from 2000 bootstrap replicates per player, resampling whole MATCHES (= innings) with replacement, seed 0. Per-ball resampling would assume away the within-innings dependence this project measured (dependence_decomposition.md: lag-1 +0.044 vs null +0.008); resampling whole innings preserves it and requires only exchangeability across a player's matches. The WP surface, LI, and the state-conditional drift correction are held fixed: these intervals are sampling variability of each player's career GIVEN the surfaces, not model uncertainty (the +0.037 autocorrelation bound is the surface's, shared by all players). Career gate >= 500 balls (86 batters, 80 bowlers); clutch-split gate >= 100 high-LI balls. Rank-adjacency uses a CI on the DIFFERENCE (independent replicate streams differenced), not CI overlap.

## Findings

- **Adjacent ranks are not separable.** Across the four top-10 tables, 0 of 36 adjacent-rank difference CIs exclude zero -- no neighbouring pair in any leaderboard is statistically distinguishable at 95%.
- **Rank-1 vs rank-10**: LI bat: +0.206 [-0.250, +0.677] (not separated); LI bowl: +0.294 [-0.119, +0.703] (not separated); WPA bat: +2.646 [-2.921, +7.979] (not separated); WPA bowl: +2.808 [-2.094, +7.733] (not separated).
- **Role-level separation is decisive**: HH Pandya vs VR Iyer (top finisher vs lowest-LI qualified anchor) differs by +0.692 [+0.332, +1.078] -- clearly separated. The leaderboards resolve ROLES, not neighbouring individuals.
- **Career WPA totals**: 5 of the top-10 batting and 8 of the top-10 bowling de-drifted totals exclude zero.
- **Clutch splits**: of 17 batters with >= 100 high-LI balls, 5 have a high-minus-low split CI excluding zero (AB de Villiers, MS Dhoni, KD Karthik, SK Raina, SA Yadav); of 22 bowlers, 1 (SL Malinga).
- **MS Dhoni**: high +0.0168 vs low -0.0010 per ball (346 high-LI balls); split +0.0179 [+0.0042, +0.0317] -- significant at 95%.
- **SA Yadav**: high -0.0103 vs low +0.0029 per ball (106 high-LI balls); split -0.0132 [-0.0305, -0.0010] -- significant at 95%.

## Mean leverage faced -- top 10 batters

| player | balls | matches | mean LI [95% CI] |
|---|---|---|---|
| HH Pandya | 827 | 68 | +1.400 [+1.090, +1.723] |
| MS Dhoni | 1714 | 99 | +1.391 [+1.151, +1.641] |
| R Tewatia | 535 | 42 | +1.390 [+0.936, +1.863] |
| DJ Bravo | 578 | 54 | +1.380 [+1.041, +1.820] |
| BJ Hodge | 537 | 29 | +1.370 [+0.932, +1.785] |
| KA Pollard | 1020 | 72 | +1.340 [+1.111, +1.553] |
| AB de Villiers | 1434 | 80 | +1.209 [+0.985, +1.442] |
| RG Sharma | 2577 | 131 | +1.206 [+1.049, +1.364] |
| DA Miller | 1278 | 67 | +1.205 [+0.957, +1.470] |
| AD Russell | 632 | 51 | +1.194 [+0.861, +1.530] |

Adjacent-rank differences:

| adjacent pair | diff [95% CI] | separated |
|---|---|---|
| HH Pandya vs MS Dhoni | +0.009 [-0.398, +0.409] | no |
| MS Dhoni vs R Tewatia | +0.001 [-0.542, +0.518] | no |
| R Tewatia vs DJ Bravo | +0.009 [-0.625, +0.595] | no |
| DJ Bravo vs BJ Hodge | +0.010 [-0.558, +0.632] | no |
| BJ Hodge vs KA Pollard | +0.030 [-0.449, +0.517] | no |
| KA Pollard vs AB de Villiers | +0.131 [-0.190, +0.444] | no |
| AB de Villiers vs RG Sharma | +0.003 [-0.291, +0.295] | no |
| RG Sharma vs DA Miller | +0.001 [-0.305, +0.290] | no |
| DA Miller vs AD Russell | +0.011 [-0.418, +0.424] | no |

## Mean leverage bowled -- top 10 bowlers

| player | balls | matches | mean LI [95% CI] |
|---|---|---|---|
| SM Curran | 545 | 28 | +1.504 [+1.228, +1.767] |
| TG Southee | 552 | 25 | +1.456 [+1.164, +1.801] |
| Avesh Khan | 797 | 38 | +1.421 [+1.081, +1.812] |
| R Vinay Kumar | 1071 | 54 | +1.342 [+1.067, +1.647] |
| Arshdeep Singh | 854 | 41 | +1.339 [+1.072, +1.610] |
| S Kaul | 569 | 26 | +1.334 [+1.023, +1.682] |
| IK Pathan | 783 | 43 | +1.289 [+0.994, +1.639] |
| A Nehra | 923 | 43 | +1.267 [+1.017, +1.543] |
| Shakib Al Hasan | 632 | 32 | +1.263 [+1.098, +1.417] |
| R Bhatia | 688 | 39 | +1.210 [+0.906, +1.542] |

Adjacent-rank differences:

| adjacent pair | diff [95% CI] | separated |
|---|---|---|
| SM Curran vs TG Southee | +0.048 [-0.386, +0.431] | no |
| TG Southee vs Avesh Khan | +0.035 [-0.432, +0.494] | no |
| Avesh Khan vs R Vinay Kumar | +0.079 [-0.358, +0.567] | no |
| R Vinay Kumar vs Arshdeep Singh | +0.004 [-0.389, +0.395] | no |
| Arshdeep Singh vs S Kaul | +0.005 [-0.400, +0.409] | no |
| S Kaul vs IK Pathan | +0.045 [-0.405, +0.511] | no |
| IK Pathan vs A Nehra | +0.023 [-0.369, +0.451] | no |
| A Nehra vs Shakib Al Hasan | +0.004 [-0.279, +0.302] | no |
| Shakib Al Hasan vs R Bhatia | +0.053 [-0.323, +0.397] | no |

## Career de-drifted WPA -- top 10 batters

CI on the total scales the per-ball-rate replicates by observed career balls: volume is treated as known, the rate as uncertain.

| player | balls | total WPA_dd [95% CI] | excl. 0 |
|---|---|---|---|
| AB de Villiers | 1434 | +5.45 [+0.96, +9.90] | yes |
| JC Buttler | 1338 | +4.82 [+1.51, +7.79] | yes |
| MS Dhoni | 1714 | +4.40 [-0.93, +9.97] | no |
| KA Pollard | 1020 | +3.78 [-0.23, +7.89] | no |
| Shubman Gill | 1403 | +3.66 [+1.07, +6.39] | yes |
| N Pooran | 683 | +3.28 [+0.96, +5.92] | yes |
| DR Smith | 861 | +3.07 [+0.69, +5.81] | yes |
| DA Miller | 1278 | +3.05 [-1.26, +7.42] | no |
| SS Iyer | 1391 | +2.91 [-0.32, +6.03] | no |
| KD Karthik | 1576 | +2.80 [-0.70, +5.91] | no |

Adjacent-rank differences:

| adjacent pair | diff [95% CI] | separated |
|---|---|---|
| AB de Villiers vs JC Buttler | +0.63 [-4.81, +6.35] | no |
| JC Buttler vs MS Dhoni | +0.42 [-6.04, +6.57] | no |
| MS Dhoni vs KA Pollard | +0.61 [-6.02, +7.51] | no |
| KA Pollard vs Shubman Gill | +0.13 [-4.60, +4.90] | no |
| Shubman Gill vs N Pooran | +0.38 [-3.77, +3.87] | no |
| N Pooran vs DR Smith | +0.21 [-3.35, +3.63] | no |
| DR Smith vs DA Miller | +0.02 [-4.95, +5.02] | no |
| DA Miller vs SS Iyer | +0.13 [-5.11, +5.43] | no |
| SS Iyer vs KD Karthik | +0.11 [-4.51, +4.78] | no |

## Career de-drifted WPA -- top 10 bowlers

| player | balls | total WPA_dd [95% CI] | excl. 0 |
|---|---|---|---|
| SP Narine | 1887 | +5.69 [+2.41, +8.81] | yes |
| R Ashwin | 2511 | +5.55 [+2.40, +8.54] | yes |
| SL Malinga | 1477 | +4.76 [+2.13, +7.33] | yes |
| Harbhajan Singh | 1730 | +4.48 [+1.68, +7.61] | yes |
| Rashid Khan | 1491 | +4.09 [+0.98, +7.06] | yes |
| Z Khan | 978 | +3.98 [+1.62, +6.98] | yes |
| B Kumar | 2139 | +3.39 [-0.91, +7.75] | no |
| Kuldeep Yadav | 858 | +3.09 [+1.20, +5.29] | yes |
| YS Chahal | 1994 | +2.89 [+0.27, +5.55] | yes |
| DW Steyn | 1204 | +2.88 [-1.02, +6.60] | no |

Adjacent-rank differences:

| adjacent pair | diff [95% CI] | separated |
|---|---|---|
| SP Narine vs R Ashwin | +0.14 [-4.27, +4.62] | no |
| R Ashwin vs SL Malinga | +0.79 [-3.38, +4.83] | no |
| SL Malinga vs Harbhajan Singh | +0.27 [-3.80, +4.00] | no |
| Harbhajan Singh vs Rashid Khan | +0.39 [-3.76, +4.68] | no |
| Rashid Khan vs Z Khan | +0.12 [-4.21, +4.10] | no |
| Z Khan vs B Kumar | +0.59 [-4.46, +6.03] | no |
| B Kumar vs Kuldeep Yadav | +0.29 [-4.72, +5.28] | no |
| Kuldeep Yadav vs YS Chahal | +0.20 [-3.05, +3.55] | no |
| YS Chahal vs DW Steyn | +0.00 [-4.70, +4.63] | no |

## Clutch splits with intervals -- top 10 batter splits

De-drifted per-ball WPA on high-LI (>= 2) minus ordinary balls. A replicate that resamples zero high-LI balls is dropped (counts below are negligible for every listed player).

| player | n_high | high | low | high - low [95% CI] | excl. 0 |
|---|---|---|---|---|---|
| AB de Villiers | 166 | +0.0285 | +0.0006 | +0.0280 [+0.0112, +0.0411] | yes |
| R Tewatia | 134 | +0.0193 | -0.0005 | +0.0198 [-0.0014, +0.0440] | no |
| MS Dhoni | 346 | +0.0168 | -0.0010 | +0.0179 [+0.0042, +0.0317] | yes |
| KD Karthik | 200 | +0.0143 | -0.0000 | +0.0144 [+0.0033, +0.0256] | yes |
| SK Raina | 115 | +0.0142 | +0.0003 | +0.0139 [+0.0021, +0.0257] | yes |
| RA Jadeja | 192 | +0.0125 | -0.0012 | +0.0138 [-0.0033, +0.0301] | no |
| KA Pollard | 214 | +0.0143 | +0.0009 | +0.0134 [-0.0019, +0.0290] | no |
| SPD Smith | 106 | +0.0102 | -0.0023 | +0.0125 [-0.0091, +0.0281] | no |
| DA Miller | 199 | +0.0120 | +0.0006 | +0.0114 [-0.0046, +0.0260] | no |
| AT Rayudu | 130 | +0.0109 | -0.0004 | +0.0113 [-0.0090, +0.0324] | no |

## Clutch splits with intervals -- top 10 bowler splits

| player | n_high | high | low | high - low [95% CI] | excl. 0 |
|---|---|---|---|---|---|
| SL Malinga | 142 | +0.0161 | +0.0019 | +0.0142 [+0.0004, +0.0264] | yes |
| SP Narine | 153 | +0.0103 | +0.0024 | +0.0079 [-0.0049, +0.0195] | no |
| YS Chahal | 136 | +0.0067 | +0.0011 | +0.0057 [-0.0058, +0.0176] | no |
| R Ashwin | 113 | +0.0070 | +0.0020 | +0.0050 [-0.0056, +0.0147] | no |
| DJ Bravo | 212 | +0.0054 | +0.0007 | +0.0047 [-0.0082, +0.0174] | no |
| Arshdeep Singh | 148 | +0.0027 | -0.0017 | +0.0044 [-0.0123, +0.0196] | no |
| JJ Bumrah | 232 | +0.0045 | +0.0012 | +0.0033 [-0.0095, +0.0149] | no |
| TA Boult | 117 | +0.0033 | +0.0002 | +0.0031 [-0.0149, +0.0225] | no |
| MM Sharma | 122 | +0.0033 | +0.0008 | +0.0025 [-0.0183, +0.0210] | no |
| HV Patel | 146 | +0.0011 | -0.0012 | +0.0023 [-0.0093, +0.0151] | no |

## Caveats

- Difference CIs treat two careers as independent; players sharing innings induce a small positive correlation, so those CIs are mildly conservative (true intervals slightly narrower).
- Intervals condition on the fitted WP/LI surfaces and the drift correction; surface-level bias is shared across players and bounded separately (leverage_validation.md, tail_diagnostics.md).
- Percentile intervals; with >= 40 matches per qualified career the normal-range diagnostics are unremarkable and BCa refinements would not change any verdict above.
