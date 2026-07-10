# Win Probability Added (WPA) -- clutch attribution

WPA = realized signed one-ball WP change, credited +to the batter and -to the bowler, on the era-adjusted RRR Markov WP. Because that WP is an exact martingale (E[WPA|state]=0), positive WPA means BEATING the model's expectation for the state, automatically weighted by the leverage of the moment. Career threshold >= 500 balls; 'high leverage' = LI >= 2.0 (>= 2.0x the average ball). **All tables rank by de-drifted WPA (`_dd`); see below.**

*Sanity:* per-ball batter+bowler WPA cancels to <= 0.00e+00 (zero-sum); raw total batting WPA +62.30 equals the telescoped innings sum y - wp(first ball) = +62.30. De-drifting preserves the zero-sum (<= 0.00e+00) and drives the grand batting total to +0.00e+00 (the +62 model surplus removed).

## Why de-drifted WPA

The WP surface is not perfectly calibrated: batting sides collectively finish **+62.3 wins** above the model's start-of-chase expectation (the memoryless tail-thinning of validation_report.md plus the modern scoring-era shift). In a zero-sum matchup that surplus cannot be real skill -- it is a **~+0.00048 WP-per-ball baseline** handed to every batter and taken from every bowler, which makes raw batter-vs-bowler totals incomparable. It is removed state-conditionally (the drift ranges ~0.00007-0.00185 across WP regimes, so a flat subtraction would misprice players by where they operate): subtract the mean batting WPA among balls starting at a similar WP. Any player is a sliver of each WP bin, so this strips the population baseline, not individual edge.

Concretely, the largest single shifts here: **V Kohli** (batting) falls +3.83 -> +2.25 once the free baseline is removed, while **R Ashwin** (bowling) rises +4.21 -> +5.55. Never compare a raw batter total to a raw bowler total; compare `total_wpa_dd`.

## Batsmen -- career WPA (de-drifted; total win probability added when chasing)

`total_wpa` = raw (shown for transparency); `total_wpa_dd` = de-drifted, the fair rank; `wpa_dd_per100` = de-drifted per 100 balls; `wpa_high_dd` / `wpa_low_dd` = de-drifted per-ball WPA in high- vs ordinary-leverage moments. A genuine clutch player has `wpa_high_dd` > 0 and ideally > `wpa_low_dd`.

| player | balls | total_wpa | total_wpa_dd | wpa_dd_per100 | mean_li | n_high | wpa_high_dd | wpa_low_dd |
|---|---|---|---|---|---|---|---|---|
| AB de Villiers | 1434 | 6.1538 | 5.4491 | 0.38 | 1.2089 | 166 | 0.0285 | 0.0006 |
| JC Buttler | 1338 | 5.6344 | 4.8201 | 0.3602 | 1.0874 | 75 | 0.0292 | 0.0021 |
| MS Dhoni | 1714 | 5.0443 | 4.3962 | 0.2565 | 1.391 | 346 | 0.0168 | -0.001 |
| KA Pollard | 1020 | 4.1903 | 3.7832 | 0.3709 | 1.3396 | 214 | 0.0143 | 0.0009 |
| Shubman Gill | 1403 | 4.4634 | 3.6569 | 0.2607 | 0.9829 | 53 | 0.03 | 0.0015 |
| N Pooran | 683 | 3.5452 | 3.2816 | 0.4805 | 0.9267 | 62 | 0.0276 | 0.0025 |
| DR Smith | 861 | 3.679 | 3.0713 | 0.3567 | 1.1478 | 39 | 0.02 | 0.0028 |
| DA Miller | 1278 | 3.5053 | 3.0476 | 0.2385 | 1.2054 | 199 | 0.012 | 0.0006 |
| SS Iyer | 1391 | 3.7109 | 2.9133 | 0.2094 | 1.0231 | 93 | 0.0159 | 0.0011 |
| KD Karthik | 1576 | 3.5735 | 2.803 | 0.1779 | 1.1098 | 200 | 0.0143 | -0.0 |
| DA Warner | 2300 | 3.9354 | 2.7235 | 0.1184 | 0.8814 | 44 | 0.0013 | 0.0012 |
| YK Pathan | 1205 | 3.2126 | 2.6942 | 0.2236 | 1.17 | 189 | 0.0071 | 0.0013 |
| AD Russell | 632 | 2.8683 | 2.667 | 0.422 | 1.1945 | 133 | 0.0108 | 0.0025 |
| RR Pant | 1197 | 3.2115 | 2.6438 | 0.2209 | 0.9242 | 72 | 0.0155 | 0.0014 |
| SA Yadav | 1350 | 3.2402 | 2.4983 | 0.1851 | 1.0208 | 106 | -0.0103 | 0.0029 |
| RG Sharma | 2577 | 3.9438 | 2.3819 | 0.0924 | 1.2061 | 252 | 0.0092 | 0.0 |
| R Tewatia | 535 | 2.5432 | 2.3766 | 0.4442 | 1.3898 | 134 | 0.0193 | -0.0005 |
| V Kohli | 2978 | 3.8271 | 2.2468 | 0.0754 | 0.8729 | 89 | -0.0018 | 0.0008 |
| CH Gayle | 1362 | 2.9074 | 2.1939 | 0.1611 | 0.8499 | 22 | 0.0485 | 0.0008 |
| SK Raina | 1676 | 2.9639 | 2.1197 | 0.1265 | 1.018 | 115 | 0.0142 | 0.0003 |

## Batsmen -- pure clutch rate (de-drifted per-ball WPA in high-leverage moments)

Ranked by `wpa_high_dd` among batters with >= 100 high-LI balls faced -- who actually converts the big moments, rate-adjusted so volume doesn't dominate. De-drifting matters most here: high-LI balls carry the heaviest drift, so the raw clutch rate was the most inflated number in the report.

| player | balls | n_high | wpa_high_dd | wpa_low_dd | total_wpa_dd |
|---|---|---|---|---|---|
| AB de Villiers | 1434 | 166 | 0.0285 | 0.0006 | 5.4491 |
| R Tewatia | 535 | 134 | 0.0193 | -0.0005 | 2.3766 |
| MS Dhoni | 1714 | 346 | 0.0168 | -0.001 | 4.3962 |
| KA Pollard | 1020 | 214 | 0.0143 | 0.0009 | 3.7832 |
| KD Karthik | 1576 | 200 | 0.0143 | -0.0 | 2.803 |
| SK Raina | 1676 | 115 | 0.0142 | 0.0003 | 2.1197 |
| RA Jadeja | 1167 | 192 | 0.0125 | -0.0012 | 1.1945 |
| DA Miller | 1278 | 199 | 0.012 | 0.0006 | 3.0476 |
| AT Rayudu | 1774 | 130 | 0.0109 | -0.0004 | 0.7493 |
| AD Russell | 632 | 133 | 0.0108 | 0.0025 | 2.667 |
| SPD Smith | 1051 | 106 | 0.0102 | -0.0023 | -1.1189 |
| RG Sharma | 2577 | 252 | 0.0092 | 0.0 | 2.3819 |
| HH Pandya | 827 | 181 | 0.0091 | -0.0014 | 0.7076 |
| YK Pathan | 1205 | 189 | 0.0071 | 0.0013 | 2.6942 |
| BJ Hodge | 537 | 107 | 0.0036 | -0.0018 | -0.3701 |
| KL Rahul | 1884 | 120 | 0.0026 | 0.001 | 2.06 |
| SA Yadav | 1350 | 106 | -0.0103 | 0.0029 | 2.4983 |

## Bowlers -- career WPA (de-drifted; win probability added by defending)

| player | balls | total_wpa | total_wpa_dd | wpa_dd_per100 | mean_li | n_high | wpa_high_dd | wpa_low_dd |
|---|---|---|---|---|---|---|---|---|
| SP Narine | 1887 | 4.8121 | 5.6923 | 0.3017 | 0.9657 | 153 | 0.0103 | 0.0024 |
| R Ashwin | 2511 | 4.2087 | 5.5474 | 0.2209 | 0.9484 | 113 | 0.007 | 0.002 |
| SL Malinga | 1477 | 4.1455 | 4.7584 | 0.3222 | 1.0193 | 142 | 0.0161 | 0.0019 |
| Harbhajan Singh | 1730 | 3.5726 | 4.4848 | 0.2592 | 0.8489 | 45 | 0.0183 | 0.0022 |
| Rashid Khan | 1491 | 3.4181 | 4.0947 | 0.2746 | 0.8736 | 91 | 0.0136 | 0.002 |
| Z Khan | 978 | 3.5566 | 3.9751 | 0.4064 | 0.9513 | 56 | 0.0081 | 0.0038 |
| B Kumar | 2139 | 2.3148 | 3.3859 | 0.1583 | 1.1733 | 219 | 0.0005 | 0.0017 |
| Kuldeep Yadav | 858 | 2.647 | 3.0922 | 0.3604 | 0.9427 | 43 | 0.0318 | 0.0021 |
| YS Chahal | 1994 | 2.0031 | 2.8893 | 0.1449 | 0.854 | 136 | 0.0067 | 0.0011 |
| DW Steyn | 1204 | 2.2487 | 2.8844 | 0.2396 | 1.1927 | 145 | 0.0032 | 0.0023 |
| JJ Bumrah | 1694 | 1.9281 | 2.7894 | 0.1647 | 1.1642 | 232 | 0.0045 | 0.0012 |
| SK Trivedi | 750 | 2.4541 | 2.6351 | 0.3514 | 0.6932 | 34 | 0.0317 | 0.0022 |
| PP Ojha | 1011 | 1.8096 | 2.2739 | 0.2249 | 0.9288 | 77 | 0.0051 | 0.002 |
| A Mishra | 1751 | 1.2585 | 2.2041 | 0.1259 | 1.0016 | 115 | 0.0024 | 0.0012 |
| M Muralitharan | 858 | 1.8465 | 2.184 | 0.2545 | 0.9123 | 53 | 0.013 | 0.0019 |
| DJ Bravo | 1503 | 1.6225 | 2.0873 | 0.1389 | 1.0726 | 212 | 0.0054 | 0.0007 |
| AR Patel | 1545 | 1.1925 | 1.9872 | 0.1286 | 0.804 | 56 | -0.0054 | 0.0015 |
| PJ Cummins | 837 | 1.5145 | 1.8578 | 0.222 | 0.7835 | 37 | 0.0401 | 0.0005 |
| I Sharma | 1273 | 1.1654 | 1.7479 | 0.1373 | 1.0102 | 62 | 0.0195 | 0.0004 |
| MJ McClenaghan | 608 | 1.3633 | 1.7364 | 0.2856 | 1.1276 | 59 | 0.0288 | 0.0001 |

## Bowlers -- pure clutch rate (de-drifted per-ball WPA in high-leverage moments)

| player | balls | n_high | wpa_high_dd | wpa_low_dd | total_wpa_dd |
|---|---|---|---|---|---|
| SL Malinga | 1477 | 142 | 0.0161 | 0.0019 | 4.7584 |
| SP Narine | 1887 | 153 | 0.0103 | 0.0024 | 5.6923 |
| R Ashwin | 2511 | 113 | 0.007 | 0.002 | 5.5474 |
| YS Chahal | 1994 | 136 | 0.0067 | 0.0011 | 2.8893 |
| DJ Bravo | 1503 | 212 | 0.0054 | 0.0007 | 2.0873 |
| JJ Bumrah | 1694 | 232 | 0.0045 | 0.0012 | 2.7894 |
| TA Boult | 1363 | 117 | 0.0033 | 0.0002 | 0.6403 |
| MM Sharma | 1221 | 122 | 0.0033 | 0.0008 | 1.3113 |
| DW Steyn | 1204 | 145 | 0.0032 | 0.0023 | 2.8844 |
| Arshdeep Singh | 854 | 148 | 0.0027 | -0.0017 | -0.8209 |
| RA Jadeja | 1902 | 106 | 0.0026 | 0.0007 | 1.5207 |
| A Mishra | 1751 | 115 | 0.0024 | 0.0012 | 2.2041 |
| HV Patel | 1280 | 146 | 0.0011 | -0.0012 | -1.2024 |
| B Kumar | 2139 | 219 | 0.0005 | 0.0017 | 3.3859 |
| P Kumar | 1177 | 110 | 0.0002 | 0.0002 | 0.2851 |
| Avesh Khan | 797 | 136 | -0.0019 | 0.0004 | 0.0254 |
| R Vinay Kumar | 1071 | 140 | -0.0038 | 0.0011 | 0.4869 |
| Mohammed Shami | 1245 | 101 | -0.0041 | -0.0007 | -1.176 |
| Sandeep Sharma | 1481 | 119 | -0.0046 | 0.0003 | -0.1771 |
| Mohammed Siraj | 1304 | 138 | -0.0046 | -0.0008 | -1.5679 |

## Reading these tables

- **total_wpa_dd** is the career clutch-value rank: it rewards being trusted with the moments AND delivering, with the model's calibration surplus removed so batters and bowlers sit on one scale.
- **wpa_high_dd vs wpa_low_dd** is the honest clutch signal: a player whose per-ball WPA is higher when leverage is high genuinely raises their game; one positive only in low-LI moments padded stats in dead situations.
- WPA scores the batter-vs-bowler matchup and inherits the model's +0.037 autocorrelation bound; de-drifting removes the first-order calibration bias but a second-order residual survives WP-binning, so treat the ranking as robust and the absolute decimals as approximate.
