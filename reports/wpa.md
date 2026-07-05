# Win Probability Added (WPA) -- clutch attribution

WPA = realized signed one-ball WP change, credited +to the batter and -to the bowler, on the era-adjusted RRR Markov WP. Because that WP is an exact martingale (E[WPA|state]=0), positive WPA means BEATING the model's expectation for the state, automatically weighted by the leverage of the moment. Career threshold >= 500 balls; 'high leverage' = LI >= 2.0 (>= 2.0x the average ball).

*Sanity:* per-ball batter+bowler WPA cancels to <= 0.00e+00 (zero-sum); total batting WPA +62.30 equals the telescoped innings sum y - wp(first ball) = +62.30 (wins minus start-of-chase expectation). Both hold, so the accounting is exact.

## Batsmen -- career WPA (total win probability added when chasing)

`total_wpa` = career WP added; `wpa_per100` = per 100 balls; `wpa_high` / `wpa_low` = per-ball WPA in high- vs ordinary-leverage moments. A genuine clutch player has `wpa_high` > 0 and ideally > `wpa_low`.

| player | balls | total_wpa | wpa_per100 | mean_li | n_high | wpa_high | wpa_low |
|---|---|---|---|---|---|---|---|
| AB de Villiers | 1434 | 6.1538 | 0.4291 | 1.2089 | 166 | 0.0293 | 0.001 |
| JC Buttler | 1338 | 5.6344 | 0.4211 | 1.0874 | 75 | 0.0299 | 0.0027 |
| MS Dhoni | 1714 | 5.0443 | 0.2943 | 1.391 | 346 | 0.0176 | -0.0008 |
| Shubman Gill | 1403 | 4.4634 | 0.3181 | 0.9829 | 53 | 0.0311 | 0.0021 |
| KA Pollard | 1020 | 4.1903 | 0.4108 | 1.3396 | 214 | 0.015 | 0.0012 |
| RG Sharma | 2577 | 3.9438 | 0.153 | 1.2061 | 252 | 0.0101 | 0.0006 |
| DA Warner | 2300 | 3.9354 | 0.1711 | 0.8814 | 44 | 0.0021 | 0.0017 |
| V Kohli | 2978 | 3.8271 | 0.1285 | 0.8729 | 89 | -0.0009 | 0.0014 |
| SS Iyer | 1391 | 3.7109 | 0.2668 | 1.0231 | 93 | 0.0167 | 0.0017 |
| DR Smith | 861 | 3.679 | 0.4273 | 1.1478 | 39 | 0.0207 | 0.0035 |
| KD Karthik | 1576 | 3.5735 | 0.2267 | 1.1098 | 200 | 0.015 | 0.0004 |
| N Pooran | 683 | 3.5452 | 0.5191 | 0.9267 | 62 | 0.0285 | 0.0029 |
| DA Miller | 1278 | 3.5053 | 0.2743 | 1.2054 | 199 | 0.0128 | 0.0009 |
| SA Yadav | 1350 | 3.2402 | 0.24 | 1.0208 | 106 | -0.0095 | 0.0034 |
| YK Pathan | 1205 | 3.2126 | 0.2666 | 1.17 | 189 | 0.0079 | 0.0017 |
| RR Pant | 1197 | 3.2115 | 0.2683 | 0.9242 | 72 | 0.0161 | 0.0018 |
| KL Rahul | 1884 | 3.1739 | 0.1685 | 1.0432 | 120 | 0.0035 | 0.0016 |
| SK Raina | 1676 | 2.9639 | 0.1768 | 1.018 | 115 | 0.0149 | 0.0008 |
| CH Gayle | 1362 | 2.9074 | 0.2135 | 0.8499 | 22 | 0.049 | 0.0014 |
| AD Russell | 632 | 2.8683 | 0.4538 | 1.1945 | 133 | 0.0115 | 0.0027 |

## Batsmen -- pure clutch rate (per-ball WPA in high-leverage moments)

Ranked by `wpa_high` among batters with >= 100 high-LI balls faced -- who actually converts the big moments, rate-adjusted so volume doesn't dominate:

| player | balls | n_high | wpa_high | wpa_low | total_wpa |
|---|---|---|---|---|---|
| AB de Villiers | 1434 | 166 | 0.0293 | 0.001 | 6.1538 |
| R Tewatia | 535 | 134 | 0.0201 | -0.0004 | 2.5432 |
| MS Dhoni | 1714 | 346 | 0.0176 | -0.0008 | 5.0443 |
| KA Pollard | 1020 | 214 | 0.015 | 0.0012 | 4.1903 |
| KD Karthik | 1576 | 200 | 0.015 | 0.0004 | 3.5735 |
| SK Raina | 1676 | 115 | 0.0149 | 0.0008 | 2.9639 |
| RA Jadeja | 1167 | 192 | 0.0134 | -0.001 | 1.6303 |
| DA Miller | 1278 | 199 | 0.0128 | 0.0009 | 3.5053 |
| AT Rayudu | 1774 | 130 | 0.0116 | 0.0001 | 1.6396 |
| AD Russell | 632 | 133 | 0.0115 | 0.0027 | 2.8683 |
| SPD Smith | 1051 | 106 | 0.0111 | -0.0018 | -0.5239 |
| RG Sharma | 2577 | 252 | 0.0101 | 0.0006 | 3.9438 |
| HH Pandya | 827 | 181 | 0.0099 | -0.001 | 1.1795 |
| YK Pathan | 1205 | 189 | 0.0079 | 0.0017 | 3.2126 |
| BJ Hodge | 537 | 107 | 0.0044 | -0.0011 | -0.0027 |
| KL Rahul | 1884 | 120 | 0.0035 | 0.0016 | 3.1739 |
| SA Yadav | 1350 | 106 | -0.0095 | 0.0034 | 3.2402 |

## Bowlers -- career WPA (win probability added by defending)

| player | balls | total_wpa | wpa_per100 | mean_li | n_high | wpa_high | wpa_low |
|---|---|---|---|---|---|---|---|
| SP Narine | 1887 | 4.8121 | 0.255 | 0.9657 | 153 | 0.0094 | 0.0019 |
| R Ashwin | 2511 | 4.2087 | 0.1676 | 0.9484 | 113 | 0.0062 | 0.0015 |
| SL Malinga | 1477 | 4.1455 | 0.2807 | 1.0193 | 142 | 0.0153 | 0.0015 |
| Harbhajan Singh | 1730 | 3.5726 | 0.2065 | 0.8489 | 45 | 0.0173 | 0.0017 |
| Z Khan | 978 | 3.5566 | 0.3637 | 0.9513 | 56 | 0.0075 | 0.0034 |
| Rashid Khan | 1491 | 3.4181 | 0.2292 | 0.8736 | 91 | 0.0127 | 0.0016 |
| Kuldeep Yadav | 858 | 2.647 | 0.3085 | 0.9427 | 43 | 0.0309 | 0.0016 |
| SK Trivedi | 750 | 2.4541 | 0.3272 | 0.6932 | 34 | 0.0312 | 0.0019 |
| B Kumar | 2139 | 2.3148 | 0.1082 | 1.1733 | 219 | -0.0002 | 0.0012 |
| DW Steyn | 1204 | 2.2487 | 0.1868 | 1.1927 | 145 | 0.0025 | 0.0018 |
| YS Chahal | 1994 | 2.0031 | 0.1005 | 0.854 | 136 | 0.006 | 0.0006 |
| JJ Bumrah | 1694 | 1.9281 | 0.1138 | 1.1642 | 232 | 0.0036 | 0.0008 |
| M Muralitharan | 858 | 1.8465 | 0.2152 | 0.9123 | 53 | 0.0123 | 0.0015 |
| PP Ojha | 1011 | 1.8096 | 0.179 | 0.9288 | 77 | 0.0041 | 0.0016 |
| DJ Bravo | 1503 | 1.6225 | 0.108 | 1.0726 | 212 | 0.0047 | 0.0005 |
| PJ Cummins | 837 | 1.5145 | 0.1809 | 0.7835 | 37 | 0.0393 | 0.0001 |
| MJ McClenaghan | 608 | 1.3633 | 0.2242 | 1.1276 | 59 | 0.028 | -0.0005 |
| Imran Tahir | 608 | 1.301 | 0.214 | 1.0177 | 71 | 0.0177 | 0.0001 |
| A Mishra | 1751 | 1.2585 | 0.0719 | 1.0016 | 115 | 0.0013 | 0.0007 |
| AR Patel | 1545 | 1.1925 | 0.0772 | 0.804 | 56 | -0.0062 | 0.001 |

## Bowlers -- pure clutch rate (per-ball WPA in high-leverage moments)

| player | balls | n_high | wpa_high | wpa_low | total_wpa |
|---|---|---|---|---|---|
| SL Malinga | 1477 | 142 | 0.0153 | 0.0015 | 4.1455 |
| SP Narine | 1887 | 153 | 0.0094 | 0.0019 | 4.8121 |
| R Ashwin | 2511 | 113 | 0.0062 | 0.0015 | 4.2087 |
| YS Chahal | 1994 | 136 | 0.006 | 0.0006 | 2.0031 |
| DJ Bravo | 1503 | 212 | 0.0047 | 0.0005 | 1.6225 |
| JJ Bumrah | 1694 | 232 | 0.0036 | 0.0008 | 1.9281 |
| TA Boult | 1363 | 117 | 0.0025 | -0.0003 | -0.0861 |
| DW Steyn | 1204 | 145 | 0.0025 | 0.0018 | 2.2487 |
| MM Sharma | 1221 | 122 | 0.0025 | 0.0004 | 0.7103 |
| Arshdeep Singh | 854 | 148 | 0.0018 | -0.0022 | -1.2611 |
| RA Jadeja | 1902 | 106 | 0.0017 | 0.0003 | 0.7368 |
| A Mishra | 1751 | 115 | 0.0013 | 0.0007 | 1.2585 |
| HV Patel | 1280 | 146 | 0.0005 | -0.0016 | -1.7426 |
| B Kumar | 2139 | 219 | -0.0002 | 0.0012 | 2.3148 |
| P Kumar | 1177 | 110 | -0.0005 | -0.0003 | -0.3826 |
| Avesh Khan | 797 | 136 | -0.0027 | -0.0 | -0.3877 |
| R Vinay Kumar | 1071 | 140 | -0.0046 | 0.0006 | -0.0792 |
| Mohammed Shami | 1245 | 101 | -0.0049 | -0.0012 | -1.8891 |
| Mohammed Siraj | 1304 | 138 | -0.0052 | -0.0013 | -2.2059 |
| Sandeep Sharma | 1481 | 119 | -0.0055 | -0.0002 | -0.9322 |

## Reading these tables

- **Total WPA** rewards both being trusted with the moments AND delivering; it is the career clutch-value rank.
- **wpa_high vs wpa_low** is the honest clutch signal: a player whose per-ball WPA is higher when leverage is high genuinely raises their game; one who is positive only in low-LI moments padded stats in dead situations.
- WPA scores the batter-vs-bowler matchup and inherits the model's +0.037 autocorrelation bound, so treat the ranking as robust and the absolute decimals as approximate.
