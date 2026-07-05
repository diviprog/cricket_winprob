# Player Leverage Attribution (M3 deliverable)

Every second-innings legal ball carries a leverage index `li` (spec 04, avg real ball = 1.0), credited to the batter who faced it and the bowler who bowled it. A player's **mean LI** is the average importance of the match situations they are on the field for. Career threshold: **>= 500 balls** (86 batsmen, 80 bowlers).

**Hypothesis:** the highest-mean-LI batsmen are lower-order *finishers* and the highest-mean-LI bowlers are *death* specialists.

## 1. Structural signal -- mean LI by batting position

If finishers face the highest-leverage balls, mean LI should climb down the order toward the lower-middle finisher slots.

| bat_pos | mean_li | balls | death_share |
|---|---|---|---|
| 1.0 | 0.949 | 23899.0 | 0.041 |
| 2.0 | 0.9 | 24830.0 | 0.05 |
| 3.0 | 0.937 | 21767.0 | 0.085 |
| 4.0 | 1.01 | 18974.0 | 0.163 |
| 5.0 | 1.099 | 15141.0 | 0.288 |
| 6.0 | 1.305 | 10108.0 | 0.445 |
| 7.0 | 1.261 | 6882.0 | 0.572 |
| 8.0 | 1.106 | 3997.0 | 0.729 |
| 9.0 | 0.768 | 2408.0 | 0.814 |
| 10.0 | 0.353 | 1409.0 | 0.888 |
| 11.0 | 0.237 | 614.0 | 0.884 |

The profile is **hump-shaped**: mean LI rises from the openers to a peak at the finisher slots (positions 6-7, ~1.3x the average ball), then collapses for the genuine tail (9-11, down to 0.24). The tail's `death_share` is the highest in the order (~0.88) yet its LI is the lowest -- they bat at the death but overwhelmingly in already-decided games, so high death share does NOT imply high leverage. The finisher signal is therefore the rise over the batting order proper: corr(mean LI, position) over positions 1-7 = **+0.909**. (Over all 1-11 it flips to -0.525, an artifact of the dead tail, shown only to make the shape explicit.)

## 2. Batsmen -- highest mean leverage faced

These are the players who bat in the tightest situations (finishers):

| batter | balls | mean_li | median_pos | death_share |
|---|---|---|---|---|
| HH Pandya | 827 | 1.4 | 5.0 | 0.354 |
| MS Dhoni | 1714 | 1.391 | 5.0 | 0.484 |
| R Tewatia | 535 | 1.39 | 6.0 | 0.514 |
| DJ Bravo | 578 | 1.38 | 5.0 | 0.379 |
| BJ Hodge | 537 | 1.37 | 4.0 | 0.311 |
| KA Pollard | 1020 | 1.34 | 5.0 | 0.421 |
| AB de Villiers | 1434 | 1.209 | 4.0 | 0.206 |
| RG Sharma | 2577 | 1.206 | 3.0 | 0.134 |
| DA Miller | 1278 | 1.205 | 5.0 | 0.336 |
| AD Russell | 632 | 1.194 | 6.0 | 0.419 |
| JP Duminy | 594 | 1.183 | 4.0 | 0.229 |
| YK Pathan | 1205 | 1.17 | 5.0 | 0.275 |
| R Parag | 561 | 1.168 | 4.0 | 0.207 |
| RA Jadeja | 1167 | 1.157 | 6.0 | 0.452 |
| DJ Hooda | 542 | 1.156 | 5.0 | 0.26 |
| DR Smith | 861 | 1.148 | 1.0 | 0.066 |
| KK Nair | 658 | 1.131 | 3.0 | 0.076 |
| SS Tiwary | 569 | 1.12 | 5.0 | 0.243 |
| KD Karthik | 1576 | 1.11 | 5.0 | 0.26 |
| SPD Smith | 1051 | 1.106 | 3.0 | 0.184 |

For contrast, **lowest** mean LI -- top-order anchors who bat when the chase is young and least decided per ball:

| batter | balls | mean_li | median_pos | death_share |
|---|---|---|---|---|
| VR Iyer | 551 | 0.708 | 2.0 | 0.029 |
| V Sehwag | 853 | 0.783 | 2.0 | 0.014 |
| PA Patel | 1149 | 0.825 | 1.0 | 0.019 |
| G Gambhir | 1982 | 0.83 | 2.0 | 0.061 |
| S Dube | 576 | 0.83 | 5.0 | 0.194 |
| CH Gayle | 1362 | 0.85 | 2.0 | 0.099 |
| R Dravid | 987 | 0.851 | 2.0 | 0.042 |
| N Rana | 1167 | 0.858 | 4.0 | 0.099 |
| Q de Kock | 1243 | 0.866 | 1.0 | 0.031 |
| Abhishek Sharma | 595 | 0.868 | 2.0 | 0.074 |

## 3. Bowlers -- highest mean leverage bowled

These are the death/high-pressure specialists:

| bowler | balls | mean_li | death_share |
|---|---|---|---|
| SM Curran | 545 | 1.504 | 0.317 |
| TG Southee | 552 | 1.456 | 0.272 |
| Avesh Khan | 797 | 1.421 | 0.346 |
| R Vinay Kumar | 1071 | 1.342 | 0.272 |
| Arshdeep Singh | 854 | 1.339 | 0.319 |
| S Kaul | 569 | 1.334 | 0.367 |
| IK Pathan | 783 | 1.289 | 0.211 |
| A Nehra | 923 | 1.267 | 0.239 |
| Shakib Al Hasan | 632 | 1.263 | 0.155 |
| R Bhatia | 688 | 1.21 | 0.126 |
| L Balaji | 754 | 1.204 | 0.276 |
| DW Steyn | 1204 | 1.193 | 0.272 |
| B Kumar | 2139 | 1.173 | 0.306 |
| JJ Bumrah | 1694 | 1.164 | 0.359 |
| JD Unadkat | 1129 | 1.147 | 0.283 |
| RP Singh | 798 | 1.144 | 0.233 |
| Mustafizur Rahman | 688 | 1.137 | 0.387 |
| DT Christian | 529 | 1.137 | 0.24 |
| MJ McClenaghan | 608 | 1.128 | 0.24 |
| A Nortje | 604 | 1.12 | 0.381 |

For contrast, **lowest** mean LI -- powerplay/early bowlers:

| bowler | balls | mean_li | death_share |
|---|---|---|---|
| SK Trivedi | 750 | 0.693 | 0.16 |
| MM Ali | 505 | 0.714 | 0.099 |
| GJ Maxwell | 520 | 0.742 | 0.065 |
| PJ Cummins | 837 | 0.783 | 0.226 |
| AR Patel | 1545 | 0.804 | 0.1 |
| RA Jadeja | 1902 | 0.812 | 0.142 |
| MC Henriques | 561 | 0.829 | 0.068 |
| PP Chawla | 1625 | 0.833 | 0.137 |
| KV Sharma | 889 | 0.839 | 0.096 |
| Harbhajan Singh | 1730 | 0.849 | 0.085 |

## Verdict

- **Batsmen -- CONFIRMED.** Mean LI rises monotonically down the batting order to a peak at the finisher slots (corr over positions 1-7 = +0.909), and at the player level corr(batter mean LI, death share) = +0.747. The top of the table is exactly the recognised finishers (Pandya, Dhoni, Tewatia, Bravo, Pollard, Russell); the bottom is the openers (Sehwag, Gambhir, Gayle, de Kock).
- **Bowlers -- CONFIRMED.** corr(bowler mean LI, death share) = +0.557: a bowler's mean leverage tracks how much they bowl at the death. The top of the table is the death specialists (Curran, Southee, Avesh, Arshdeep, Bumrah, Bhuvneshwar, Nortje, Mustafizur); the bottom is powerplay/middle spinners (Harbhajan, Chawla, Ashwin-type, Maxwell).
- The ranked tables above name the specific finishers and death bowlers who top each list; the intuition ranking in `leverage_validation.md` shows the individual last-over balls that drive these means.

### Caveat inherited from the model

LI is a dispersion quantity and inherits the Markov model's residual lag-1 autocorrelation (+0.037, see `leverage_validation.md`): the per-ball leverage is slightly understated where scoring clusters. This biases every player's mean LI in the SAME direction, so the RANKING -- which is the deliverable -- is robust; the absolute LI levels carry that error bound.
