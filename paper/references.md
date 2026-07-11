# Verified references

Each entry was verified by web search on 2026-07-11 (publisher / journal pages
located; no blind citing). One-line relevance notes say what the paper is FOR in
our related-work positioning. BibTeX keys suggested for the LaTeX pass.

## Cricket win probability / DP substrate (prior art we build on)

**[brooker2011] Brooker, S. & Hogan, S. (2011).** *A Method for Inferring Batting
Conditions in ODI Cricket from Historical Data.* Working Paper No. 44/2011,
Department of Economics and Finance, University of Canterbury.
— The research programme behind **WASP** (Winning and Score Predictor), the
DP-based ball-by-ball cricket WP/score model, first broadcast on Sky Sport NZ in
November 2012 (HRV Cup). This is the direct prior art for our exact-DP substrate:
cite prominently, position our DP-WP as *their construction, used as a substrate*.
Verified: [UC repository](https://ir.canterbury.ac.nz/items/53c441f6-8376-4a1c-98fd-1003fce64ca2),
[working paper PDF](https://www.cmi.ac.in/~rlk/Cricket_Analytics/AuxiliaryMaterial/12637455_Hogan.pdf),
[WASP background](https://en.wikipedia.org/wiki/WASP_(cricket_calculation_tool)).

**[clarke1988] Clarke, S.R. (1988).** *Dynamic Programming in One-Day Cricket —
Optimal Scoring Rates.* Journal of the Operational Research Society, 39(4),
331–337.
— The oldest DP-in-limited-overs-cricket paper; establishes backward induction
over (balls, wickets) state as a decades-old idea. Verified:
[T&F](https://www.tandfonline.com/doi/abs/10.1057/jors.1988.60).

**[clarke1999] Clarke, S.R. & Norman, J.M. (1999).** *To run or not? Some dynamic
programming models in cricket.* Journal of the Operational Research Society,
50(5), 536–545.
— Further DP models of in-innings decisions; related-work depth for the DP line.
Verified: [T&F](https://www.tandfonline.com/doi/abs/10.1057/palgrave.jors.2600705).

**[duckworth1998] Duckworth, F.C. & Lewis, A.J. (1998).** *A fair method for
resetting the target in interrupted one-day cricket matches.* Journal of the
Operational Research Society, 49(3), 220–227.
— The canonical state-resource valuation in cricket (balls + wickets → resources);
ancestor of every state-value model including ours.

**[stern2016] Stern, S.E. (2016).** *The Duckworth-Lewis-Stern method: extending
the Duckworth-Lewis methodology to deal with modern scoring rates.* Journal of
the Operational Research Society, 67(12), 1469–1480.
— The modern-scoring-rate update to DL: directly relevant precedent for our
era-adjustment finding (scoring-era shift breaks fitted cricket models).
Verified: [Springer](https://link.springer.com/article/10.1057/jors.2016.30).

**[preston2000] Preston, I. & Thomas, J. (2000).** *Batting Strategy in Limited
Overs Cricket.* Journal of the Royal Statistical Society: Series D (The
Statistician), 49(1), 95–106.
— DP representation of batting strategy; shows optimal chase strategy differs
from target-setting — background for why the chase is the clean modelling unit.
Verified: [Wiley](https://rss.onlinelibrary.wiley.com/doi/10.1111/1467-9884.00223).

## The Swartz line (ball-by-ball simulators with player heterogeneity)

**[davis2015] Davis, J., Perera, H. & Swartz, T.B. (2015).** *A Simulator for
Twenty20 Cricket.* Australian & New Zealand Journal of Statistics, 57(1), 55–71.
— Ball-by-ball T20 simulator whose outcome probabilities depend on **batsman,
bowler**, overs and wickets, fit by hierarchical empirical Bayes. This is the
closest methodological neighbour and the key positioning contrast: they take the
player-latent route (heterogeneity in the state) that we deliberately exclude to
keep the state small and the WP exactly solvable — and our M4 ablation +
dependence decomposition quantify what that exclusion costs (0.094 nats of
calibration; a short-range sequential residual). Verified:
[Wiley](https://onlinelibrary.wiley.com/doi/full/10.1111/anzs.12109),
[preprint](https://www.sfu.ca/~tswartz/papers/t20sim.pdf).

## Leverage / clutch attribution (concepts we transfer from baseball)

**[tango2006] Tango, T. (2006).** *Crucial Situations* (Parts 1–3). The Hardball
Times, May–June 2006.
— The **leverage index**: expected |WP swing| of a state, normalized to the
average situation = 1. Our `swing`/LI is this definition transferred to the
cricket chase DAG. Verified:
[Part 1](https://tht.fangraphs.com/crucial-situations/),
[Part 2](https://tht.fangraphs.com/crucial-situations-part-2/),
[Part 3](https://tht.fangraphs.com/crucial-situations-part-three/).

**[tango2007] Tango, T., Lichtman, M. & Dolphin, A. (2007).** *The Book: Playing
the Percentages in Baseball.* Potomac Books.
— Book-form treatment of leverage and WP-based analysis; standard citation for
the LI concept.

**[studeman2004] Studeman, D. (2004).** *The One About Win Probability.* The
Hardball Times.
— The canonical WPA exposition (win probability added, +offence/−defence,
zero-sum per event). Verified:
[THT](https://tht.fangraphs.com/the-one-about-win-probability/).

## Cricket pressure / leverage-adjacent (must-cite overlap)

**[bhattacharjee2016] Bhattacharjee, D. & Lemmer, H.H. (2016).** *Quantifying the
pressure on the teams batting or bowling in the second innings of limited overs
cricket matches.* International Journal of Sports Science & Coaching, 11(5),
683–692.
— The cricket "pressure index" line: a second-innings situational-pressure
measure. Overlaps our leverage index in intent; distinguish: pressure index is a
heuristic composite, ours is the exact MAD of a derived martingale. Verified:
[ResearchGate record](https://www.researchgate.net/publication/308876293)
(SAGE, October 2016, vol 11, pp. 683–692).

## Data

**IPL ball-by-ball data** via Kaggle (Cricsheet-derived). Cite the concrete
dataset + Cricsheet (cricsheet.org, ODbL) with access date in the final draft.

## Deliberately NOT cited as novelty sources

Our DP-WP construction (WASP/Clarke line), LI (Tango), and WPA (Studeman) are
all prior art. The paper's novel claims are only: the calibration–leverage
tension (scoped to the (b,w,r) filtration), the diagnostic that localizes the
calibration gap and eliminates marginals, the dependence decomposition
(sequential scoring persistence, wickets anti-cluster, heterogeneity ~18%), the
constructive block-bootstrap closure, and state-conditional WPA de-drifting.
