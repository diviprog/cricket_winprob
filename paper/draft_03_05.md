# Draft: Sections 3–5

Status: first full prose draft (2026-07-12). Numbers cross-checked against
`reports/era_adjustment.md`, `reports/validation_report.md`,
`reports/leverage_validation.md`, `reports/player_leverage.md`,
`reports/wpa.md`, and `reports/baseline_comparison.md`. Notation as in
`draft_06_07.md`.

---

## 3. An exactly solvable win-probability model

### 3.1 Data and the ball-level contract

We use ball-by-ball data for the Indian Premier League, 2008–2026
(Cricsheet-derived), restricted to second innings: 1,162 chases, 130,029
legal deliveries. Chases are the deliberate choice of scope. With a fixed
target, the scoreboard state $s = (b, w, r)$ — legal balls remaining, wickets
in hand, runs required — is a complete description of the task, and the
terminal conditions are exact: $r \le 0$ is a win, $w = 0$ or $b = 0$ with
$r > 1$ is a loss, and finishing level ($b = 0$, $r = 1$) is a tie, valued
$\tfrac12$.

Each delivery is reduced to a pre-ball state, one of eight outcomes
$o \in \{0, 1, 2, 3, 4, 5, 6, W\}$, and the realized match label
$y \in \{0, 1\}$. Three data-contract decisions are made once and disclosed
rather than modelled: extras are folded (a wide or no-ball adjusts runs
without consuming a legal ball), run-outs are folded into $W$, and matches
whose target changed mid-innings (rain-affected) are dropped. Monotonicity of
$b$, $w$, $r$ within innings and agreement of $y$ with the recorded winner
are enforced at ingestion.

The train/held-out split is temporal and fixed once for every experiment in
the paper: the three most recent seasons (2024–2026; 18,010 balls, 164
matches, realized win rate 0.498) are held out, and the remaining 998 innings
(112,019 balls) train every model.

### 3.2 The recursion

The single estimated object is the one-ball outcome distribution $p_o(s)$.
Everything else is arithmetic. Outcome $o = k$ runs sends $(b,w,r)$ to
$(b{-}1, w, r{-}k)$; a wicket sends it to $(b{-}1, w{-}1, r)$. Every legal
ball decrements $b$, so the transition graph is acyclic and finite, and win
probability is obtained in a single backward sweep:

$$V(s) = \sum_o p_o(s)\, V\!\big(s'_o\big),$$

with the terminal values above. No iteration, no fitting of $V$ itself, and
one immediate structural consequence: $V$ satisfies the recursion *exactly*
(machine-verified to $10^{-9}$ on random states), so under the model the WP
process is a martingale — from any state, the probability-weighted WP of the
next states equals the current WP. Section 4 builds on this property;
Sections 6–7 show its price.

### 3.3 Estimating the outcome distribution

The baseline estimator conditions $p_o$ on required run rate
($\text{RRR} = 6r/b$, six bins), wickets in hand, and innings phase
(powerplay / middle / death), with two-level shrinkage: each cell shrinks
toward its (phase, wickets) parent, which shrinks toward the global
distribution, with Dirichlet-style pseudo-counts; unseen cells fall back down
the same chain. RRR and phase are estimation features only — the dynamic-
programming state remains $(b,w,r)$.

### 3.4 The scoring-era problem, handled inside the estimation layer

T20 scoring has drifted upward sharply: training seasons average 1.344 runs
per legal ball against 1.598 in the held-out 2024–2026 seasons, the highest-
scoring in league history. A model fit uniformly on 2008–2023 systematically
under-predicts modern chases. We correct this *inside the estimation layer*,
where it cannot damage the structure: each training ball receives an
exponential recency weight with half-life $h$ seasons, and every level of the
shrinkage hierarchy is fit from the same weights. The result is still one
outcome distribution and one backward sweep, so $V$ remains an exact
martingale — unlike post-hoc recalibration of $V$, which would break the
recursion (§8).

The half-life is tuned by rolling-origin cross-validation on the three most
recent *training* seasons, each predicted one step ahead from strictly
earlier data — the same forecasting task the real held-out seasons pose, with
no contact with them. Because scoring trends monotonically, raw CV loss
decreases toward the aggressive end of the grid while the effective sample
size collapses; the selection rule, fixed in advance, takes the most
aggressive half-life on the improving curve with mean effective sample size
at least 15% of the training data. This selects $h = 0.75$ seasons. On the
held-out seasons the era-adjusted model improves Brier from 0.1637 to 0.1489,
log loss from 0.5313 to 0.4741, and ECE from 0.1627 to 0.1310 — closing 63%
of the Brier gap to the strongest simple reference (below) while keeping the
martingale. The correction has an honest ceiling: reweighting can pull the
fitted scoring level only toward the newest training season (weighted mean
1.424), not past it toward the held-out 1.598; going further would require
trend extrapolation, which we decline.

---

## 4. Leverage and clutch attribution on the exact surface

### 4.1 Definitions and validation

Because the model exposes the full next-ball distribution — not merely a win
probability — the *swing* of a state is computable exactly:

$$\text{swing}(s) = \sum_o p_o(s)\,\bigl|V(s'_o) - V(s)\bigr|,$$

the conditional mean absolute deviation of the next-ball WP, following the
absolute-value convention of baseball's leverage index. The leverage index is
swing normalized by the average over the balls that actually occurred, so the
average real delivery has $\text{LI} = 1$; the reference average is 0.0170 WP
per ball. Two checks anchor the machinery. Analytically, in a synthetic
model where one ball decides the match as a coin flip, swing must approach
$2p(1-p) = 0.5$; the implementation returns 0.5000. Empirically, the
martingale property must hold along real sequences: the mean signed one-ball
WP change over all 130,029 balls is +0.00048, near zero in every WP decile.
The one dispersion diagnostic that *fails* is residual lag-1 autocorrelation
of runs within partnerships, +0.037 where the model implies zero — recorded
here as the honest error bound on everything in this section, and the thread
Section 7 pulls.

### 4.2 Who plays the high-leverage moments

Leverage concentrates exactly where folk wisdom says: mean LI is 0.94 in the
powerplay, 0.79 in the middle overs, and 1.59 at the death, rising to 2.67 in
the final over — and the highest-leverage individual balls are all tight
last-over chases. The batting-position profile is more interesting than the
folklore: mean LI *rises* down the order from the openers (0.95) to a peak at
the finisher slots six and seven (1.31 and 1.26) and then collapses for the
genuine tail (0.24 at eleven). The tail bats at the death more than anyone
(death-ball share 0.88) yet faces the *least* leverage, because it bats in
matches already decided — high death share does not imply high leverage, and
a naive correlation over all eleven positions is negative (−0.53) purely from
this artifact. Over the batting order proper (positions 1–7) the correlation
is +0.91.

At the player level the hypothesis holds directly: the batters facing the
highest average leverage are the recognized finishers (H. Pandya 1.40,
Dhoni 1.39, Pollard 1.34, all with career medians at positions 5–6), the
lowest are top-order anchors; and mean LI correlates with death-over share at
+0.75 for batters and +0.56 for bowlers. Who *faces* the big moments is a
role, and the model measures it cleanly. Match-clustered intervals on these
leaderboards (§9; `reports/player_uncertainty.md`) make the resolution
explicit: no adjacent pair in any top-ten table is statistically
distinguishable, while the top finisher and the lowest-leverage qualified
anchor differ in mean LI by +0.69 (95% CI [+0.33, +1.08]). The rankings
separate roles, not neighbouring names.

### 4.3 Win probability added, and making it fair

WPA charges each ball's realized WP change to its participants:
$\Delta V$ to the striker, $-\Delta V$ to the bowler, with the final ball
stepping to the realized $y$. Two accounting identities hold to machine
precision and are enforced by tests: WPA is zero-sum on every ball, and a
side's WPA telescopes over an innings to $y - V(\text{first ball})$. Because
$E[\Delta V \mid s] = 0$ under the model, positive WPA is earned only by
beating the state-conditioned expectation, automatically weighted by the
moment's leverage — the formal version of "won the big moments."

The martingale is exact under the model but the model is not calibrated
(§6), and the aggregate consequence surfaces immediately: summed over all
innings, batting WPA is +62.3 wins rather than zero. This is the model's
calibration bias wearing the costume of skill — a baseline of roughly
+0.00048 WP per ball credited to every batter and debited from every bowler,
making raw cross-role comparisons invalid. We remove it *state-
conditionally*: each ball's WPA is de-drifted by the mean WPA of all balls
beginning in the same WP bin (width 0.02), because the drift is far from
uniform — near zero in decided states, up to 0.0019 per ball in contested
ones — so a flat correction would mis-price players by where they operate.
De-drifting preserves the per-ball zero-sum, drives the grand total to zero
by construction, and redistributes standings by state mix: the largest
individual effects are V. Kohli falling from +3.83 to +2.25 (a high-volume
batter who faced more high-drift, contested balls than average) and
R. Ashwin rising from +4.21 to +5.55.

The de-drifted clutch split is the section's payoff. Splitting each career
at LI = 2 — balls worth at least twice the average — separates *performing*
from *being present*: M.S. Dhoni is essentially flat on ordinary balls
(−0.001 per ball) and strongly positive on his 346 high-leverage balls
(+0.017 per ball), the statistical signature the word "finisher" gestures
at; by contrast a genuinely excellent overall batter can invert the split
(S.A. Yadav: +0.003 ordinary, −0.010 high-leverage). Match-clustered
bootstrap intervals (§9) put both named splits outside zero: Dhoni's
high-minus-low difference is +0.018 per ball (95% CI [+0.004, +0.032]),
Yadav's −0.013 ([−0.031, −0.001]). One caution stands regardless: per-ball
differences below about ±0.005 sit inside the dependence error bound of
§4.1, which these intervals condition on rather than capture.

---

## 5. How good is the exact WP? Validation and fitted baselines

### 5.1 Protocol

All models are scored on the identical held-out split by Brier score, log
loss, and expected calibration error, with reliability curves. Because all
~120 balls of an innings share one outcome label, per-ball resampling
understates uncertainty by roughly $\sqrt{120}$; every model comparison
therefore reports a *paired, match-clustered* bootstrap on the log-loss
difference — whole matches resampled, per-ball differences paired — and
"significant" means the 95% interval excludes zero.

### 5.2 The exact model against simple references

| model | Brier | log loss | ECE |
|---|---|---|---|
| constant base rate | 0.2503 | 0.6937 | 0.0169 |
| base Markov (state-free $p_o$) | 0.2040 | 0.8011 | 0.2235 |
| RRR Markov | 0.1637 | 0.5313 | 0.1627 |
| RRR Markov + era ($h{=}0.75$) | 0.1489 | 0.4741 | 0.1310 |
| logistic on RRR alone | 0.1404 | 0.4379 | 0.0773 |
| logistic, engineered features | 0.1350 | 0.4157 | 0.0700 |
| XGBoost on $(b,w,r)$ | 0.1414 | 0.4374 | 0.0771 |
| XGBoost, full features | 0.1490 | 0.4607 | 0.0847 |

*Table T1. Held-out comparison (18,010 balls, 164 matches). Markov models
and logistics fit on the full training split; XGBoost fits surrender the 2023
season to early stopping.*

The exact model beats the trivial references soundly, and the era adjustment
is a significant gain (−0.057 nats, CI [−0.079, −0.035]). But a *one-feature*
logistic on required run rate alone beats the full backward-induction model.
That inversion is the paper's first symptom: the structured model is not
short of information — it conditions on strictly more than the logistic sees
— it is mis-mapping the information it has.

### 5.3 Three results from the fitted baselines

**The gap is structure, not state.** On the identical inputs $(b,w,r)$, an
unconstrained gradient-boosted model beats the exact Markov WP by 0.094 nats
(CI [0.026, 0.174]), with ECE 0.077 against 0.163. Same information,
different mapping. This figure is conservative: the boosted models lose their
most recent training season to early stopping.

**Hidden state is not the missing ingredient.** Adding the striker's
balls-faced count — the observable proxy for a "set" batsman, and the
enrichment the sport's folklore recommends — to the boosted model moves
held-out log loss by +0.0004 nats (CI [−0.0032, +0.0035]). A null from the
most flexible model we can fit is strong evidence: whatever the exact model
is missing, it is not carried by this variable.

**Under distribution shift, rigidity is robustness.** The best model on the
held-out seasons is the plain logistic, beating every boosted variant. The
richest feature set is the *worst* boosted model precisely because it can
see the target total: it fits 2008–2023 scoring conditions closely (best
training loss, 0.410) and transfers worst (0.461), and season-wise early
stopping cannot catch this because the validation season sits on the training
side of the era shift. We flag this as a practical warning for sports models
generally: under a large covariate shift, capacity converts into era
overfitting.

Together, Table T1 and the ablation frame the question the rest of the paper
answers. The exact model's deficit is not missing information and not a
missing latent variable; it is a property of the mapping from state to
probability. Section 6 locates that property.
