# Draft: Sections 1–2

Status: first full prose draft (2026-07-12). Citations keyed to
`paper/references.md` (all verified). This completes the body: §1–2 here,
§3–5, §6–7, §8–10 in their draft files.

---

## 1. Introduction

Every ball of a Twenty20 run chase moves a probability. Broadcasters show it,
bettors trade it, and analysts increasingly use its movements to say which
players matter: how much a delivery *could* swing the win probability is the
leverage of the moment, and how much a player's deliveries *did* swing it is
his win probability added. Cricket has a distinguished line of models for the
probability itself, from the Duckworth–Lewis resource valuations
[duckworth1998; stern2016] through dynamic-programming treatments of
limited-overs strategy [clarke1988; preston2000] to WASP, the ball-by-ball
DP forecaster used in broadcast [brooker2011]. Baseball has a mature
tradition for the attribution layer [studeman2004; tango2006]. This paper
connects the two and reports something that, to our knowledge, has not been
stated: the property that makes the attribution layer *exact* and the
property that makes the probability layer *correct* are, for this class of
model, the same assumption with opposite signs.

The model class is the natural one. For a second-innings chase the scoreboard
state — balls remaining, wickets in hand, runs required — describes the task
completely, and every legal delivery consumes a ball, so the state graph is
acyclic. Estimate one object, the distribution of the next ball's outcome
given the state, and win probability for *every* state follows from a single
backward-induction sweep, exactly. Nothing about winning is fitted. The
construction has two consequences that flow in opposite directions, and the
paper is organized around the collision.

The first consequence is a gift. Constructed this way, win probability is a
martingale: from any state, the expected next-ball change is exactly zero.
Leverage becomes a well-defined conditional dispersion rather than a
heuristic, and win probability added becomes a fair game — a player
accumulates credit only by beating the state's expectation, weighted
automatically by the size of the moment. On eighteen seasons of Indian
Premier League chases this machinery confirms the sport's folk hypothesis
with clean accounting: leverage concentrates at the death (2.7× the average
ball in the final over), the batters who face the most of it are the
recognized finishers, and the clutch decomposition separates players who
perform in high-leverage moments from players who are merely present in them.

The second consequence is a debt, and quantifying it is the paper's main
work. The same model's probabilities are systematically *over-dispersed* —
too confident at both extremes. Easy chases are called safer than they are;
hard-but-live chases are called more doomed (a band the model rates at 18%
wins 29% of the time), a shape that persists out of sample. We localize the
error and eliminate the candidate causes in turn. It is not missing
information: on the identical state, an unconstrained learner beats the exact
model by 0.094 nats, and the obvious latent-state proxy — how long the
striker has been at the crease — is a precise null. It is not estimation
error: the model's one-ball outcome distribution matches the empirical one to
total variation ≤ 0.02 everywhere. What remains, necessarily, is the
independence assumption itself, and we identify what it discards. A
permutation decomposition that provably absorbs innings-level heterogeneity
shows the dependence is *sequential*: run-scoring persists over a three-to-
five-ball range, four-fifths of it beyond anything shared innings conditions
explain — while wickets, against the folklore of collapse, anti-cluster.
A block-bootstrap simulator that injects exactly this measured dependence,
with marginals held fixed and the experiment replicated across seeds, closes
26% of the calibration gap, saturating at exactly the range the decomposition
measured. The mechanism is identified, quantified, and reproduced.

The collision is then unavoidable, and we state it as the paper's thesis.
Exact leverage requires the martingale; the martingale requires conditional
independence of deliveries given the scoreboard state; and that independence
is precisely what miscalibrates the probabilities. Within this state
description, exactly attributable leverage and calibrated win probability
cannot come from the same object — and the escape routes each surrender
something: enriching the state (the measured null, and a dependence with no
obvious pre-ball observable), fitting the probability directly (calibrated,
but it drifts along real paths and cannot host fair attribution), modelling
the dependence generatively (calibrated, but attribution becomes inference
over a hidden regime), or recalibrating post hoc (breaks the recursion
outright). We conclude with the practical resolution we adopt: two surfaces
with declared roles, attribution on the exact one with its error bound
published, probability reporting on a calibrated one.

Contributions, explicitly scoped:

1. an exact chase win-probability engine with leverage and WPA layers, and
   the finisher/death-bowler hypothesis confirmed with exact accounting
   (§3–4), including a state-conditional de-drifting correction that makes
   cross-role WPA totals comparable (§4.3);
2. a diagnosis of the exact model's miscalibration as over-dispersion, with
   marginal error eliminated and the residual pinned on conditional
   dependence (§5–6);
3. a permutation decomposition separating sequential dependence from shared-
   latent heterogeneity — finding short-range scoring persistence and wicket
   anti-clustering — and a replicated constructive experiment closing a
   quarter of the gap by injecting the measured dependence (§7);
4. the calibration–leverage tradeoff, stated relative to the scoreboard
   state, with each exit route priced (§8).

We are explicit about what is *not* claimed as new: the DP win-probability
construction is WASP's and ultimately Clarke's; leverage and WPA are
baseball's. The novelty is the diagnosis, the decomposition, the constructive
demonstration, and the tradeoff they jointly establish.

## 2. Related work

**Win probability and state valuation in cricket.** Dynamic programming
enters limited-overs cricket with Clarke [clarke1988], who solved optimal
scoring rates over a (balls, wickets) state, with later in-innings decision
models in the same tradition [clarke1999; preston2000]. The
Duckworth–Lewis method [duckworth1998] institutionalized the valuation of
(overs, wickets) resources, and Stern's update [stern2016] recalibrated it to
modern scoring rates — the same era-shift problem our recency-weighted
estimation layer addresses (§3.4), there solved for target resetting, here
for probability calibration. WASP [brooker2011] is the direct ancestor of
our substrate: a ball-by-ball DP model estimating expected runs and win
probability from historical transition frequencies, broadcast in New Zealand
from 2012. Our Section 3 is this construction, deliberately kept minimal; the
paper's contribution begins where these models stop, in asking what the
construction's independence assumption does to its own outputs.

**Ball-by-ball simulators with player heterogeneity.** The closest
methodological neighbours are the Swartz-line simulators, particularly Davis,
Perera and Swartz [davis2015], whose T20 simulator draws each ball from
outcome distributions conditioned on batsman and bowler identities via
hierarchical empirical Bayes. That line and ours make opposite choices on the
same fork: they enrich the state with player latents and obtain richer
expectations by simulation, surrendering exactness; we hold the state to the
scoreboard and obtain exact, martingale-consistent attribution, surrendering
— as this paper quantifies — calibration. Our M4 ablation and dependence
decomposition can be read as measuring the cost of our side of that fork, and
§8.2 as pricing theirs.

**Leverage and clutch attribution.** The leverage index originates with
Tango [tango2006; tango2007], win probability added with the sabermetric
tradition around Studeman [studeman2004]; both are defined against empirical
win-expectancy tables. Transferring them to a *derived*, exactly-solvable WP
changes their epistemic status — leverage becomes the conditional MAD of a
constructed martingale rather than an estimate against a fitted surface —
which is exactly why the calibration of that construction matters enough to
occupy the second half of this paper. In cricket, situational pressure has
been quantified by the pressure-index line of Bhattacharjee and Lemmer
[bhattacharjee2016]; those indices are heuristic composites designed for
descriptive comparison, where our leverage is a model-derived quantity with
an exactness guarantee (and a measured bias).

**Calibration and dependence.** That unmodelled positive dependence
over-disperses aggregate outcomes is classical in statistics; our
contribution is not the phenomenon but its identification and constructive
quantification inside a deployed model class, and the observation that the
independence being violated is load-bearing for the model's attribution
layer, not incidental. The permutation device of §7.2 — within-group
shuffling as an exact null that preserves group composition while destroying
order — is standard machinery applied to a decomposition question we have not
seen posed in this setting: *which kind* of dependence a scoreboard-state
model is missing.

To our knowledge, no prior work states the tension between exact
(martingale-based) attribution and probability calibration in sports win
models, measures both sides of it on one dataset, or demonstrates its
mechanism constructively. That is the gap this paper fills.
