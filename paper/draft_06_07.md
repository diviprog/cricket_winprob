# Draft: Sections 6–7

Status: first full prose draft (2026-07-12). Numbers cross-checked against
`reports/tail_diagnostics.md`, `reports/dependence_decomposition.md`, and
`reports/correlation_experiment.md`. Figure/table labels follow the scaffold
(T2, T3, F3, F4). Notation assumed from earlier sections: a chase state is
$s = (b, w, r)$ (legal balls remaining, wickets in hand, runs required);
$p_o(s)$ is the fitted one-ball outcome distribution over
$\{0,1,2,3,4,5,6,W\}$; $V(s)$ is the win probability obtained from $p_o$ by
backward induction; $y \in \{0,1\}$ is the realized match outcome.

---

## 6. Localizing the calibration gap

Section 5 established that the Markov win probability is miscalibrated in a
way that additional state cannot repair: on the identical state $(b,w,r)$, an
unconstrained gradient-boosted model beats the backward-induction WP by 0.094
nats of held-out log loss (95% match-clustered CI [0.026, 0.174]), while
adding a set-batsman proxy to that same unconstrained model moves log loss by
+0.0004 nats — a decisive null. The information in the state is not the
problem. Something about the *mapping* from state to probability is. This
section localizes the error; Section 7 identifies its cause and reproduces it
constructively.

### 6.1 Design

Two decisions matter for what follows. First, all diagnostics in this section
are computed on the training split, scored by the model fit to that same
split. Section 5 showed that the held-out seasons sit on the far side of a
large scoring-era shift; evaluating train-on-train removes that shift from the
frame, so whatever miscalibration remains is structural rather than temporal.
(Section 6.4 returns to the held-out data to confirm the structure is not an
in-sample artifact.) Second, the WP surface under examination is the
era-adjusted RRR Markov model of Section 3 — the surface on which the leverage
and WPA results of Section 4 are actually computed, so any defect found here
propagates to those results directly.

We slice the state space by required run rate (RRR $= 6r/b$), the natural
difficulty axis of a chase, using bins finer than the six the outcome model
conditions on, and compare the model's mean WP against the empirical win rate
within each slice:

$$G(\text{slice}) \;=\; \overline{V(s)}_{\,\text{slice}} \;-\;
\overline{y}_{\,\text{slice}}.$$

A positive $G$ means the model overrates the chasing side in that band; a
negative $G$ means it underrates them.

### 6.2 The gap flips sign: over-dispersion, not tail-thinning

Entering this analysis, our working hypothesis was *tail-thinning*: a model
that treats balls as independent draws should underweight the scoring
explosions that rescue difficult chases, and therefore rate hard chases as
harder than they are — a gap that is negative and grows monotonically with
RRR. The data reject this cleanly (Table T2).

| RRR slice | balls | mean model WP | empirical win rate | gap $G$ |
|---|---|---|---|---|
| 0–6   | 15,210 | 0.977 | 0.963 | **+0.014** |
| 6–8   | 24,190 | 0.785 | 0.753 | **+0.032** |
| 8–10  | 33,801 | 0.447 | 0.518 | −0.071 |
| 10–12 | 18,194 | 0.183 | 0.294 | **−0.112** |
| 12–14 | 8,034  | 0.080 | 0.165 | −0.085 |
| 14–16 | 4,021  | 0.047 | 0.100 | −0.054 |
| 16–18 | 2,263  | 0.028 | 0.062 | −0.033 |
| 18–22 | 2,133  | 0.011 | 0.022 | −0.011 |
| 22–40 | 2,652  | 0.006 | 0.014 | −0.008 |

*Table T2. Calibration gap by required run rate, training split (112,019
balls). The gap changes sign: positive on easy chases, deepest in the live
middle, fading toward zero for near-hopeless chases.*

The gap is not monotone in RRR (rank correlation with slice order: −0.08). It
is positive on the easiest chases — the model calls them *safer* than they are
— and strongly negative in the contested middle, worst at RRR 10–12, where
chases the model rates at 18% are won 29% of the time. This is the signature
of **over-dispersion**: the model's probabilities are pushed too far toward
both 0 and 1. Outcomes are systematically less determined than the model
believes, in both directions.

This refines, and partly corrects, the summary a coarser analysis would give.
Averaged over all live balls, the model does under-predict (weighted mean WP
0.482 against an empirical 0.522), which reads as simple under-confidence. The
sign-flip shows why that reading is wrong: live balls merely cluster on the
hard side of the crossover. The defect is symmetric; the aggregate bias is a
composition effect.

Splitting Table T2 by wickets in hand adds one detail that matters later. The
positive band — over-confidence on easy chases — lives almost entirely in the
7–10-wickets-in-hand group (+0.020 and +0.033 in the two easiest slices). At
this point in the analysis a natural story suggests itself: the model, drawing
each ball independently, cannot represent batting collapses, so it misses the
occasional easy chase that falls apart from six down. Section 7 tests that
story directly, and it turns out to be wrong in an instructive way.

### 6.3 The marginals are not the problem

One family of explanations must be eliminated before dependence can be
blamed: perhaps the one-ball distributions $p_o(s)$ are simply misestimated
where it matters. The outcome model bins RRR coarsely, with a single 15+
catch-all shrunk toward calmer parent cells, so extreme chases are exactly
where estimation error should concentrate — and if the model understates
$p_4 + p_6$ at high RRR, it would mechanically understate the win probability
of hard chases.

We therefore compare the model's own per-ball probabilities against empirical
frequencies along the same fine RRR axis. For the scoring tail, the shortfall
$\;\overline{p_4 + p_6}^{\,\text{emp}} - \overline{p_4 + p_6}^{\,\text{model}}$
never exceeds 0.006 in absolute value in any slice; for the wicket
probability the corresponding figure is 0.008. Both are at the noise level,
and neither shows a trend in RRR. (Section 7.3 completes this check for the
full eight-outcome distribution; the conclusion is unchanged.) The model
knows, essentially exactly, how often each outcome occurs from every kind of
situation. Finer bins or lighter shrinkage would buy nothing.

### 6.4 The elimination, and its out-of-sample check

The two facts above force a conclusion by construction. If the one-ball
marginals are correct *and* consecutive balls are conditionally independent
given the state, backward induction reproduces $E[y \mid b,w,r]$ exactly —
that is what the recursion computes. The marginals are correct (§6.3), yet
$V(s)$ misses $E[y \mid s]$ with a systematic, sign-flipping shape (§6.2).
The only assumption left standing is independence. The residual is
**conditional dependence between balls, given the state**.

The same table computed on the held-out seasons confirms the shape is not an
artifact of in-sample fitting. There the era shift is back in frame, so the
level drops as expected — the deep middle trough reaches −0.267 at RRR 8–10 —
but the structure survives: still positive on the easiest chases (+0.026),
still sign-flipping, still fading in the hopeless tail. Whatever produces the
over-dispersion is a property of the game, not of the fit.

What Section 6 cannot do is say what the dependence *is*. That distinction —
between two mechanisms that are observationally identical to every diagnostic
used so far — turns out to decide both the interpretation and the available
remedies, and it is where we turn next.

---

## 7. The dependence: what it is, and that it closes the gap

### 7.1 Two mechanisms, one signature

"Conditional dependence given the state" admits two physically different
readings.

*Serial correlation.* Outcomes influence nearby outcomes: a batter who has
just scored keeps scoring; boundaries and dot-balls arrive in runs. The
process has momentum that the state $(b,w,r)$ does not carry.

*Shared latent heterogeneity.* Each innings (or partnership) has a hidden
level — pitch, ground dimensions, quality of the bowling attack, who is
batting — that all its balls share. Conditional on the innings, balls could be
perfectly independent; pooling across innings still produces dependence
relative to a model that sees only $(b,w,r)$.

Every diagnostic so far is blind to the difference. Both mechanisms leave the
one-ball marginals intact; both over-disperse innings trajectories; both
produce positive residual autocorrelation; and both are carried by the
consecutive-ball blocks used in Section 7.3, since twenty consecutive balls
share their innings' latent as well as their local ordering. The distinction
nonetheless matters twice over. It decides what the leverage and WPA results
of Section 4 mean (a set-batsman momentum effect is chargeable to players in a
way that pitch conditions are not), and it decides the remedy (a latent
innings effect suggests state augmentation; short-range momentum does not
correspond to any pre-ball observable). It also matters for our own earlier
claims: the residual lag-1 autocorrelation of +0.037 reported in Section 4 as
a "hidden-state error bound" was computed after removing state means but not
innings effects, so it, too, conflates the two mechanisms.

### 7.2 A permutation decomposition

The two mechanisms can be separated with an exact resampling argument.
Working with residuals $x_t = \text{runs}_t - \overline{\text{runs}}_{(\text{phase},
w)}$ (state means removed, as in Section 4), consider shuffling the residuals
*within each innings* — values permuted, positions fixed. The shuffle
destroys all ordering, and with it any serial structure. But it preserves
each innings' composition: any two balls of the same innings still share that
innings' latent level, so the shared-latent contribution to any pair statistic
survives the shuffle untouched. Repeating the shuffle builds a null
distribution with two useful properties:

1. the null is *centred at the heterogeneity component*, not at zero — the
   permuted autocorrelation is exactly what innings-level sharing alone would
   produce; and
2. an observed statistic *above the null band* demonstrates dependence beyond
   innings-level heterogeneity, with the mechanical bias of any demeaning
   step absorbed into the null automatically.

We validated the discriminator on synthetic data before applying it: i.i.d.
sequences are not flagged (null centred at zero, observed inside the band); a
pure innings-level random effect is fully absorbed (intraclass correlation
recovered, lag profile flat at that level, observed inside the band at every
lag); an AR(1) process is flagged with the correct decaying profile.

Applied to the 112,019 training balls (200 permutations), the decomposition
gives an unambiguous answer (Figure F3):

| lag | observed | null mean | null 97.5% | above band? |
|---|---|---|---|---|
| 1  | +0.0436 | +0.0079 | +0.0140 | **yes** |
| 2  | +0.0376 | +0.0081 | +0.0139 | **yes** |
| 3  | +0.0287 | +0.0079 | +0.0136 | **yes** |
| 5  | +0.0155 | +0.0076 | +0.0131 | **yes** |
| 10 | +0.0127 | +0.0081 | +0.0148 | no |
| 20 | +0.0058 | +0.0076 | +0.0137 | no |
| 30 | +0.0074 | +0.0081 | +0.0139 | no |

*Figure F3 (data). Within-innings autocorrelation of state-demeaned run
residuals, against the within-innings permutation null.*

Three observations, in increasing order of consequence.

**The dependence is mostly not heterogeneity.** The null centre at lag 1 —
the innings-composition component — is +0.008, against an observed +0.044:
shared innings conditions account for about 18% of the signal. The null
centre agrees with the innings-level intraclass correlation computed
independently (0.0079 vs 0.0081), an internal consistency check on the whole
construction.

**The remainder is short-range and sequential.** The excess over the null
decays from +0.036 at lag 1 to +0.021 at lag 3 to +0.008 at lag 5, and is
indistinguishable from the null by lag 10–20: a correlation range of roughly
three to five balls. It survives the demeaning ladder — removing innings
means leaves lag-1 at +0.027 against a null upper bound of −0.003; removing
partnership means and restricting to within-partnership pairs leaves the
observed value above its (strongly negative, bias-absorbing) null band as
well. Dependence that survives partnership demeaning cannot be an artifact of
*who* is batting or *how set* they are in the aggregate; it is ball-adjacent
scoring momentum.

**Wickets do not cluster — they anti-cluster.** Applying the identical
machinery to wicket-indicator residuals reverses the sign: at lags 1–3 the
observed autocorrelation sits *below* the permutation band (lag 1: −0.009
against a null of [−0.006, +0.005]). Immediately after a wicket — beyond
what the state adjustment already accounts for — another wicket in the next
few balls is *less* likely than independence predicts. Batting sides
consolidate. This kills the collapse story suggested at the end of §6.2: the
over-confidence on easy chases is not the model missing wicket cascades. It
does not need to be. Positive scoring persistence alone produces the
two-sided gap, because persistence is symmetric: bursts rescue hard chases
(the model was too pessimistic there), and the mirror-image droughts sink
easy ones (too optimistic there). One mechanism, both signs.

### 7.3 Constructive confirmation

The elimination argument of §6.4 and the decomposition of §7.2 identify the
cause but do not yet demonstrate it. The demonstration should run forwards:
inject the measured dependence into an otherwise identical model and watch
the calibration gap close. We build a forward innings simulator with a
dependence dial.

**Completing the marginal check.** The simulator resamples real balls, so all
comparisons require that the model's marginals match the empirical ones in
full, not only in the scoring and wicket tails checked in §6.3. Total
variation distance between the model's eight-outcome distribution and the
empirical one is at most 0.018 across every RRR slice — and at most 0.0006 in
the four most heavily populated slices (RRR 0–12, 91,000 of the 112,019
balls), which carry both signs of the flip and the worst single slice (Table
T3). The elimination is complete at the level of the entire one-ball
distribution.

**Modes.** From each evaluated start state we run Monte-Carlo continuations
to termination under three regimes. *model-iid* draws every ball
independently from the fitted $p_o(s)$; since this reproduces exactly the
process backward induction integrates, its mean terminal value must equal the
DP win probability, and it does — validating the simulator's mechanics.
*block-K* resamples runs of $K$ consecutive real balls, matched to the
current situation cell at each block start: the per-ball marginals stay flat
by construction, but real local dependence up to range $K$ rides along.
$K=1$ is an independent draw from the empirical cell marginal and must — and
does — agree with model-iid. Raising $K$ raises only one thing: the amount of
real sequential structure the simulation honours.

**Protocol.** 400 continuations per state, 220 evaluated states per RRR
slice, and the entire experiment replicated over six seeds, each seed
redrawing both the evaluated states and the Monte-Carlo paths. The
calibration criterion is the mean absolute slice gap, $\overline{|G|}$.
Because the resampler uses flat empirical frequencies, this experiment uses
the plain (non-era-weighted) fit throughout, so that all modes share the same
marginals; its independence references reproduce the same gap structure as
§6.2. As a control, each mode's *realized* outcome distribution — what its
simulations actually consumed — is compared against the $K=1$ mode's: block
modes stay within total variation 0.0033 of independence across all seeds,
so any gap movement is attributable to dependence alone.

| mode | mean $\overline{\lvert G\rvert}$ | range over 6 seeds |
|---|---|---|
| exact DP (reference) | 0.0624 | — |
| model-iid | 0.0618 | 0.0587–0.0667 |
| block, K=1 | 0.0613 | 0.0582–0.0663 |
| block, K=3 | 0.0565 | 0.0532–0.0611 |
| block, K=8 | 0.0504 | 0.0477–0.0549 |
| **block, K=20** | **0.0451** | 0.0432–0.0490 |
| block, K=40 | 0.0502 | 0.0481–0.0545 |
| block, K=80 | 0.0480 | 0.0455–0.0529 |
| block, K=120 | 0.0456 | 0.0433–0.0498 |

*Figure F4 (data). Mean absolute calibration gap by dependence range $K$,
replicated over six seeds.*

Injecting real dependence with marginals held fixed closes the gap
monotonically out to $K=20$: from 0.0613 to 0.0451, a **26% reduction**
relative to independence. The minimum's location is stable — $K{=}20$ beats
$K{=}40$ in six seeds of six, and the $K{=}20$ mean lies below every seed's
$K{=}1$ value — and its scale is the one §7.2 predicts. A block's excess
cumulative variance grows with $K$ only until $K$ exceeds the correlation
range several times over; a three-to-five-ball range saturates near
$K \approx 20$. The two measurements were made by unrelated methods, and they
agree. The closure is also two-sided, as over-dispersion requires: block
modes raise the simulated WP in the hard bands and lower it in the easy ones.

Twenty-six percent is a floor, not an estimate of the whole effect. The block
bootstrap breaks dependence at every block boundary and captures none beyond
$K$; splicing longer segments does not help, because a long donor segment's
internal state diverges from the simulation's own state accounting, which is
why the curve goes flat rather than continuing downward past $K=20$.
Measuring the full contribution requires a generative dependent process — a
Markov-switching outcome model, say — which is no longer a diagnostic but a
different model, with a cost that Section 8 takes up.

**A control that failed, honestly reported.** We also built what was intended
as a heterogeneity-only arm: draw cell-matched balls from *random* positions
of a single donor innings, carrying the donor's latent conditions without its
local ordering. Its results are reported because they are instructive, not
because they are usable. The arm produced a mean absolute gap of 0.019 —
better calibrated than replaying whole real trajectories, which no injection
of innings-level heterogeneity could achieve — while drifting from the shared
marginals by total variation 0.018, five times any block mode. The design is
degenerate: re-matching every ball to the simulation's evolving cell within
one donor turns the arm into a nearest-neighbour replay of empirical
outcomes, an estimator of $E[y \mid s]$ in disguise rather than an injection
of dependence. We flag it as a caution for constructive calibration
experiments of this kind; the mechanism split rests on the permutation
decomposition of §7.2, which has no analogous failure mode.

### 7.4 Where this leaves the model

Assembled, Sections 6 and 7 say the following. The exact Markov WP is
miscalibrated with a specific, out-of-sample-stable shape: too extreme in
both directions. Its one-ball marginals are essentially perfect, so the error
is the independence assumption itself. The dependence that independence
discards is identified — short-range sequential scoring persistence of about
three to five balls, four-fifths of it beyond anything innings conditions
explain, with wicket clustering explicitly ruled out — and injecting exactly
that dependence, marginals fixed, closes a quarter of the calibration gap at
precisely the range the decomposition measured, replicated across seeds.

The independence assumption is not incidental to the model. It is what makes
the state graph acyclic, the backward induction exact, and the win
probability a martingale — the property Section 4's leverage and WPA
constructions stand on. The next section states the resulting tradeoff
precisely.
