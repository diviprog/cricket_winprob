# Draft: Sections 8–10

Status: first full prose draft (2026-07-12). Follows `draft_06_07.md`; same
notation. §8 is the thesis section; §9 limitations; §10 conclusion.

---

## 8. The tradeoff

The results of Sections 6–7 are not a defect report. They are one half of a
tradeoff whose other half is the machinery of Section 4, and the two halves
are the same assumption viewed from opposite sides.

### 8.1 The statement

Consider the three properties one might want from a single win-probability
object over the chase state $s = (b, w, r)$:

1. **Exactness.** $V$ satisfies $V(s) = \sum_o p_o(s)\, V(s'_o)$ identically,
   so that $V$ is a martingale by construction and leverage (the conditional
   MAD of $\Delta V$) and WPA (with $E[\text{WPA} \mid s] = 0$) are exact
   bookkeeping rather than estimates.
2. **Calibration.** $V(s) = E[y \mid s]$: among situations the model calls
   $p$, the chasing side wins fraction $p$.
3. **The state is $(b,w,r)$.** No latent variables, no player identities, no
   history beyond the scoreboard.

Sections 6–7 show empirically that these three properties are jointly
unsatisfiable for T20 chases. The construction that delivers (1) assumes
conditional independence of balls given $s$; the true process carries
short-range sequential scoring persistence beyond $s$ (§7.2); therefore the
constructed $V$ misses $E[y \mid s]$ with the over-dispersed shape of §6.2,
and injecting the missing dependence — the only remaining degree of freedom,
with marginals verified correct — moves $V$ toward calibration (§7.3). Under
(3), exactness and calibration exclude each other. This is an empirical
demonstration for this model class and this sport, not a theorem; but the
elimination in §6.4 gives it more force than a fitted-model comparison would,
because it identifies *which assumption* fails and confirms the mechanism
constructively.

It is worth stating why the tradeoff does not dissolve from the other
direction. One could fit $V^*(s) \approx E[y \mid s]$ directly — Section 5's
gradient-boosted baseline is exactly this — and obtain calibration by
construction. But $V^*$ is a projection of a dependent process onto a
non-sufficient state, and along real ball sequences it drifts: knowing the
current state *and the outcome that produced it* shifts the expectation of
$V^*$ at the next ball, because the last outcome predicts near-future scoring
(§7.2) in a way the state does not capture. After a boundary, the true win
probability sits above $V^*(s')$; after a dot-ball stretch, below.
Consequently $E[\Delta V^* \mid \text{path}] \neq 0$, WPA computed on $V^*$ is
not zero-mean given the situation — a batter would earn "clutch" credit partly
for the surface's blind spot — and leverage loses its interpretation as the
dispersion of a fair game. A calibrated surface over this state cannot host
exact attribution either. The two desiderata do not merely trade against each
other within one model; they bifurcate into different objects.

### 8.2 The escape hatches, addressed

*Enrich the state.* Property (3) is the scope of the claim, so the natural
escape is a richer state carrying the latent that drives the dependence. Two
results constrain this route. The obvious enrichment — how long the striker
has been at the crease, the direct observable proxy for "set" — adds nothing:
+0.0004 nats in the most flexible model we could fit (§5), a null with a
match-clustered CI tight around zero. And the decomposition says why finding a
better enrichment is hard: the residual dependence is ball-adjacent with a
three-to-five-ball range, surviving innings and partnership demeaning, so it
does not correspond to any slowly-varying observable a scoreboard state could
carry. It behaves like a hidden regime, not a hidden covariate. We cannot
rule out that some pre-ball-observable enrichment restores both properties —
the claim is scoped, not absolute — but the one route the folklore suggests
is closed, measurably.

*Model the dependence generatively.* A Markov-switching outcome process (a
hidden hot/cold scoring regime with three-to-five-ball persistence) would
recover calibration, and §7.3's 26% is a floor on what it would gain. The
cost is structural. The state graph over $(b,w,r,\text{regime})$ still admits
backward induction in principle, but the regime is unobservable: win
probability becomes an expectation over a filtered posterior, leverage
becomes model-dependent, and WPA attribution requires inferring each ball's
regime before crediting the player — attribution stops being bookkeeping and
becomes inference, with all of its instability. This is the same fork taken
by the simulator line of Davis, Perera and Swartz, who condition outcomes on
player identities: richer, better calibrated in expectation, and no longer
exact. The choice between the branches is real, and nothing in the data makes
it for you.

*Recalibrate post hoc.* A monotone map from $V$ to observed frequencies fixes
the reliability diagram and destroys the recursion: the mapped surface no
longer satisfies $V = \sum_o p_o V'$, so the martingale property — and with
it every Section 4 guarantee — is lost. This was the concrete choice faced in
Section 3: the recalibrated surface was measurably better calibrated and was
rejected for the leverage layer for exactly this reason. In light of §6–7
that rejection was not conservatism; it was forced.

### 8.3 Practical resolution

For applied work the resolution we adopt, and recommend, is two surfaces with
declared roles. *Report* win probability from a calibrated surface (here,
even a one-feature logistic transfers across eras better than the exact
model). *Attribute* — leverage, WPA, clutch splits — on the exact martingale
surface, publishing its known error bound alongside: residual lag-1
autocorrelation +0.037, calibration gap up to 0.11 in the contested band,
both stable out of sample. The attribution rankings are robust to this bias
because it is shared across players and largely removed by the
state-conditional de-drifting of Section 4; the probabilities are not, which
is why they should come from the other surface. What one must not do is take
WPA totals computed on the exact surface and read them as calibrated
probability claims — the +62.3-win aggregate drift of Section 4 is precisely
that error, measured.

---

## 9. Limitations

**Scope of data.** One league (IPL, 2008–2026) and one innings type. Chases
were chosen deliberately — a fixed target makes $(b,w,r)$ a complete
scoreboard description and the terminal conditions exact — but this means the
tradeoff is demonstrated, not shown universal. First innings, other T20
leagues, and the 50-over format are open. We expect the mechanism (scoring
persistence beyond a scoreboard state) to transfer; its magnitude may not.

**The 26% is a floor with an unmeasured ceiling.** The block bootstrap breaks
dependence at every block boundary and captures nothing beyond its window;
its curve saturating at $K \approx 20$ reflects the method's reach as much as
the effect's size. The full contribution of sequential dependence to the
calibration gap awaits a generative dependent model, which is future work —
and, per §8, a different object rather than a repair.

**No clean simulator-side heterogeneity control.** Our designed control
degenerated into a nearest-neighbour replay (§7.3), so the
serial-vs-heterogeneity split rests on the permutation decomposition alone.
That test is exact and validated on synthetics, but a second, independent
constructive probe would be better, and designing one that neither breaks
marginals nor degenerates appears genuinely hard.

**Attribution simplifications.** Run-outs are folded into the wicket outcome
and credited to the bowler; extras are folded into the batter's ball count
(a wide faced increments deliveries faced without consuming a legal ball).
Both are disclosed data-contract decisions rather than modelling claims, and
neither touches the calibration analysis, but WPA at the individual level
inherits them.

**Player-level uncertainty.** Leaderboards carry match-clustered bootstrap
intervals (2,000 replicates per player, whole innings resampled; per-ball
resampling would assume away the dependence §7 measured;
`reports/player_uncertainty.md`). They sharpen §4 in both directions. No
adjacent pair in any top-ten table is statistically separable (0 of 36
adjacent-rank difference CIs exclude zero), so the rankings resolve roles
rather than neighbouring names, and the role contrast is decisive (mean LI
+0.69, CI [+0.33, +1.08], top finisher against the lowest-leverage qualified
anchor). The headline clutch splits survive: Dhoni's is +0.018 per ball (CI
[+0.004, +0.032]), the Yadav inversion −0.013 (CI [−0.031, −0.001]); five of
seventeen qualifying batters have splits excluding zero. The intervals
condition on the fitted surfaces; the model-level dependence bound of §4.1
sits on top of them, and per-ball differences below about ±0.005 remain
inside it.

---

## 10. Conclusion

We built the exactly-solvable win-probability model for T20 chases and used
its martingale structure to make leverage and clutch attribution exact,
confirming the folk hypothesis about finishers and death bowlers with clean
accounting. We then turned the same rigor on the model itself and found its
probabilities systematically over-dispersed; localized the error; eliminated
the marginals; decomposed the residual dependence into a short-range
sequential scoring persistence that innings conditions mostly do not explain
— with wickets, against expectation, anti-clustering — and closed a quarter
of the calibration gap by constructively injecting exactly the dependence we
had measured, at exactly the range we had measured it.

The finding is that these two threads are one. The independence assumption
that makes the win probability exact — acyclic state graph, one-sweep
backward induction, martingale attribution — is the same assumption whose
failure miscalibrates it. Within a scoreboard state, you may have exact
attribution or calibrated probability, and you must choose; we make the
choice explicit and give each surface its role. The elegance and the flaw are
the same property. We suspect this tension is not specific to cricket: any
sport whose win-probability models assume event-level independence over a
compact state, and whose real event streams carry momentum at any timescale,
faces the same fork — usually unstated. Stating it, measuring both sides of
it, and pricing the exit routes is what this paper contributes.
