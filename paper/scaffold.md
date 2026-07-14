# Paper scaffold

Working title: **The Calibration–Leverage Tradeoff in Exactly-Solvable Win-Probability Models**
(alt: *Leverage or Calibration: a Structural Tension in Ball-by-Ball Win Probability*)

Target: arXiv preprint first (the common prerequisite), then a sports-analytics
venue if it reads strongly. This file is the outline + the claim→evidence map, not
prose. Every number below is already produced and lives in `reports/`.

---

## Draft abstract (v0 — for framing; tighten later)

> We study ball-by-ball win probability (WP) for second-innings run chases in
> Twenty20 cricket, built as an exactly-solvable Markov model: we estimate a
> single object — the per-ball outcome distribution over {0..6, wicket} — and
> derive WP for every game state by backward induction over the acyclic
> (balls, wickets, runs-required) chase graph. This construction makes WP an exact
> martingale, which in turn makes *leverage* (how much a ball can swing WP) and
> *win probability added* (WPA) well-defined and exactly attributable; we use them
> to confirm that finishers and death bowlers occupy the highest-leverage moments.
> We then show the model's WP is systematically miscalibrated, and that this is
> not incidental. Localizing the error, we rule out tail-thinning and marginal
> mis-estimation (the model's per-ball outcome distribution matches empirical to
> a total variation of ≤0.02 at every required-run-rate), leaving unmodelled
> dependence given the state as the only possible cause — and we identify it: a
> permutation-null decomposition shows short-range sequential run-scoring
> persistence (~3–5 balls; innings-level heterogeneity contributes only ~18%;
> wickets, if anything, anti-cluster). A block-bootstrap simulator that injects
> the real dependence while holding the marginals fixed closes 26% of the
> calibration gap (replicated over six seeds), saturating at block lengths of
> ~20 balls, consistent with the measured correlation range — a constructive
> lower bound confirming the diagnosis. The result is a structural tradeoff
> relative to the (balls, wickets, runs) state description: exact leverage
> requires the martingale, the martingale requires conditional ball independence
> on that state, and that independence is precisely what miscalibrates the WP.
> Exactly-attributable leverage and well-calibrated WP cannot be obtained from
> the same object over this state.

---

## Contribution statement (the spine)

The paper's one novel claim — **scoped to the state description**, which is what
makes it defensible:

> An exactly-solvable win-probability (WP) model — estimate the one-ball outcome
> distribution, derive WP by backward induction over the `(b,w,r)` chase state —
> is a **martingale by construction**, which is exactly what makes leverage and
> WPA well-defined. But the conditional ball-independence **on that state** which
> yields the exact martingale is precisely what miscalibrates the WP. **Relative
> to the `(b,w,r)` filtration**, exactly-attributable leverage and well-calibrated
> WP cannot be obtained from the same object.

Scoping caveats to state explicitly (a referee will otherwise supply them):
- This is an empirical demonstration for this model class, not a theorem.
- Enriching the state with the latent driving the dependence could in principle
  restore both properties over the augmented state. M4 rules out the obvious
  enrichment (`striker_balls`, a decisive null), not the class; the decomposition
  shows the residual is short-range sequential, which no *pre-ball-observable*
  scalar obviously captures — but say "one proxy ruled out", not "impossible".
- A directly-fitted calibrated WP (≈ E[y|b,w,r]) is NOT a martingale along real
  paths when the true process has hidden dependence, so it cannot host exact
  leverage either — the two objects genuinely bifurcate.

We do not claim the DP-WP model itself (prior art — WASP / the Clarke DP line).
We claim: (i) the scoped tension above, (ii) the diagnostic that localizes the
calibration gap and eliminates marginals, (iii) the **dependence decomposition**
(permutation-null lag profiles: sequential scoring persistence of ~3–5 balls;
heterogeneity only ~18%; wickets ANTI-cluster), (iv) the constructive
block-bootstrap closure (26%, 6-seed replicated, marginals held fixed), and (v)
state-conditional WPA de-drifting as a secondary method.

**NOT the contribution (say so explicitly, up front):** the DP win-probability
substrate (WASP did this; Clarke 1988 is the DP-in-cricket origin), leverage
index and WPA (borrowed from baseball: Tango; Studeman).

---

## Section-by-section outline

### 1. Introduction
- Ball-by-ball WP in limited-overs cricket; why chases (fixed target, clean state).
- Leverage/WPA as the "who plays the big moments" tool (the applied hook: finishers
  and death bowlers).
- The surprise: the exact model that makes leverage clean is the one that
  miscalibrates WP — and this is structural, not a bug.
- Contribution bullets. Explicit "what is prior art" paragraph.

### 2. Related work  [ALL CITATIONS VERIFIED — full entries in `paper/references.md`]
- **WASP** — Brooker & Hogan (Working Paper 44/2011, U. Canterbury; broadcast Sky
  Sport NZ 2012). The DP ball-by-ball cricket WP/score model; the direct substrate
  prior art. Cite prominently; position our DP-WP as *their construction, used as
  a substrate*.
- **The DP-in-cricket origin line** — Clarke (1988, JORS: DP optimal scoring
  rates); Clarke & Norman (1999, JORS); Preston & Thomas (2000, JRSS-D: DP batting
  strategy, chase vs target-setting asymmetry).
- **Duckworth–Lewis(–Stern)** — Duckworth & Lewis (1998, JORS); Stern (2016,
  JORS). Resource/state valuation; Stern's modern-scoring-rate update is direct
  precedent for our era-adjustment finding.
- **The Swartz line (closest neighbour, key contrast)** — Davis, Perera & Swartz
  (2015, ANZJS): ball-by-ball T20 simulator whose outcome probabilities condition
  on batsman and bowler via hierarchical empirical Bayes. They take the
  player-latent route we deliberately exclude to keep the state small and WP
  exactly solvable; our M4 ablation + dependence decomposition quantify what that
  exclusion costs.
- **Baseball LI/WPA** — Tango ("Crucial Situations", THT 2006; *The Book* 2007);
  Studeman ("The One About Win Probability", THT). Source of the concepts we
  transfer.
- **Cricket pressure index** — Bhattacharjee & Lemmer (2016, IJSSC 11(5),
  683–692). Overlaps leverage in intent; distinguish: heuristic composite vs the
  exact MAD of a derived martingale.
- The gap in the literature: nobody states the leverage↔calibration tension, nor
  isolates the calibration error to short-range sequential dependence
  constructively.

### 3. The exact Markov WP (substrate, §2 prior art made precise)
- State `(b,w,r)`; the one estimated object `p_o(state)` over {0..6,W}.
- `transition`; acyclic DAG (b strictly decrements) ⇒ one backward-induction sweep.
- Terminal conditions (win r≤0, loss w=0, tie r=1 at b=0).
- Martingale property `E[ΔWP|state]=0` derived, not fitted. → §5,§7 lean on it.
- RRR-conditioned outcome estimation with two-level shrinkage; era adjustment as a
  recency-weighted estimation layer (half-life 0.75 seasons, rolling-origin CV)
  that PRESERVES the martingale.
- Data: IPL second innings, 130,029 balls; train / held-out (last 3 seasons) split.
  Source: `src/ingest.py`, `src/wp_markov.py`, `src/outcome_model.py`,
  `src/era_adjust.py`; `reports/era_adjustment.md`.

### 4. Leverage and WPA (applied layer)
- Leverage: `swing(s)=Σ p_o|WP(s'_o)−WP(s)|` (MAD, baseball convention); LI=swing/avg.
  Numbers: avg_swing **0.0170**; closed-form `swing((1,1,1))=0.5000`.
- Leverage by phase: powerplay **0.935**, middle **0.788**, death **1.592**;
  over 19 = **2.672**. Highest-LI balls = tight last overs; lowest = blowouts.
- **Applied result (the hook): finisher/death-bowler hypothesis confirmed.**
  `corr(mean LI, death share)` = **+0.75** batters, **+0.56** bowlers.
- WPA: `WP(after)−WP(before)`, +batter/−bowler; zero-sum; telescopes to
  `y−wp(first ball)` (**+62.3** total). `E[WPA|state]=0` ⇒ WPA = beating
  state-conditioned expectation, leverage-weighted.
- Clutch signal `wpa_high` vs `wpa_low`: MS Dhoni **+0.0176** vs **−0.0008**.
- **State-conditional de-drifting** (secondary contribution): the +62.3 is model
  calibration bias, not skill; subtract mean WPA in the same WP-bin (not a flat
  constant — drift ranges 0.00007→0.00185). Kohli **+3.83→+2.25**, Ashwin
  **+4.21→+5.55**. Sources: `src/leverage.py`, `src/player_leverage.py`,
  `src/wpa.py`; `reports/leverage_validation.md`, `player_leverage.md`, `wpa.md`.

### 5. WP validation and the calibration gap
- Metrics: Brier, log loss, ECE, reliability; held-out split; match-clustered
  bootstrap (rows within a match share a label → per-row CIs ~√120 too narrow).
- RRR Markov beats base-rate; trails a one-feature RRR-logistic (under-prediction).
- **M4 fitted baselines** (`src/baseline.py`, `reports/baseline_comparison.md`):
  - Plain logistic best on held-out (**log loss 0.4157**), beating every XGBoost —
    under the era covariate shift (train target mean 166.9 vs held-out 194.0),
    rigidity is robustness. (Contra the usual "XGBoost wins" expectation.)
  - On identical `(b,w,r)`, unconstrained XGBoost beats the RRR Markov WP by
    **0.094 nats** (CI [0.026, 0.174]): the gap is structure, not missing state.
  - **M5 licensing ablation (negative result):** adding `striker_balls` moves
    held-out log loss by **+0.00041 nats, CI [−0.00322, +0.00345]** (straddles 0).
    Hidden state not licensed — a decisive null from the most flexible model.

### 6. Diagnosing the gap (the method)
- `src/tail_diagnostics.py`, `reports/tail_diagnostics.md`. On TRAIN (era shift out
  of frame) to isolate structure error.
- **Not tail-thinning:** gap = `mean(Markov WP) − empirical` SIGN-FLIPS: +0.014
  easy → −0.112 at RRR 10–12 → ~0 hopeless; corr with RRR order −0.08 (flat).
  It is **over-dispersion** (WP too extreme both ways).
- **Not marginals:** boundary shortfall ≤0.006, wicket shortfall ≤0.008 (noise).
- Refines `validation_report.md`: n-weighted mean WP **0.482** < empirical
  **0.522** (under-predicts on average) but only because live balls cluster on the
  hard side of the sign-flip — the defect is over-dispersion, not uniform
  under-confidence.
- Elimination: correct marginals + independence ⇒ WP = E[y|state]; it doesn't ⇒
  conditional dependence given the state.
- **OOS robustness:** the same gap table on the held-out split (era shift in
  frame, so the level moves; the check is the non-monotone sign-flip SHAPE).
  Numbers from `tail_diagnostics.md` §"Out-of-sample robustness".

### 7. What the dependence IS (decomposition) + constructive confirmation (keystone)
- **7a. Decomposition** (`src/dependence_decomposition.py`,
  `reports/dependence_decomposition.md`): within-innings permutation null —
  shuffling within an innings PRESERVES the innings-latent contribution to pair
  statistics while destroying ordering, so `observed > null band` = dependence
  beyond innings heterogeneity, and the null centre estimates the heterogeneity
  share. Discriminator validated on synthetics (iid not flagged; pure innings
  random effect absorbed; AR(1) flagged).
  - Runs residuals: lag-1 **+0.0436** vs null centre **+0.0079** → heterogeneity
    share **18%**; excess decays +0.036/+0.021/+0.008 at lags 1/3/5, gone by
    ~10–20 → **short-range sequential scoring persistence (~3–5 balls)**.
    Survives innings AND partnership demeaning → ball-adjacent, not a latent.
  - Cross-check: null centre (+0.0079) ≈ innings ICC (+0.0081).
  - **Wickets ANTI-cluster** at lags 1–3 (below null). "Wicket clusters" is dead;
    the mechanism is scoring bursts and their mirror-image droughts — a two-sided
    variance effect matching the sign-flip.
- **7b. Constructive closure** (`src/correlation_experiment.py`,
  `reports/correlation_experiment.md`), replicated over 6 seeds:
  - **Part A — full marginal:** TV(model, empirical) over all 8 outcomes ≤
    **0.018**, ≤ **0.0006** in the big-gap bands. Completes the elimination.
  - `model_iid` reproduces the Markov DP WP; independence references agree at
    every seed (markov 0.0624 ~ model_iid 0.0618 ~ block_K1 0.0613).
  - Mean |gap| minimum **0.0451 at K=20** — a **26% reduction** vs independence;
    K=20 < K=40 in **6/6 seeds** and separated from every seed's K=1 → the
    saturation is resolved, and its scale is consistent with the ~3–5-ball
    correlation range (cumulative block variance saturates once K ≫ range).
  - **Marginal fidelity:** block modes drift from K=1 by ≤ TV **0.0033** across
    all seeds ⇒ the closure is dependence, not marginal change.
  - Closes in **both directions** (up on hard, down on easy) = over-dispersion.
  - **Scattered-arm negative result (report honestly):** the designed
    heterogeneity-only control degenerated — per-ball re-matching within a donor
    innings makes it a nearest-neighbour replay of empirical outcomes (closed
    MORE than whole real trajectories; broke its own marginal control, TV 0.018).
    Kept as a methodological caution; the mechanism question is settled by 7a.
  - **26% is a lower bound** the method cannot exceed (boundary breakage +
    spliced segments diverge from sim state). Full contribution needs a
    generative dependent model — which forfeits the exact martingale (→ §8).

### 8. The tradeoff (tie it together)
- Formal statement, **scoped**: leverage/WPA require the martingale; the
  martingale requires conditional independence given `(b,w,r)`; that independence
  causes the over-dispersion of §6–7. Therefore exact leverage and WP calibration
  cannot be had together from one object **over the `(b,w,r)` state**.
- State-enrichment escape hatch, addressed: a richer state could restore both in
  principle; M4's `striker_balls` null closes the obvious route; the residual is
  short-range sequential, not obviously capturable by any pre-ball-observable
  scalar. A directly-fitted calibrated WP is not a martingale along real paths,
  so it cannot host exact leverage either.
- M2.5's choice of independence for leverage was forced, not incidental.
- Practical resolutions (state them, don't necessarily build): (a) two surfaces —
  exact-martingale WP for leverage, recalibrated WP for reporting; (b) a
  Markov-switching generative model that trades exact leverage for calibration.

### 9. Limitations & future work
- Single innings type (chases) and single league (IPL) — generalizability untested.
- Block bootstrap under-measures the dependence; a generative Markov-switching
  model is the way to get the full number (and it forfeits exact leverage — the
  tension in action).
- The heterogeneity-only simulator control is an open design problem (our
  scattered arm degenerated); the permutation-null decomposition carries the
  mechanism claim alone.
- Run-outs folded into `W` (spec 01); no separate fielding attribution.

### 10. Conclusion
- The exact model's elegance and its calibration flaw are the same property.
- A clean negative-result-adjacent contribution: rigor over a headline model.

---

## Figures & tables (all data already in `reports/`)

- **T1** Model comparison: base/RRR Markov, logistic, XGBoost — Brier/logloss/ECE
  (from `baseline_comparison.md`). Include the striker_balls ablation CI.
- **F1** Reliability curves, held-out (`reliability.png`, `baseline_reliability.png`).
- **F2** Leverage by over (bar), highlighting the death spike to 2.67
  (`leverage_validation.md`).
- **T2** Calibration gap by RRR showing the sign-flip (`tail_diagnostics.md` Diag 1),
  with the OOS column alongside.
- **T3** Full one-ball marginal TV by RRR (`correlation_experiment.md` Part A).
- **F3** Lag profile of runs residuals with the permutation-null band — observed
  decaying curve exiting the flat band at lags 1–5
  (`dependence_decomposition.md`). The mechanism figure.
- **F4 (headline)** Mean |gap| vs block length K with cross-seed min–max bands,
  bottoming at K=20; marginal-fidelity TV overlaid flat near 0
  (`correlation_experiment.md`).
- **F5** WPA clutch: wpa_high vs wpa_low scatter, Dhoni/ABdV annotated (`wpa.md`).
- **T4 (optional)** De-drift cross-role inversion examples (`wpa.md`).

---

## What still needs doing before submission (priority order)

1. ~~Secure citations~~ **DONE** — all verified, full entries + relevance notes in
   `paper/references.md` (incl. the Swartz line and the Clarke DP origin).
2. ~~Out-of-sample robustness~~ **DONE** — OOS section in `tail_diagnostics.md`.
3. ~~Error bars on the K-curve~~ **DONE** — 6-seed replication; saturation gates
   pass (6/6 seeds).
4. ~~Mechanism disentangling~~ **DONE** — `dependence_decomposition.md` (sequential,
   ~18% heterogeneity, wickets anti-cluster).
5. **Statistical CIs** on the leverage/player results (reuse the clustered
   bootstrap from `baseline.py`) — §4 currently reports point estimates.
6. **(Stretch, venue-dependent)** external benchmark (bookmaker closing odds or
   WASP) and/or a second T20 league for generalizability — only if targeting a
   journal.
7. Decide format: draft in Markdown, convert to LaTeX (arXiv) once structure holds.
   **In progress:** §3–5 (`paper/draft_03_05.md`), §6–7 (`paper/draft_06_07.md`),
   §8–10 (`paper/draft_08_10.md`) drafted, all numbers cross-checked against the
   source reports; figures F3/F4 rendered (`paper/make_figures.py`,
   `paper/figures/`). Remaining prose: §1 intro, §2 related work, abstract polish,
   then assembly + LaTeX.

## Claim → evidence map (for grounding while writing)

| Claim | Number | Source report |
|---|---|---|
| leverage machinery correct | swing((1,1,1))=0.5000 | leverage_validation.md |
| death is the leveraged phase | over 19 LI 2.672 | leverage_validation.md |
| finisher hypothesis | corr(LI, death share) +0.75 | player_leverage.md |
| hidden-state error bound | lag-1 autocorr +0.0368 | leverage_validation.md |
| WPA accounting exact | telescope +62.3, zero-sum 0 | wpa.md |
| structure gap | XGB(b,w,r) beats RRR Markov 0.094 nats | baseline_comparison.md |
| M5 not licensed | striker +0.00041, CI straddles 0 | baseline_comparison.md |
| logistic beats XGBoost | log loss 0.4157 | baseline_comparison.md |
| gap is over-dispersion | sign-flip, worst −0.112 @ RRR 10–12 | tail_diagnostics.md |
| shape persists OOS | sign flip True; +0.026 easy / −0.267 @ 8–10 | tail_diagnostics.md |
| not marginals | shortfall ≤0.006/0.008 | tail_diagnostics.md |
| full marginal matches | TV ≤0.018 (≤0.0006 big-gap bands) | correlation_experiment.md |
| dependence is sequential | lag-1 +0.0436 vs null +0.0079 (18% het) | dependence_decomposition.md |
| ~3–5 ball range | excess +0.036/+0.021/+0.008 @ lags 1/3/5 | dependence_decomposition.md |
| wickets anti-cluster | lag 1–3 below null band | dependence_decomposition.md |
| dependence closes gap | 26% at K=20, 6/6 seeds, block fidelity ≤0.0033 | correlation_experiment.md |
| saturation resolved | K=20 < K=40 in 6/6 seeds | correlation_experiment.md |
| scattered arm degenerate | gap 0.019 < whole-trajectory 0.046; TV 0.018 | correlation_experiment.md |
| adjacent ranks not separable | 0/36 adjacent-rank diff CIs exclude 0 | player_uncertainty.md |
| role separation decisive | Pandya vs Iyer mean-LI +0.692 [+0.332, +1.078] | player_uncertainty.md |
| Dhoni clutch split significant | +0.0179 [+0.0042, +0.0317] (346 high-LI balls) | player_uncertainty.md |
| Yadav inversion significant | −0.0132 [−0.0305, −0.0010] | player_uncertainty.md |
