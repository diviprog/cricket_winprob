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
> We then show the model's WP is systematically miscalibrated, and that this is not
> incidental. Localizing the error, we rule out tail-thinning and marginal
> mis-estimation (the model's per-ball outcome distribution matches empirical to a
> total variation of ≤0.02 at every required-run-rate), leaving over-dispersion
> from ball-to-ball correlation as the only possible cause. A block-bootstrap
> simulator that injects real serial correlation while holding the marginals fixed
> closes 28% of the calibration gap, saturating at partnership scale (~20 balls) —
> a constructive lower bound confirming the diagnosis. The result is a structural
> tradeoff: exact leverage requires the martingale, the martingale requires ball
> independence, and independence is precisely what miscalibrates the WP. Exactly-
> attributable leverage and well-calibrated WP cannot be obtained from the same
> object.

---

## Contribution statement (the spine)

The paper's one novel claim, stated three ways so we keep it central:

> An exactly-solvable win-probability (WP) model — estimate the one-ball outcome
> distribution, derive WP by backward induction — is a **martingale by
> construction**, which is exactly what makes leverage and WPA well-defined. But
> the ball-independence that yields that exact martingale is **precisely what
> miscalibrates the WP**. You cannot have both an exactly-attributable leverage
> and a well-calibrated WP from the same object.

We do not claim the DP-WP model itself (that is prior art — WASP). We claim: (i)
the tension above, (ii) a diagnostic method that isolates the cause of the
calibration gap, and (iii) a constructive experiment quantifying it. The
state-conditional WPA de-drifting is a secondary methodological contribution.

**NOT the contribution (say so explicitly, up front):** the DP win-probability
substrate (WASP did this), leverage index and WPA (borrowed from baseball).

---

## Section-by-section outline

### 1. Introduction
- Ball-by-ball WP in limited-overs cricket; why chases (fixed target, clean state).
- Leverage/WPA as the "who plays the big moments" tool (the applied hook: finishers
  and death bowlers).
- The surprise: the exact model that makes leverage clean is the one that
  miscalibrates WP — and this is structural, not a bug.
- Contribution bullets. Explicit "what is prior art" paragraph.

### 2. Related work  [CITATIONS TO SECURE — see checklist below]
- **WASP** (Winning and Score Predictor), Brooker & Hogan — the DP ball-by-ball
  cricket WP/score model. The direct substrate prior art; cite prominently and
  position our DP-WP as *their idea, used as a substrate*.
- **Duckworth–Lewis–Stern** — resource/state-value model for cricket targets.
- **Baseball leverage index** — Tango ("Crucial Situations"); **WPA / win
  expectancy** — Studeman. Source of LI/WPA we transfer.
- Cricket Markov / outcome-model literature (Preston & Thomas; Bailey & Clarke;
  others — TO FIND). Position our outcome-model + backward-induction framing.
- The gap in the literature: nobody states the leverage↔calibration tension, nor
  isolates the calibration error to ball correlation constructively.

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
  conditional dependence (correlation).

### 7. Constructive confirmation (the keystone)
- `src/correlation_experiment.py`, `reports/correlation_experiment.md`.
- **Part A — full marginal:** TV(model, empirical) over all 8 outcomes ≤ **0.018**,
  and ≤ **0.0006** in the big-gap bands. Completes the elimination.
- **Part B — block-bootstrap simulator, dependence knob K:**
  - `model_iid` reproduces the Markov DP WP (max |diff| ~0.002) → simulator valid.
  - Independence references agree (markov 0.0624 ~ model_iid 0.0637 ~ block_K1 0.0631).
  - Mean |gap| falls to a minimum **0.0454 at K=20** — a **28% reduction** vs
    independence — then plateaus/re-widens (U-shape).
  - **Marginal fidelity:** realized per-ball distribution drifts from K=1 by ≤ TV
    **0.0024** across all K ⇒ closure is correlation, not marginal change.
  - Closes in **both directions** (up on hard, down on easy) = over-dispersion.
  - **Partnership-scale:** saturates at K≈20 (~3 overs) — a within-partnership set-
    batsman persistence, same source as the +0.037 lag-1 autocorrelation.
  - **28% is a lower bound** the method cannot exceed (boundary breakage + spliced
    segments diverge from sim state). Full contribution needs a generative
    correlated model — which forfeits the exact martingale (→ §8).

### 8. The tradeoff (tie it together)
- Formal statement: leverage/WPA require the martingale; the martingale requires
  conditional independence given `(b,w,r)`; independence causes the over-dispersion
  of §6–7. Therefore exact leverage and WP calibration cannot be had together from
  one object.
- M2.5's choice of independence for leverage was forced, not incidental.
- Practical resolutions (state them, don't necessarily build): (a) two surfaces —
  exact-martingale WP for leverage, recalibrated WP for reporting; (b) a
  Markov-switching generative model that trades exact leverage for calibration.

### 9. Limitations & future work
- Single innings type (chases) and single league (IPL) — generalizability untested.
- Block bootstrap under-measures correlation; generative Markov-switching model is
  the way to get the full number (and it forfeits exact leverage — the tension in
  action).
- Diagnosis is in-sample on train (by design, to isolate structure) — confirm
  out-of-sample as a robustness check.
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
- **T2** Calibration gap by RRR showing the sign-flip (`tail_diagnostics.md` Diag 1).
- **T3** Full one-ball marginal TV by RRR (`correlation_experiment.md` Part A).
- **F3 (headline)** Mean |gap| vs block length K — the U-shape bottoming at K=20,
  with the marginal-fidelity TV overlaid flat near 0 (`correlation_experiment.md`).
- **F4** WPA clutch: wpa_high vs wpa_low scatter, Dhoni/ABdV annotated (`wpa.md`).
- **T4 (optional)** De-drift cross-role inversion examples (`wpa.md`).

---

## What still needs doing before submission (priority order)

1. **Secure citations** (§2). Confirm exact refs for WASP (Brooker & Hogan / Hogan),
   DLS, Tango leverage, Studeman WPA, and 2–3 cricket Markov/WP academic papers.
   *I flagged these from memory — verify each; do not cite blind.*
2. **Out-of-sample robustness** for §6 diagnosis (re-run the gap localization on the
   held-out split; confirm the sign-flip persists).
3. **Statistical CIs** on the leverage/player results (reuse the clustered bootstrap
   from `baseline.py`) — §4 currently reports point estimates.
4. **(Stretch, venue-dependent)** external benchmark (bookmaker closing odds or WASP)
   and/or a second T20 league for generalizability — only if targeting a journal.
5. Decide format: draft in Markdown, convert to LaTeX (arXiv) once structure holds.

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
| not marginals | shortfall ≤0.006/0.008 | tail_diagnostics.md |
| full marginal matches | TV ≤0.018 (≤0.0006 big-gap bands) | correlation_experiment.md |
| correlation closes gap | 28% at K=20, fidelity TV ≤0.0024 | correlation_experiment.md |
| partnership scale | saturates K≈20 | correlation_experiment.md |
