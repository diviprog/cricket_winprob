# cricket-leverage

A win-probability and leverage-index engine for **IPL second-innings run
chases**, built as an exactly-solvable Markov model and validated against
realized match outcomes.

## Idea

Rather than fitting win probability (WP) directly, we estimate a single object —
the per-ball outcome distribution over `{0,1,2,3,4,5,6,W}` conditioned on the
match state — and derive WP *for free* by exact backward induction. The chase
state graph is acyclic (every legal ball uses up one of the 120 balls), so WP is
solvable in a single backward sweep with no iteration. Leverage is then the
per-ball dispersion of the next-ball WP.

WP is validated on the **mean** (calibration, Brier, log loss against the
realized 0/1 winner) and leverage on the **spread** (martingale and
autocorrelation diagnostics) — separately, because the two failure modes are
different.

## Layout

```
src/
  ingest.py            # dataset → per-ball (b, w, r, outcome, y) contract
  state.py             # exact, pure transition core
  wp_markov.py         # terminal conditions + backward induction
  outcome_model.py     # the estimated statistical layer (base / RRR + recency weights)
  era_adjust.py        # recency-weight half-life tuning (rolling-origin CV)
  validation.py        # calibration, Brier, log loss, reliability curves
  leverage.py          # swing, leverage index, martingale/autocorr diagnostics
  player_leverage.py   # who FACES the highest-leverage balls (finisher hypothesis)
  wpa.py               # WPA clutch attribution + state-conditional de-drifting
  baseline.py          # M4 fitted logistic/XGBoost baselines + M5 licensing ablation
  tail_diagnostics.py  # localizes the WP calibration gap (over-dispersion, not marginals)
  dependence_decomposition.py  # serial vs shared-latent: permutation-null lag profiles + ICC
  correlation_experiment.py    # constructive block-bootstrap closure of the gap (6-seed)
tests/                 # DP consistency, ingestion, leverage, baseline, dependence checks
data/                  # raw + processed (gitignored)
reports/               # generated .md findings (validation, leverage, wpa, baseline, ...)
paper/                 # paper scaffold, verified references, claim→evidence map
```

## Setup

```bash
uv sync
uv run python -m src.ingest        # build the processed ball table
uv run pytest                      # core + ingestion checks
uv run python -m src.validation    # WP metrics vs baselines
```

Requires a Kaggle API token at `~/.kaggle/kaggle.json` for the data download.

## Primary results

**Win probability.** The exact RRR Markov WP beats the constant base-rate on
Brier and log loss and is well-ordered, but trails a one-feature (RRR) logistic:
it systematically *under-predicts*, from two spec-anticipated effects — a
memoryless right-tail thinning and a scoring-era shift (the held-out 2024–26
seasons are the highest-scoring in IPL history). The recency-weighted outcome
model (half-life 0.75 seasons, tuned by rolling-origin CV) closes a significant
part of the era gap while keeping WP an exact martingale.

**Leverage.** `avg_swing = 0.0170` WP/ball; the closed-form check
`swing((1,1,1)) = 0.5000` holds exactly. Leverage is concentrated at the death
as hypothesized — mean leverage index by phase is powerplay 0.94, middle 0.79,
death 1.59, peaking at **2.67 in over 19**. The highest-LI balls are recognizably
tight last-over chases; the lowest are blowouts. The **finisher / death-bowler
hypothesis is confirmed**: the batsmen and bowlers who face the most leverage are
finishers and death specialists (`corr(mean LI, death share) = +0.75` for
batters). Residual within-partnership lag-1 autocorrelation is **+0.037** — the
honest error bound on every leverage number, since the model assumes independent
balls.

**Clutch attribution (WPA).** Per-ball WPA is zero-sum (batter `+`, bowler `−`)
and telescopes exactly per innings to `y − wp(first ball)`; batting sides finish
**+62.3 wins** above the start-of-chase expectation. That surplus is the model's
calibration bias, not skill, so career totals are **de-drifted
state-conditionally** (subtract the mean WPA of balls at a similar WP) to make
batter-vs-bowler totals comparable. The honest clutch signal is `wpa_high` vs
`wpa_low`: players like AB de Villiers and MS Dhoni are strongly positive in
high-leverage moments while flat-to-negative in ordinary ones.

**Fitted baselines + the hidden-state question (M4).** Against the same held-out
split, a plain logistic (log loss **0.4157**) beats every XGBoost variant — under
the large era shift, rigidity is robustness. On the *identical* `(b,w,r)`, an
unconstrained XGBoost still beats the RRR Markov WP by **0.094 nats**: the gap is
structure, not missing state. Crucially, adding a "set batsman" proxy
(`striker_balls`) moves held-out log loss by **+0.00041 nats (95% CI straddling
zero)** under a match-clustered bootstrap — so **hidden state (M5) is not
licensed**, a deliberate negative result.

**Why the WP is imperfect, exactly (calibration diagnosis).** The 0.094-nat gap
is **over-dispersion**: WP pushed too far toward 0/1 (over-confident on easy
chases, under-confident in the live middle, worst `−0.11` at RRR 10–12; the
sign-flip shape persists out of sample). It is **not** tail-thinning (the gap is
non-monotone in RRR) and **not** marginal mis-estimation (the model's full
per-ball outcome distribution matches empirical to total variation ≤ 0.02 at
every RRR). With the marginals ruled out, the residual is unmodelled
**dependence given the state** — and a permutation-null decomposition identifies
it: short-range **sequential run-scoring persistence** (~3–5 ball range, lag-1
+0.036 above the null), surviving innings and partnership demeaning. Innings
heterogeneity contributes only ~18%, and wickets **anti-cluster** at short lags —
so the mechanism is scoring bursts and their mirror-image droughts, not wicket
clusters. A block-bootstrap simulator that injects the real dependence with
marginals held fixed closes **26% of the gap** (6-seed replication; saturating at
block length ≈ 20, consistent with the measured correlation range). This exposes
a structural tension, **relative to the `(b,w,r)` state**: exact leverage
requires the martingale, the martingale requires conditional ball independence
on that state — the very assumption that causes the calibration gap. A richer
state could in principle restore both, but the obvious enrichment is a proven
null (M4). Closing the gap on this state forfeits exact leverage, so the
limitation is documented rather than traded away.

## Status

v1 covers second-innings chases only. Milestones **M0** (ingestion), **M1**
(exact core), **M2** (RRR outcome model + WP validation), **M2.5** (era
adjustment), **M3** (leverage + player attribution), **WPA** (clutch attribution
+ de-drifting), and **M4** (fitted baselines + M5 ablation) are complete. **M5**
(hidden state) was gated on M4 motivating it; the ablation returned a decisive
no, so it is not built. The calibration gap is diagnosed and banked as a finding.
First-innings WP and player/venue embeddings remain out of v1 scope.
