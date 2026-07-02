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
  ingest.py         # dataset → per-ball (b, w, r, outcome, y) contract
  state.py          # exact, pure transition core
  wp_markov.py      # terminal conditions + backward induction
  outcome_model.py  # the estimated statistical layer (base / RRR)
  validation.py     # calibration, Brier, log loss, reliability curves
tests/              # DP consistency + ingestion sanity checks
data/               # raw + processed (gitignored)
```

## Setup

```bash
uv sync
uv run python -m src.ingest        # build the processed ball table
uv run pytest                      # core + ingestion checks
uv run python -m src.validation    # WP metrics vs baselines
```

Requires a Kaggle API token at `~/.kaggle/kaggle.json` for the data download.

## Status

v1 covers second-innings chases only. Milestones M0 (ingestion), M1 (exact
core), and M2 (RRR outcome model + WP validation) are the current scope.
Leverage index, fitted baselines, and hidden-state refinements follow.
