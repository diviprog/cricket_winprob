---
name: regen-reports
description: Regenerate the project's derived reports in dependency order, with runtime warnings. Use when reports/ must be rebuilt after a change to src/, when the user says "regenerate reports", or when a report is stale relative to its generating module.
---

Regenerate reports for cricket_winprob. Reports are generated artifacts — never
hand-edit them; fix the generating module and re-run.

## Rules

- Interpreter: `.venv/bin/python -m src.<module>` (never `uv run`).
- **One job at a time**, long ones in the background (`run_in_background`) — this
  machine swaps under concurrent load.
- Everything is seeded: re-running an unchanged module reproduces identical
  numbers. Only re-run what is downstream of the actual change.

## Dependency order and runtimes

| step | command (module) | writes | needs | runtime |
|---|---|---|---|---|
| 1 | `src.ingest` | `data/processed/second_innings_balls.parquet` | Kaggle raw | ~1 min |
| 2 | `src.leverage` | appends `wp`,`li` columns to parquet; `reports/leverage_validation.md` | 1 | ~5 min |
| 3 | `src.wpa` | `reports/wpa.md` | 2 (needs `wp`,`li`) | ~1 min |
| 4 | `src.player_leverage` | `reports/player_leverage.md` | 2 | ~1 min |
| 5 | `src.validation` | `reports/validation_report.md` + png | 1 | ~2 min |
| 6 | `src.era_adjust` | `reports/era_adjustment.md` + png | 1 | ~5 min (solve_wp ×~32) |
| 7 | `src.baseline` | `reports/baseline_comparison.md` + png | 1 | ~5 min |
| 8 | `src.tail_diagnostics` | `reports/tail_diagnostics.md` | 1 | ~3 min |
| 9 | `src.dependence_decomposition` | `reports/dependence_decomposition.md` | 1 | ~20 s |
| 10 | `src.correlation_experiment` | `reports/correlation_experiment.md` | 1 | **~35 min** (6 seeds) |
| 11 | `src.player_uncertainty` | `reports/player_uncertainty.md` | 2 (needs `wp`,`li`) | ~1 min |
| 12 | `paper/make_figures.py` (script, not -m) | `paper/figures/F3,F4` | 9, 10 (parses those reports) | ~5 s |

Steps 3–11 are independent of each other except as noted; only re-run what
changed. Warn the user before launching step 10.

## After regeneration

1. Run the test suite: `.venv/bin/python -m pytest tests/ -q` (all must pass;
   several tests are data-gated on the parquet).
2. If any regenerated numbers changed, tell the user which `paper/draft_*.md`
   sections cite them (the ledger is the claim→evidence map in
   `paper/scaffold.md`) — or run /paper-sync.
3. Commit regenerated reports together with the code change that caused them,
   never separately.
