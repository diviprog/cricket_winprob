---
name: paper-sync
description: Cross-check every number and mechanism claim in the paper drafts against the committed reports. Use after regenerating any report, before committing paper prose, or when the user asks whether the paper is consistent with the analysis.
---

Verify that `paper/` is consistent with `reports/` — the reports are the source
of truth; drafts trail them (code/analysis-first repo).

## Procedure

1. **Ledger first.** Read the claim→evidence map at the bottom of
   `paper/scaffold.md`. For each row, open the named report and confirm the
   number still matches. Flag mismatches with old → new values.
2. **Draft sweep.** Grep each `paper/draft_*.md` for numerals and check the
   load-bearing ones against their reports. Priority order:
   - `draft_06_07.md` (the spine): Table T2 gap values, TV bounds, lag-profile
     table, K-curve table, 26%/6-seed claims, OOS values.
   - `draft_03_05.md`: Table T1 metrics, era-adjustment numbers, LI/WPA figures.
   - `draft_01_02.md` and `draft_08_10.md`: headline numbers only (0.094 nats,
     26%, ~3–5-ball range, 18%, +0.037, 2.7×).
3. **Mechanism phrasing.** The mechanism is *sequential run-scoring persistence;
   wickets ANTI-cluster; heterogeneity ~18%*. Grep drafts + README for stale
   phrasings — any unqualified "wicket clusters", "tail-thinning" (as a
   confirmed mechanism), or unscoped "cannot have both" (must be scoped to the
   (b,w,r) state / filtration).
4. **Figures.** If `dependence_decomposition.md` or `correlation_experiment.md`
   changed, re-run `.venv/bin/python paper/make_figures.py` (it parses those
   reports, so figures and text stay identical by construction) and view the
   PNGs to confirm they render.
5. **Report.** List every inconsistency found (file, line, old → new) and fix
   them in the drafts — never by editing a report. Update the scaffold ledger
   if a number legitimately changed. If nothing is stale, say so explicitly.
