"""02 -- Outcome model: the one estimated object for the Markov WP.

Every version implements the same interface

    outcome_dist(state) -> {outcome: prob}   # state = (b, w, r); sums to 1.0

and is produced behind `make_outcome_model(kind, fitted_params)` so it is
interchangeable in the WP core (03) and validation (05). RRR and phase are
features DERIVED from the state for estimation only; they never enter the DP
state (spec 00).

Kinds:
  "base"   -- single global distribution (V1 scaffold, spec 02 V1)
  "rrr"    -- conditioned on (rrr_bin, w, phase) with smoothing (V2, added in M2)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .config import OUTCOMES
from .state import State

OutcomeDist = Callable[[State], dict[str, float]]


# --- feature derivation (estimation-only; not DP state) --------------------
# RRR bin edges (spec 02). Tuned against data density in M2.
RRR_BIN_EDGES = [6.0, 8.0, 10.0, 12.0, 15.0]  # -> 6 bins: <6,6-8,8-10,10-12,12-15,15+


def rrr_bin(rrr: float) -> int:
    """Bucket required run rate into an ordinal bin index."""
    return int(np.searchsorted(RRR_BIN_EDGES, rrr, side="right"))


def phase_of(b: int) -> str:
    """Coarse innings stage from balls remaining (spec 02)."""
    balls_bowled = 120 - b
    over_idx = balls_bowled // 6
    if over_idx < 6:
        return "powerplay"
    if over_idx < 15:
        return "middle"
    return "death"


def _normalize(counts: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values())
    return {o: counts.get(o, 0.0) / total for o in OUTCOMES}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_base(train: pd.DataFrame) -> dict[str, float]:
    """Global empirical outcome frequency over training legal balls (V1).

    Given a small additive floor so every outcome is strictly positive (leverage
    and log loss both dislike zeros), which also makes it a safe shrinkage prior.
    """
    counts = train["outcome"].value_counts().to_dict()
    floored = {o: counts.get(o, 0.0) + 0.5 for o in OUTCOMES}
    return _normalize(floored)


def _shrink(counts: dict[str, float], prior: dict[str, float], alpha: float) -> dict[str, float]:
    """Dirichlet-style shrinkage of `counts` toward `prior` with concentration
    `alpha` pseudo-counts. Large cells barely move; sparse cells fall back to the
    prior. Positivity is inherited from the (positive) prior."""
    n = sum(counts.values())
    return {o: (counts.get(o, 0.0) + alpha * prior[o]) / (n + alpha) for o in OUTCOMES}


def fit_rrr(train: pd.DataFrame, alpha: float = 10.0) -> dict:
    """V2: distribution keyed on (rrr_bin, w, phase), hierarchically smoothed.

    Two-level shrinkage (spec 02): each cell shrinks toward its (phase, w) parent
    marginal, which itself shrinks toward the global base. Unseen keys fall back
    down the same chain at lookup time. Fit on the TRAIN split only.
    """
    base = fit_base(train)

    def counts_by(keys: list[str]) -> dict:
        g = train.groupby(keys)["outcome"].value_counts()
        out: dict = {}
        for idx, c in g.items():
            key = idx[:-1]
            outcome = idx[-1]
            key = key[0] if len(key) == 1 else tuple(key)
            out.setdefault(key, {})[outcome] = float(c)
        return out

    train = train.assign(rrr_bin=train["rrr"].map(rrr_bin))

    parent_counts = counts_by(["phase", "w"])
    parents = {k: _shrink(c, base, alpha) for k, c in parent_counts.items()}

    cell_counts = counts_by(["rrr_bin", "w", "phase"])
    cells = {}
    for (rb, w, ph), c in cell_counts.items():
        parent = parents.get((ph, w), base)
        cells[(rb, w, ph)] = _shrink(c, parent, alpha)

    return {"cells": cells, "parents": parents, "base": base}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_outcome_model(kind: str, fitted_params) -> OutcomeDist:
    if kind == "base":
        dist = dict(fitted_params)  # state-independent

        def outcome_dist(state: State) -> dict[str, float]:
            return dist

        return outcome_dist

    if kind == "rrr":
        cells = fitted_params["cells"]
        parents = fitted_params["parents"]
        base = fitted_params["base"]

        def outcome_dist(state: State) -> dict[str, float]:
            b, w, r = state
            rb = rrr_bin(6.0 * r / b)  # b >= 1 for live states
            ph = phase_of(b)
            # fallback chain: (rrr_bin, w, phase) -> (phase, w) parent -> base
            d = cells.get((rb, w, ph))
            if d is None:
                d = parents.get((ph, w), base)
            return d

        return outcome_dist

    raise ValueError(f"unknown outcome model kind: {kind!r}")
