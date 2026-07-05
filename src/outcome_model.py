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

Era adjustment (added after M2.5): the fitters accept optional per-row `weights`.
Passing recency weights (see `recency_weights`) tilts the estimated p_o(state)
toward recent seasons, correcting the scoring-era shift the M2.5 diagnostics found
to be the dominant held-out error. This lives entirely in the estimation layer:
still ONE outcome distribution, still solved by one backward-induction sweep, so
WP remains an exact martingale WP(s) = sum_o p_o WP(s'). The DP state is untouched.
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


# --- era adjustment: recency weighting (estimation-layer only) --------------
def recency_weights(
    train: pd.DataFrame, half_life: float | None, season_col: str = "season", date_col: str = "date"
) -> np.ndarray:
    """Exponential recency weights over the chronological season order.

    A ball in the season `k` seasons before the most recent one in `train` gets
    weight `0.5 ** (k / half_life)`: the latest season weighs 1, and weight halves
    every `half_life` seasons back. `half_life=None` (or non-finite) returns uniform
    weights, recovering the unweighted fit exactly.

    Unit is the SEASON ordinal, not the calendar year: IPL is annual and the odd
    labels ("2007/08", "2020/21") are one season each, so ordinal distance is the
    natural "seasons ago". Order is taken from the passed frame's own seasons (by
    first match date), so carving a validation season out of `train` correctly
    makes the newest remaining season the reference point -- no leakage.

    Weights are normalized to mean 1 so the additive floor (`fit_base`) and the
    Dirichlet concentration `alpha` (`_shrink`) keep their unweighted meaning: a
    weighted cell has the same effective sample size as its raw row count.
    """
    if half_life is None or not np.isfinite(half_life):
        return np.ones(len(train), dtype=float)
    order = train.groupby(season_col)[date_col].min().sort_values()
    season_idx = {s: i for i, s in enumerate(order.index)}
    latest = len(season_idx) - 1
    ages = train[season_col].map(lambda s: latest - season_idx[s]).to_numpy(dtype=float)
    w = 0.5 ** (ages / half_life)
    return w * (len(w) / w.sum())  # mean 1


def _weighted_counts(train: pd.DataFrame, keys: list[str], weights: np.ndarray) -> dict:
    """Sum weights by (keys..., outcome). With unit weights this equals the raw
    value_counts used before the era adjustment (identical fit when weights=1)."""
    g = train.assign(_w=weights).groupby([*keys, "outcome"], observed=True)["_w"].sum()
    out: dict = {}
    for idx, c in g.items():
        key = idx[:-1]
        outcome = idx[-1]
        key = key[0] if len(key) == 1 else tuple(key)
        out.setdefault(key, {})[outcome] = float(c)
    return out


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_base(train: pd.DataFrame, weights: np.ndarray | None = None) -> dict[str, float]:
    """Global empirical outcome frequency over training legal balls (V1).

    Given a small additive floor so every outcome is strictly positive (leverage
    and log loss both dislike zeros), which also makes it a safe shrinkage prior.

    `weights` (mean-1 per-row recency weights, see `recency_weights`) tilts the
    frequency toward recent seasons; `None` recovers the plain value-count fit.
    """
    if weights is None:
        counts = train["outcome"].value_counts().to_dict()
    else:
        counts = train.assign(_w=weights).groupby("outcome", observed=True)["_w"].sum().to_dict()
    floored = {o: counts.get(o, 0.0) + 0.5 for o in OUTCOMES}
    return _normalize(floored)


def _shrink(counts: dict[str, float], prior: dict[str, float], alpha: float) -> dict[str, float]:
    """Dirichlet-style shrinkage of `counts` toward `prior` with concentration
    `alpha` pseudo-counts. Large cells barely move; sparse cells fall back to the
    prior. Positivity is inherited from the (positive) prior."""
    n = sum(counts.values())
    return {o: (counts.get(o, 0.0) + alpha * prior[o]) / (n + alpha) for o in OUTCOMES}


def fit_rrr(train: pd.DataFrame, alpha: float = 10.0, weights: np.ndarray | None = None) -> dict:
    """V2: distribution keyed on (rrr_bin, w, phase), hierarchically smoothed.

    Two-level shrinkage (spec 02): each cell shrinks toward its (phase, w) parent
    marginal, which itself shrinks toward the global base. Unseen keys fall back
    down the same chain at lookup time. Fit on the TRAIN split only.

    `weights` (mean-1 per-row recency weights, see `recency_weights`) makes every
    level -- global base, (phase, w) parents, and cells -- recency-weighted from
    the same weights, so the whole fallback chain is era-consistent. `None`
    reproduces the original M2 fit bit-for-bit (unit weights collapse the weighted
    sums back to value counts).
    """
    w_arr = np.ones(len(train), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    base = fit_base(train, weights=w_arr)

    train = train.assign(rrr_bin=train["rrr"].map(rrr_bin))

    def counts_by(keys: list[str]) -> dict:
        return _weighted_counts(train, keys, w_arr)

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
