"""M1 -- Exact Markov win probability (spec 03).

Given an `outcome_dist(state) -> {outcome: prob}` (spec 02), derive WP for every
live state by backward induction over the acyclic (b, w, r) graph. No statistics
live here -- only terminal conditions and the recursion. Test with a
hand-specified distribution before wiring in a fitted model.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .config import BALLS_PER_INNINGS, OUTCOMES, TIE_VALUE, WICKETS
from .state import State, transition

OutcomeDist = Callable[[State], dict[str, float]]


def terminal_value(state: State) -> float | None:
    """Return the terminal WP, or None if the state is live. Order matters.

    r <= 0 is checked first and may fire at any b (target reached mid-over); it
    only ever returns early and never recurses, so induction on b is unaffected.
    """
    b, w, r = state
    if r <= 0:
        return 1.0  # target reached, won
    if w == 0:
        return 0.0  # all out, lost
    if b == 0:
        # out of balls with runs still required
        return TIE_VALUE if r == 1 else 0.0  # r == 1 means finished level -> tie
    return None  # live


class WPTable:
    """Precomputed WP surface with a callable interface `wp(state) -> float`."""

    def __init__(self, arr: np.ndarray, r_max: int):
        # arr[b, w, r] holds WP for live states; terminal states resolved on the fly.
        self._arr = arr
        self.r_max = r_max

    def __call__(self, state: State) -> float:
        b, w, r = state
        tv = terminal_value(state)
        if tv is not None:
            return tv
        if r > self.r_max:
            raise KeyError(f"r={r} exceeds tabulated r_max={self.r_max}; raise r_max")
        return float(self._arr[b, w, r])

    def predict(self, b, w, r) -> np.ndarray:
        """Vectorized WP lookup for arrays of live states (1<=r<=r_max)."""
        b = np.asarray(b, dtype=int)
        w = np.asarray(w, dtype=int)
        r = np.asarray(r, dtype=int)
        if r.max() > self.r_max:
            raise KeyError(f"r={r.max()} exceeds tabulated r_max={self.r_max}; raise r_max")
        return self._arr[b, w, r]


def solve_wp(
    outcome_dist: OutcomeDist,
    b_max: int = BALLS_PER_INNINGS,
    w_max: int = WICKETS,
    r_max: int = 400,
) -> WPTable:
    """Backward induction ordered by ascending b. Each b=k value is a finite sum
    over already-computed b=k-1 values, so a single sweep is exact."""
    # arr[b, w, r]; only live cells (b>=1, w>=1, 1<=r<=r_max) are meaningful.
    arr = np.zeros((b_max + 1, w_max + 1, r_max + 1), dtype=np.float64)

    def child(b: int, w: int, r: int) -> float:
        if r <= 0:
            return 1.0
        if w == 0:
            return 0.0
        if b == 0:
            return TIE_VALUE if r == 1 else 0.0
        return arr[b, w, r]

    for b in range(1, b_max + 1):
        for w in range(1, w_max + 1):
            for r in range(1, r_max + 1):
                dist = outcome_dist((b, w, r))
                acc = 0.0
                for o in OUTCOMES:
                    p = dist.get(o, 0.0)
                    if p == 0.0:
                        continue
                    nb, nw, nr = transition((b, w, r), o)
                    acc += p * child(nb, nw, nr)
                arr[b, w, r] = acc

    return WPTable(arr, r_max)


def check_consistency(
    wp: WPTable, outcome_dist: OutcomeDist, sample_states, tol: float = 1e-9
) -> list[str]:
    """Self-check: WP(s) == sum_o p_o(s) * WP(s'_o) for live states (spec 03).

    Tests the DP against its own definition (not against reality). Returns a list
    of failure strings; empty means consistent.
    """
    fails = []
    for s in sample_states:
        if terminal_value(s) is not None:
            continue
        dist = outcome_dist(s)
        implied = sum(p * wp(transition(s, o)) for o, p in dist.items())
        if abs(implied - wp(s)) > tol:
            fails.append(f"WP{s}={wp(s):.6f} != sum_o p_o WP(s')={implied:.6f}")
    return fails
