"""The exact deterministic core: state transitions (spec 03).

No statistics live here. A state is the tuple (b, w, r):
    b = legal balls remaining before the ball
    w = wickets in hand before the ball
    r = runs still required before the ball

Invariant that makes backward induction exact (spec 00): every non-terminal
outcome decrements b by exactly one, so the transition graph is a DAG ordered by
b and no state is ever revisited.
"""

from __future__ import annotations

State = tuple[int, int, int]


def transition(state: State, o: str) -> State:
    """Next state after outcome `o` in {"0".."6", "W"}.

    Wicket: lose a wicket, use a ball, runs unchanged.
    Runs k: use a ball, subtract k from runs required.
    """
    b, w, r = state
    if o == "W":
        return (b - 1, w - 1, r)
    k = int(o)
    return (b - 1, w, r - k)
