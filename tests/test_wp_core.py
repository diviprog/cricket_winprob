"""M1 gate: the exact core, tested with a hand-specified distribution (no data).

Covers the WP-consistency self-check (DP against its own definition) and the
spec 03 sanity WPs and monotonicity trends.
"""

import random

import pytest

from src.outcome_model import make_outcome_model
from src.state import transition
from src.wp_markov import check_consistency, solve_wp, terminal_value

# A plausible hand-set T20 outcome distribution (sums to 1), close to the global
# empirical frequencies. Being homogeneous it is a crude scaffold, so the sanity
# WPs below use thresholds appropriate to that (a realistic model pushes them
# closer to the spec's ~1 / ~0 ideals).
HAND_DIST = {
    "0": 0.31,
    "1": 0.39,
    "2": 0.065,
    "3": 0.005,
    "4": 0.12,
    "5": 0.001,
    "6": 0.058,
    "W": 0.051,
}


@pytest.fixture(scope="module")
def wp():
    model = make_outcome_model("base", HAND_DIST)
    return solve_wp(model, r_max=300)


@pytest.fixture(scope="module")
def model():
    return make_outcome_model("base", HAND_DIST)


def test_hand_dist_sums_to_one():
    assert abs(sum(HAND_DIST.values()) - 1.0) < 1e-12


def test_transition_decrements_b_by_one():
    for o in ["0", "1", "2", "3", "4", "5", "6", "W"]:
        nb, nw, nr = transition((50, 5, 40), o)
        assert nb == 49  # every outcome uses exactly one ball


def test_terminal_conditions():
    assert terminal_value((10, 5, 0)) == 1.0  # target reached
    assert terminal_value((10, 5, -3)) == 1.0  # overshoot
    assert terminal_value((10, 0, 20)) == 0.0  # all out
    assert terminal_value((0, 5, 5)) == 0.0  # out of balls, behind
    assert terminal_value((0, 5, 1)) == 0.5  # finished level -> tie
    assert terminal_value((10, 5, 20)) is None  # live


def test_wp_consistency_random_sample(wp, model):
    rng = random.Random(0)
    states = [(rng.randint(1, 120), rng.randint(1, 10), rng.randint(1, 250)) for _ in range(2000)]
    fails = check_consistency(wp, model, states)
    assert not fails, "\n".join(fails[:10])


def test_wp_range_valid(wp):
    for s in [(120, 10, 180), (6, 5, 2), (6, 1, 40), (60, 5, 60)]:
        assert 0.0 <= wp(s) <= 1.0


def test_sanity_wps(wp):
    # Thresholds relaxed for the crude homogeneous scaffold (spec 03 quotes the
    # ideal ~1 / ~0 for a realistic model).
    assert wp((6, 5, 2)) > 0.85  # 2 needed off 6, 5 in hand -> very likely
    assert wp((6, 1, 40)) < 0.02  # 40 off 6, last pair -> near hopeless
    assert 0.30 < wp((120, 10, 170)) < 0.70  # plausible pre-chase number


def test_monotonic_in_r(wp):
    # WP increases as runs required falls (holding b, w)
    vals = [wp((60, 5, r)) for r in range(80, 20, -5)]
    assert all(x <= y + 1e-12 for x, y in zip(vals, vals[1:]))


def test_monotonic_in_w(wp):
    # WP increases with wickets in hand (holding b, r)
    vals = [wp((60, w, 60)) for w in range(1, 11)]
    assert all(x <= y + 1e-12 for x, y in zip(vals, vals[1:]))
