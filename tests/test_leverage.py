"""M3: leverage guardrails (spec 04/05). Pure-math checks need no data; the
normalization check needs the processed ball table."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.leverage import (
    average_swing,
    compute_leverage,
    era_wp,
    leverage_index,
    swing,
)
from src.outcome_model import make_outcome_model
from src.wp_markov import solve_wp

R_MAX = 300


# --- pure math (no data) ---------------------------------------------------
def test_closed_form_coin_flip_ending_is_half():
    """spec 04: when a ball ends the match as a coin flip, swing = 2p(1-p) -> 0.5
    at p=0.5, the theoretical maximum. At (1,1,1): a single wins, a wicket loses."""
    half = {o: 0.0 for o in ["0", "1", "2", "3", "4", "5", "6", "W"]}
    half["1"], half["W"] = 0.5, 0.5
    dist = make_outcome_model("base", half)
    wp = solve_wp(dist, r_max=R_MAX)
    assert abs(swing((1, 1, 1), dist, wp) - 0.5) < 1e-9


def test_swing_nonnegative_and_dead_state_is_zero():
    """A state that is already decided has ~0 swing: every landing keeps WP pinned."""
    d = {"0": 0.4, "1": 0.35, "2": 0.06, "3": 0.01, "4": 0.1, "5": 0.0, "6": 0.05, "W": 0.03}
    dist = make_outcome_model("base", d)
    wp = solve_wp(dist, r_max=R_MAX)
    # 1 run needed off 120 balls, 10 in hand -> WP ~ 1, no ball moves it
    assert swing((120, 10, 1), dist, wp) < 1e-3
    # swing is a probability-weighted absolute deviation -> never negative
    for s in [(1, 1, 1), (60, 5, 60), (12, 3, 20), (120, 10, 1)]:
        assert swing(s, dist, wp) >= 0.0


def test_coin_flip_more_leveraged_than_lopsided():
    d = {"0": 0.4, "1": 0.35, "2": 0.06, "3": 0.01, "4": 0.1, "5": 0.0, "6": 0.05, "W": 0.03}
    dist = make_outcome_model("base", d)
    wp = solve_wp(dist, r_max=R_MAX)
    # a tight last-over state must swing more than an early comfortable one
    assert swing((3, 5, 4), dist, wp) > swing((114, 9, 20), dist, wp)


# --- data-backed normalization --------------------------------------------
data = pytest.mark.skipif(
    not config.BALLS_PARQUET.exists(),
    reason="run `uv run python -m src.ingest` first",
)


@data
def test_leverage_index_normalized_to_mean_one():
    """LI = swing / avg_swing over real balls, so the mean LI across the reference
    set is exactly 1 by construction (spec 04)."""
    df = pd.read_parquet(config.BALLS_PARQUET)
    train = df[df["split"] == "train"]
    r_max = int(df["r"].max()) + 1
    dist, wp = era_wp(train, r_max)
    li, avg = compute_leverage(df, dist, wp)
    assert abs(float(li.mean()) - 1.0) < 1e-9
    assert avg > 0
    # leverage_index at a state equals its swing scaled by avg
    s = (1, 6, 3)  # a tight last-over state present in the data
    assert np.isclose(leverage_index(s, avg, dist, wp), swing(s, dist, wp) / avg)


@data
def test_average_swing_matches_manual_mean():
    df = pd.read_parquet(config.BALLS_PARQUET)
    train = df[df["split"] == "train"]
    dist, wp = era_wp(train, int(df["r"].max()) + 1)
    states = list(zip(df["b"].head(2000), df["w"].head(2000), df["r"].head(2000)))
    got = average_swing(states, dist, wp)
    manual = np.mean([swing(s, dist, wp) for s in states])
    assert np.isclose(got, manual)
