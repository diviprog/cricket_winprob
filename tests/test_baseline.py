"""M4 guardrails (spec 06): feature correctness, no leakage, bootstrap sanity.

The bootstrap tests are pure-math and need no data. The feature tests need the
processed ball table.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.baseline import (
    FULL_FEATS,
    LOGIT_FEATS,
    MARKOV_FEATS,
    build_features,
    cluster_bootstrap_delta,
    per_ball_log_loss,
)

data = pytest.mark.skipif(
    not config.BALLS_PARQUET.exists(),
    reason="run `.venv/bin/python -m src.ingest` first",
)


# --- bootstrap (pure math) --------------------------------------------------
def test_bootstrap_point_estimate_is_the_pooled_mean_difference():
    rng = np.random.default_rng(0)
    a, b = rng.random(500), rng.random(500)
    mids = np.repeat(np.arange(50), 10)
    point, lo, hi = cluster_bootstrap_delta(a, b, mids, n_boot=200)
    assert np.isclose(point, b.mean() - a.mean())
    assert lo < point < hi


def test_bootstrap_identical_models_give_exactly_zero_delta_and_zero_width_ci():
    """Paired on the same predictions, every resample difference is exactly 0."""
    rng = np.random.default_rng(1)
    a = rng.random(300)
    mids = np.repeat(np.arange(30), 10)
    point, lo, hi = cluster_bootstrap_delta(a, a.copy(), mids, n_boot=200)
    assert point == 0.0 and lo == 0.0 and hi == 0.0


def test_bootstrap_detects_a_real_uniform_improvement():
    """B beats A by a constant on every ball -> CI must exclude 0 and sit negative."""
    rng = np.random.default_rng(2)
    a = rng.random(1000) + 1.0
    b = a - 0.1
    mids = np.repeat(np.arange(100), 10)
    point, lo, hi = cluster_bootstrap_delta(a, b, mids, n_boot=300)
    assert np.isclose(point, -0.1)
    assert hi < 0


def test_bootstrap_clustering_widens_the_ci_versus_pretending_balls_are_independent():
    """The whole reason we cluster: with a per-match effect, treating each ball as
    its own cluster understates uncertainty. 100 matches x 50 balls."""
    rng = np.random.default_rng(3)
    match_effect = rng.normal(0, 1.0, size=100)
    d = np.repeat(match_effect, 50) + rng.normal(0, 0.01, size=5000)
    a = np.zeros(5000)
    mids = np.repeat(np.arange(100), 50)
    _, lo_c, hi_c = cluster_bootstrap_delta(a, d, mids, n_boot=400)
    _, lo_i, hi_i = cluster_bootstrap_delta(a, d, np.arange(5000), n_boot=400)
    assert (hi_c - lo_c) > 5 * (hi_i - lo_i)


def test_per_ball_log_loss_matches_the_mean_used_by_validation():
    from src.validation import log_loss

    rng = np.random.default_rng(4)
    p, y = rng.uniform(0.05, 0.95, 200), rng.integers(0, 2, 200)
    assert np.isclose(per_ball_log_loss(p, y).mean(), log_loss(p, y))


# --- features (data-backed) -------------------------------------------------
@data
def test_target_recovered_equals_runs_required_at_first_ball():
    """build_features derives `target` as max(r) per match; it must equal r on the
    innings' first ball, where no runs have been scored yet."""
    df = pd.read_parquet(config.BALLS_PARQUET)
    f = build_features(df)
    first = f.sort_values("ball_index").groupby("match_id").first()
    assert (first["target"] == first["r"]).all()


@data
def test_crr_is_consistent_with_runs_scored_and_balls_bowled():
    df = pd.read_parquet(config.BALLS_PARQUET)
    f = build_features(df)
    bowled = config.BALLS_PER_INNINGS - f["b"]
    live = bowled > 0
    manual = 6.0 * (f["target"] - f["r"]) / bowled.where(live, 1)
    assert np.allclose(f.loc[live, "crr"], manual[live])
    assert (f.loc[~live, "crr"] == 0.0).all()  # ball 1: no rate exists yet


@data
def test_no_feature_leaks_the_label():
    """A feature that encodes y would let a single-split decision stump score
    perfectly. `y` and post-ball columns must never enter a feature set."""
    banned = {"y", "outcome", "split"}
    for feats in (MARKOV_FEATS, LOGIT_FEATS, FULL_FEATS):
        assert not (set(feats) & banned)

    df = pd.read_parquet(config.BALLS_PARQUET)
    f = build_features(df)
    # no engineered feature may be a deterministic function of the label:
    # correlation is a weak check, but a perfect |corr| ~ 1 would be damning.
    for c in FULL_FEATS:
        assert abs(np.corrcoef(f[c].astype(float), f["y"].astype(float))[0, 1]) < 0.95


@data
def test_striker_balls_is_a_strictly_pre_ball_counter():
    """`striker_balls` counts DELIVERIES the striker faced before this ball.

    It is NOT bounded by `120 - b` (legal balls bowled). Spec 01 folds extras
    rather than modelling them as balls, so a wide/no-ball increments the raw
    `batter_balls` counter without decrementing `b`. Measured on this dataset:
    35 of 130,029 rows exceed `120 - b` (max overshoot 2), and 31 of 8,694
    batter-innings open at striker_balls > 0 because the batter faced an extra
    before their first legal ball. Both are the extras decision, not a bug.

    What DOES hold, and is what the M4 ablation relies on: the counter is
    strictly pre-ball -- it never decreases and never repeats a value, so it can
    encode no information about the ball currently being predicted.
    """
    df = pd.read_parquet(config.BALLS_PARQUET).sort_values(["match_id", "ball_index"])
    assert df["striker_balls"].min() == 0
    assert df["striker_balls"].max() < config.BALLS_PER_INNINGS

    d = df.groupby(["match_id", "batter"], sort=False)["striker_balls"].diff().dropna()
    # never 0 or negative: a strictly increasing pre-ball counter, no double counting
    assert (d >= 1).all()
    # +1 is the norm; >1 means extras were faced in between (0.38% of increments)
    assert (d == 1).mean() > 0.99
