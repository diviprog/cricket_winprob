"""M2: outcome model guardrails (spec 02). Needs the processed ball table."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.outcome_model import (
    OUTCOMES,
    fit_base,
    fit_rrr,
    make_outcome_model,
    recency_weights,
)

pytestmark = pytest.mark.skipif(
    not config.BALLS_PARQUET.exists(),
    reason="run `uv run python -m src.ingest` first",
)


@pytest.fixture(scope="module")
def train():
    df = pd.read_parquet(config.BALLS_PARQUET)
    return df[df["split"] == "train"]


def _check_dist(d):
    assert abs(sum(d.values()) - 1.0) < 1e-9
    assert set(d) == set(OUTCOMES)
    assert all(p > 0.0 for p in d.values())  # no zeros after smoothing


def test_base_is_valid_dist(train):
    _check_dist(fit_base(train))


def test_rrr_dists_valid_across_state_grid(train):
    model = make_outcome_model("rrr", fit_rrr(train))
    # sweep live states incl. sparse extremes that force fallbacks
    for b in [120, 90, 60, 30, 6, 1]:
        for w in [1, 5, 10]:
            for r in [1, 20, 80, 200]:
                _check_dist(model((b, w, r)))


def test_rrr_unseen_key_falls_back(train):
    model = make_outcome_model("rrr", fit_rrr(train))
    # an absurd-but-live state (huge RRR, deep death) must still return a valid dist
    _check_dist(model((1, 1, 300)))


# --- era adjustment: recency weighting -------------------------------------
def test_recency_weights_uniform_when_none(train):
    w = recency_weights(train, half_life=None)
    assert np.allclose(w, 1.0)


def test_recency_weights_mean_one_and_recent_heavier(train):
    w = recency_weights(train, half_life=3.0)
    assert abs(w.mean() - 1.0) < 1e-9  # normalized so smoothing semantics hold
    order = train.groupby("season")["date"].min().sort_values().index.tolist()
    newest, oldest = order[-1], order[0]
    sw = pd.Series(w, index=train.index)
    # recent seasons must carry strictly more weight per ball than old ones
    assert sw[train["season"] == newest].iloc[0] > sw[train["season"] == oldest].iloc[0]


def test_unit_weights_reproduce_unweighted_fit(train):
    """Passing all-ones weights must reproduce the M2 fit bit-for-bit, so the era
    knob is a strict superset and the M2 gate report is unchanged."""
    a = fit_rrr(train)
    b = fit_rrr(train, weights=np.ones(len(train)))
    assert set(a["cells"]) == set(b["cells"])
    for k in a["cells"]:
        for o in OUTCOMES:
            assert a["cells"][k][o] == b["cells"][k][o]


def test_recency_shifts_distribution_toward_recent_scoring(train):
    """A short half-life must move the global outcome frequency toward the recent,
    higher-scoring era: strictly more expected runs per ball than the flat fit."""

    def exp_runs(d):
        return sum((0 if o == "W" else int(o)) * d[o] for o in OUTCOMES)

    flat = fit_base(train)
    recent = fit_base(train, weights=recency_weights(train, half_life=1.0))
    assert exp_runs(recent) > exp_runs(flat)
