"""M2: outcome model guardrails (spec 02). Needs the processed ball table."""

import pandas as pd
import pytest

from src import config
from src.outcome_model import OUTCOMES, fit_base, fit_rrr, make_outcome_model

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
