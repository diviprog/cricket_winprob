"""M0 gate as a test. Skips if the processed parquet has not been built yet."""

import pandas as pd
import pytest

from src import config
from src.ingest import run_sanity_checks

pytestmark = pytest.mark.skipif(
    not config.BALLS_PARQUET.exists(),
    reason="run `uv run python -m src.ingest` first to build the ball table",
)


@pytest.fixture(scope="module")
def balls():
    return pd.read_parquet(config.BALLS_PARQUET)


def test_m0_sanity_gate(balls):
    fails = run_sanity_checks(balls)
    assert not fails, "M0 sanity gate failures:\n" + "\n".join(fails)


def test_contract_columns(balls):
    expected = {"match_id", "season", "b", "w", "r", "outcome", "y", "rrr", "phase", "split"}
    assert expected <= set(balls.columns)


def test_outcomes_in_vocabulary(balls):
    assert set(balls["outcome"].unique()) <= set(config.OUTCOMES)


def test_split_has_both_partitions(balls):
    assert set(balls["split"].unique()) == {"train", "holdout"}
