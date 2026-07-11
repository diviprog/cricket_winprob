"""Guardrails for the correlation experiment's replication + scattered arm.

Data-gated (need the processed ball table); run with tiny parameters so the
suite stays fast -- these check MECHANICS (reproducibility, marginal fidelity,
fallback bookkeeping), not the scientific numbers.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.correlation_experiment import BlockSampler, run_block_experiment
from src.outcome_model import fit_rrr, make_outcome_model
from src.wp_markov import solve_wp

data = pytest.mark.skipif(
    not config.BALLS_PARQUET.exists(),
    reason="run `.venv/bin/python -m src.ingest` first",
)

TINY = dict(block_lengths=[1, 3], per_slice=8, n_sims=40, scattered_k=3)


@pytest.fixture(scope="module")
def setup():
    df = pd.read_parquet(config.BALLS_PARQUET)
    train = df[df["split"] == "train"].copy()
    dist = make_outcome_model("rrr", fit_rrr(train))
    wp = solve_wp(dist, r_max=int(df["r"].max()) + 1)
    return train, dist, wp, BlockSampler(train)


@data
def test_same_seed_reproduces_bit_identically(setup):
    """The seed drives both the state sample and the MC draws; a replicate with
    the same seed must reproduce every number exactly, a different seed must not."""
    train, dist, wp, sampler = setup
    a, _, _ = run_block_experiment(train, dist, wp, seed=7, sampler=sampler, **TINY)
    b, _, _ = run_block_experiment(train, dist, wp, seed=7, sampler=sampler, **TINY)
    c, _, _ = run_block_experiment(train, dist, wp, seed=8, sampler=sampler, **TINY)
    num = a.select_dtypes("number").columns
    assert np.allclose(a[num].to_numpy(), b[num].to_numpy(), equal_nan=True)
    assert not np.allclose(a[num].to_numpy(), c[num].to_numpy(), equal_nan=True)


@data
def test_scattered_mode_marginal_fidelity_and_fallback(setup):
    """Mechanics of the scattered arm's bookkeeping. NOTE: the full-run fidelity
    number (TV ~0.018, ~5-10x the block modes') is precisely the diagnostic that
    exposed the arm as degenerate (see correlation_experiment.py docstring) -- so
    this test asserts the bookkeeping is computed and sane (loose bound), not that
    the arm is a valid mechanism probe."""
    train, dist, wp, sampler = setup
    _, realized, fb_rate = run_block_experiment(
        train, dist, wp, seed=0, sampler=sampler,
        block_lengths=[1], per_slice=25, n_sims=120, scattered_k=20)
    tv = 0.5 * np.abs(realized["scattered_K20"] - realized["block_K1"]).sum()
    assert tv < 0.05  # marginals preserved (loose bound: tiny-sample MC noise)
    assert 0.0 <= fb_rate < 0.5  # donors usually cover the needed cells


@data
def test_scattered_draws_come_from_the_donor_innings(setup):
    """Index integrity: every (innings, cell) pool in the sampler contains only
    positions whose innings and cell actually match the key."""
    train, dist, wp, sampler = setup
    rng = np.random.default_rng(0)
    keys = rng.choice(list(sampler.ic_ptr.keys()), size=200)
    for cb in keys:
        start, length = sampler.ic_ptr[int(cb)]
        pos = sampler.ic_positions[start:start + length]
        assert (sampler.inn_of[pos] == int(cb) // 1000).all()
        assert (sampler.cell[pos] == int(cb) % 1000).all()
