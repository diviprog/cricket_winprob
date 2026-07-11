"""Guardrails for the dependence decomposition (serial vs shared-latent).

The synthetic tests validate the DISCRIMINATOR itself: the within-innings
permutation null must (a) sit at ~0 for iid data, (b) absorb a pure innings-level
random effect (so a shared latent is NOT flagged as sequential dependence), and
(c) flag genuine AR(1) structure. The data-gated test ties the machinery back to
the M3 headline autocorrelation.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.dependence_decomposition import (
    band,
    demean_by,
    icc_oneway,
    lag_corr,
    observed_and_null,
    prep,
    within_group_shuffle,
)


def _groups(n_groups: int, size: int) -> np.ndarray:
    return np.repeat(np.arange(n_groups), size)


# --- primitives -------------------------------------------------------------
def test_lag_corr_matches_manual_within_groups():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    g = _groups(2, 100)
    got, n = lag_corr(x, g, 1)
    # manual: pairs (t, t+1) that do not cross the group boundary at 100
    a = np.concatenate([x[0:99], x[100:199]])
    b = np.concatenate([x[1:100], x[101:200]])
    assert n == 198
    assert np.isclose(got, np.corrcoef(a, b)[0, 1])


def test_demean_by_zeroes_group_means():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=3.0, size=500)
    g = _groups(5, 100)
    d = demean_by(x, g)
    for k in range(5):
        assert abs(d[g == k].mean()) < 1e-12


def test_within_group_shuffle_permutes_values_within_groups_only():
    rng = np.random.default_rng(2)
    x = np.arange(300, dtype=float)
    g = _groups(3, 100)
    xp = within_group_shuffle(x, g, rng)
    for k in range(3):
        assert set(xp[g == k]) == set(x[g == k])  # same values per group
    assert not np.array_equal(xp, x)  # actually shuffled


# --- the discriminator ------------------------------------------------------
def test_iid_data_null_centred_at_zero_and_observed_inside_band():
    rng = np.random.default_rng(3)
    n_g, size = 300, 60
    x = rng.normal(size=n_g * size)
    inn = _groups(n_g, size)
    part = _groups(n_g * 3, size // 3)  # arbitrary finer partition
    obs, null = observed_and_null(x, inn, part, n_perm=100, seed=0)
    m, lo, hi = band(null[("state", "innings", 1)])
    assert abs(m) < 0.01  # null centred at ~0 with no latent
    assert lo <= obs[("state", "innings", 1)] <= hi  # iid not flagged


def test_innings_random_effect_recovered_by_icc_and_absorbed_by_null():
    """A pure shared latent: x = u[innings] + noise. The ICC must recover the
    variance share, the lag profile must be FLAT at ~ICC, and -- the critical
    property -- the permutation null must ABSORB it (observed inside the band),
    because permutation preserves innings composition."""
    rng = np.random.default_rng(4)
    n_g, size = 400, 60
    u = rng.normal(size=n_g)  # var 1
    x = np.repeat(u, size) + rng.normal(scale=2.0, size=n_g * size)  # noise var 4
    inn = _groups(n_g, size)
    part = _groups(n_g * 3, size // 3)
    true_icc = 1.0 / (1.0 + 4.0)

    assert abs(icc_oneway(x, inn) - true_icc) < 0.03

    obs, null = observed_and_null(x, inn, part, n_perm=100, seed=0)
    # flat profile at ~ICC for the state variant
    prof = [obs[("state", "innings", lag)] for lag in (1, 5, 20)]
    assert all(abs(p - true_icc) < 0.04 for p in prof)
    # the null band CONTAINS the observed value at every lag: a shared latent
    # must not be flagged as sequential dependence
    for lag in (1, 5, 20):
        m, lo, hi = band(null[("state", "innings", lag)])
        assert lo <= obs[("state", "innings", lag)] <= hi
        assert abs(m - true_icc) < 0.04  # null centre estimates the latent share
    # innings demeaning kills it
    assert abs(obs[("state+innings", "innings", 1)]) < 0.03


def test_ar1_flagged_as_sequential_with_decaying_profile():
    rng = np.random.default_rng(5)
    n_g, size, rho = 300, 60, 0.5
    xs = []
    for _ in range(n_g):
        e = rng.normal(size=size)
        v = np.empty(size)
        v[0] = e[0]
        for t in range(1, size):
            v[t] = rho * v[t - 1] + e[t]
        xs.append(v)
    x = np.concatenate(xs)
    inn = _groups(n_g, size)
    part = _groups(n_g * 3, size // 3)
    obs, null = observed_and_null(x, inn, part, n_perm=100, seed=0)
    # decaying profile ~ rho^lag
    assert obs[("state", "innings", 1)] > 0.4
    assert obs[("state", "innings", 5)] < 0.15
    # flagged: observed far above the null band at lag 1, even after demeaning
    for variant in ("state", "state+innings"):
        m, lo, hi = band(null[(variant, "innings", 1)])
        assert obs[(variant, "innings", 1)] > hi


# --- data-gated continuity ---------------------------------------------------
data = pytest.mark.skipif(
    not config.BALLS_PARQUET.exists(),
    reason="run `.venv/bin/python -m src.ingest` first",
)


@data
def test_partnership_lag1_matches_m3_construction_on_train():
    """The (state variant, partnership scope, lag 1) cell of the profile is the
    same statistic as leverage.partnership_lag1_autocorr -- computed here on the
    train split via a different (vectorized) pairing, so equality is a real
    cross-implementation check."""
    from src.leverage import partnership_lag1_autocorr

    df = pd.read_parquet(config.BALLS_PARQUET)
    train = df[df["split"] == "train"].copy()
    d = prep(train)
    got, _ = lag_corr(d["runs"], d["part"], 1)
    want = partnership_lag1_autocorr(train)["lag1_autocorr"]
    assert np.isclose(got, want, atol=1e-10)
