"""Guardrails for the player-level cluster bootstrap (player_uncertainty.py).

Pure-math tests validate the resampler itself (coverage, clustering, masked
degeneracy); the data-gated tests pin determinism and agreement with the
point estimates the wpa module publishes.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.player_uncertainty import (
    N_BOOT,
    SEED,
    bootstrap_player,
    cluster_boot_mean,
)

data = pytest.mark.skipif(
    not config.BALLS_PARQUET.exists(),
    reason="run `.venv/bin/python -m src.ingest` first",
)


def _draws(rng, n_matches, n_boot=500):
    return rng.integers(0, n_matches, size=(n_boot, n_matches))


# --- pure math ---------------------------------------------------------------
def test_cluster_boot_mean_is_deterministic_and_centred_on_the_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1.0, size=600)
    codes = np.repeat(np.arange(60), 10)
    reps1 = cluster_boot_mean(x, codes, 60, _draws(np.random.default_rng(7), 60))
    reps2 = cluster_boot_mean(x, codes, 60, _draws(np.random.default_rng(7), 60))
    assert np.array_equal(reps1, reps2)  # same seed -> bit-identical
    lo, hi = np.percentile(reps1, [2.5, 97.5])
    assert lo < x.mean() < hi  # CI covers the sample mean
    assert lo < 0.5 < hi  # ... and the truth, at this n


def test_clustering_widens_ci_when_matches_carry_a_shared_effect():
    """The reason to cluster: per-match random effects make balls within a match
    co-move; treating each ball as its own cluster understates the variance."""
    rng = np.random.default_rng(1)
    n_matches, per = 80, 40
    match_effect = np.repeat(rng.normal(0, 0.5, n_matches), per)
    x = match_effect + rng.normal(0, 0.2, n_matches * per)
    codes = np.repeat(np.arange(n_matches), per)

    reps_cluster = cluster_boot_mean(x, codes, n_matches, _draws(np.random.default_rng(2), n_matches))
    ball_codes = np.arange(len(x))  # every ball its own "cluster" = iid bootstrap
    reps_iid = cluster_boot_mean(
        x, ball_codes, len(x), np.random.default_rng(3).integers(0, len(x), size=(500, len(x)))
    )
    width = lambda r: np.subtract(*np.percentile(r, [97.5, 2.5]))  # noqa: E731
    assert width(reps_cluster) > 2 * width(reps_iid)


def test_masked_mean_returns_nan_only_when_resample_has_no_masked_balls():
    """Mask lives entirely in match 0; replicates that never draw match 0 must be
    NaN, all others must equal the masked values' resampled mean (finite)."""
    x = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 1.0])
    codes = np.array([0, 0, 1, 1, 2, 2])
    mask = np.array([True, True, False, False, False, False])
    draws = np.array([[0, 1, 2], [1, 2, 2], [0, 0, 1]])
    reps = cluster_boot_mean(x, codes, 3, draws, mask=mask)
    assert np.isnan(reps[1])
    assert reps[0] == 5.0 and reps[2] == 5.0


def test_bootstrap_player_record_is_internally_consistent():
    rng = np.random.default_rng(4)
    n = 800
    sub = pd.DataFrame(
        {
            "match_id": np.repeat(np.arange(40), 20),
            "li": np.abs(rng.normal(1.0, 0.8, n)),
            "wpa_bat_dd": rng.normal(0.001, 0.02, n),
        }
    )
    rec = bootstrap_player(sub, "wpa_bat_dd", np.random.default_rng(SEED))
    assert rec["balls"] == n and rec["matches"] == 40
    assert rec["li_lo"] < rec["mean_li"] < rec["li_hi"]
    assert rec["total_lo"] < rec["total_dd"] < rec["total_hi"]
    assert len(rec["reps_li"]) == N_BOOT
    if rec["reps_split"] is not None:
        assert rec["split_lo"] < rec["split"] < rec["split_hi"]


# --- data-gated ---------------------------------------------------------------
@data
def test_point_estimates_agree_with_the_wpa_module_on_real_data():
    from src.wpa import HIGH_LI, per_ball_wpa

    df = pd.read_parquet(config.BALLS_PARQUET)
    b = per_ball_wpa(df)
    name = b["batter"].value_counts().idxmax()  # most-faced batter: stable target
    sub = b[b["batter"] == name]
    rec = bootstrap_player(sub, "wpa_bat_dd", np.random.default_rng(SEED))
    assert np.isclose(rec["mean_li"], sub["li"].mean())
    assert np.isclose(rec["total_dd"], sub["wpa_bat_dd"].sum())
    hi = sub[sub["li"] >= HIGH_LI]
    if rec["reps_split"] is not None:
        assert np.isclose(rec["wpa_high"], hi["wpa_bat_dd"].mean())


@data
def test_same_seed_reproduces_bit_identically_on_real_data():
    from src.wpa import per_ball_wpa

    df = pd.read_parquet(config.BALLS_PARQUET)
    b = per_ball_wpa(df)
    name = b["batter"].value_counts().idxmax()
    sub = b[b["batter"] == name]
    r1 = bootstrap_player(sub, "wpa_bat_dd", np.random.default_rng(SEED))
    r2 = bootstrap_player(sub, "wpa_bat_dd", np.random.default_rng(SEED))
    assert np.array_equal(r1["reps_li"], r2["reps_li"])
    assert np.array_equal(r1["reps_total"], r2["reps_total"])
    assert r1["li_lo"] == r2["li_lo"] and r1["total_hi"] == r2["total_hi"]
