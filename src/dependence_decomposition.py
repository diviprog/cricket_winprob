"""Decompose the model's residual dependence: serial correlation vs shared latents.

WHY THIS MODULE EXISTS
----------------------
correlation_experiment.py showed constructively that honouring real ball-to-ball
dependence closes ~28% of the WP calibration gap. But "dependence relative to the
independent model" has two very different mechanisms with IDENTICAL signatures in
every diagnostic so far:

  (a) genuine SERIAL correlation -- outcome_t depends on recent outcomes
      (momentum, a batsman "getting set" ball by ball);
  (b) shared LATENT heterogeneity -- all balls of an innings (pitch, teams,
      target conditions) or of a partnership (who is batting) share a hidden
      level that (b,w,r) does not see.

Both inflate trajectory dispersion, both leave the one-ball marginals intact,
and both are carried by consecutive-ball blocks in the bootstrap. The paper's
mechanism claim ("scoring bursts and wicket clusters") assumes (a); the M4 null
on `striker_balls` hints at (b). This module separates them.

HOW THE SEPARATION WORKS
------------------------
Residual = outcome value minus its (phase, w) state mean (same construction as
`leverage.partnership_lag1_autocorr`), on the train split, for BOTH runs-per-ball
and the wicket indicator.

1. LAG PROFILE. Within-innings autocorrelation of residuals at lags 1..30.
   A shared innings latent contributes EQUALLY at every lag (any two balls of the
   innings share it), so it produces a FLAT profile; serial correlation decays
   with lag. Shape is the first discriminator.

2. PERMUTATION NULL (the exact test). Shuffle residual values WITHIN each innings
   (positions, partnerships, and innings composition unchanged), recompute every
   statistic, 200 reps. The key property: within-innings permutation PRESERVES
   the innings-level shared-latent contribution to the autocorrelation (any two
   balls of the same innings still share the innings mean), while destroying all
   ordering and all partnership-level structure. So the null band is centred at
   the innings-composition component, NOT at zero, and

       observed above the null band  <=>  dependence BEYOND innings-level
                                          heterogeneity exists.

   The null also absorbs the mechanical negative bias that demeaning induces
   (~ -1/(n_group-1)), so the demeaned variants need no analytic correction.

3. DEMEANING LADDER. Each profile is computed for three residual variants:
   state-demeaned (as in M3), + innings-demeaned, + partnership-demeaned. Where
   the signal collapses locates the level that carries it: if the partnership-
   demeaned variant falls inside its null band while the innings-demeaned one
   does not, the dependence is a partnership-scale latent ("set batsman" as a
   slowly-varying hidden state), not ball-adjacent momentum.

4. ICC. One-way ANOVA intraclass correlation of the state-demeaned residuals at
   the innings and partnership level -- the variance-share reading of the same
   question, and a cross-check: the flat null-band centre of (2) should sit near
   the innings ICC.

Verdict logic (mirrors the interpretation matrix in the report) is computed
programmatically so the report states whatever the data shows.

Run:  .venv/bin/python -m src.dependence_decomposition
Writes reports/dependence_decomposition.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

LAGS = [1, 2, 3, 5, 10, 20, 30]
PART_LAGS = [1, 2, 3, 5, 10]  # partnerships are short; long lags have few pairs
N_PERM = 200
SEED = 0
VARIANTS = ["state", "state+innings", "state+partnership"]


# ---------------------------------------------------------------------------
# array-level primitives (unit-tested directly)
# ---------------------------------------------------------------------------
def demean_by(x: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """x minus its group mean (codes must be non-negative ints)."""
    cnt = np.bincount(codes)
    s = np.bincount(codes, weights=x)
    means = s / np.maximum(cnt, 1)
    return x - means[codes]


def lag_corr(x: np.ndarray, scope: np.ndarray, lag: int) -> tuple[float, int]:
    """Pooled Pearson correlation of (x_t, x_{t+lag}) over pairs that stay within
    the same `scope` group (arrays must be sorted so groups are contiguous)."""
    a, b = x[:-lag], x[lag:]
    m = scope[:-lag] == scope[lag:]
    a, b = a[m], b[m]
    n = len(a)
    if n < 30:
        return float("nan"), n
    va, vb = a - a.mean(), b - b.mean()
    denom = np.sqrt((va**2).sum() * (vb**2).sum())
    if denom == 0:
        return float("nan"), n
    return float((va * vb).sum() / denom), n


def icc_oneway(x: np.ndarray, codes: np.ndarray) -> float:
    """One-way ANOVA ICC(1) for unbalanced groups: the share of variance carried
    by the group level, correcting the between-group mean variance for the
    within-group noise it inherits (E[MSB] = sigma_w^2 + n0 * sigma_b^2)."""
    cnt = np.bincount(codes).astype(float)
    live = cnt > 0
    k = int(live.sum())
    n_tot = float(len(x))
    means = np.bincount(codes, weights=x) / np.maximum(cnt, 1)
    grand = float(x.mean())
    ssb = float((cnt[live] * (means[live] - grand) ** 2).sum())
    ssw = float(((x - means[codes]) ** 2).sum())
    msb = ssb / (k - 1)
    msw = ssw / (n_tot - k)
    n0 = (n_tot - float((cnt[live] ** 2).sum()) / n_tot) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def within_group_shuffle(
    x: np.ndarray, sorted_codes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return x with values shuffled within each contiguous group (codes must be
    sorted). Positions keep their group; values move within it."""
    order = np.lexsort((rng.random(len(x)), sorted_codes))
    return x[order]


# ---------------------------------------------------------------------------
# data prep
# ---------------------------------------------------------------------------
def prep(train: pd.DataFrame) -> dict:
    """Sorted arrays: residuals (runs, wicket) + innings / partnership codes.

    Residual construction mirrors leverage.partnership_lag1_autocorr: value minus
    its (phase, w) mean, partnership id = wickets already fallen in the innings.
    """
    b = train.sort_values(["match_id", "ball_index"]).copy()
    b["_runs"] = b["outcome"].map(lambda o: 0 if o == "W" else int(o)).astype(float)
    b["_wkt"] = (b["outcome"] == "W").astype(float)
    for col in ["_runs", "_wkt"]:
        b[col + "_r"] = b[col] - b.groupby(["phase", "w"])[col].transform("mean")
    inn = pd.factorize(b["match_id"])[0]
    wfallen = (
        b.groupby("match_id")["outcome"]
        .transform(lambda s: (s == "W").cumsum().shift(1).fillna(0))
        .astype(int)
        .to_numpy()
    )
    part = pd.factorize(inn.astype(np.int64) * 16 + wfallen)[0]
    return {
        "runs": b["_runs_r"].to_numpy(),
        "wkt": b["_wkt_r"].to_numpy(),
        "inn": inn,
        "part": part,
    }


# ---------------------------------------------------------------------------
# profiles: observed + permutation null
# ---------------------------------------------------------------------------
def _variants(x: np.ndarray, inn: np.ndarray, part: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "state": x,
        "state+innings": demean_by(x, inn),
        "state+partnership": demean_by(x, part),
    }


def _profile_set(x, inn, part) -> dict:
    """All (variant, scope, lag) correlations for one residual series."""
    out = {}
    vs = _variants(x, inn, part)
    for vname, xv in vs.items():
        for lag in LAGS:
            out[(vname, "innings", lag)] = lag_corr(xv, inn, lag)[0]
        for lag in PART_LAGS:
            out[(vname, "partnership", lag)] = lag_corr(xv, part, lag)[0]
    return out


def observed_and_null(x, inn, part, n_perm=N_PERM, seed=SEED) -> tuple[dict, dict]:
    """Observed profile set plus, for every key, the permutation-null draws.

    Innings-demeaning commutes with within-innings shuffling (the innings mean is
    a set property), so the permuted innings-demeaned series is just the shuffle
    of the observed one. Partnership means must be recomputed per rep, because
    shuffled values cross partnership boundaries.
    """
    obs = _profile_set(x, inn, part)
    rng = np.random.default_rng(seed)
    null: dict = {k: np.empty(n_perm) for k in obs}
    for r in range(n_perm):
        xp = within_group_shuffle(x, inn, rng)
        prof = _profile_set(xp, inn, part)
        for k, v in prof.items():
            null[k][r] = v
    return obs, null


def band(draws: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.nanmean(draws)),
        float(np.nanpercentile(draws, 2.5)),
        float(np.nanpercentile(draws, 97.5)),
    )


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------
def verdict(obs: dict, null: dict, icc_inn: float, icc_part: float) -> dict:
    """Programmatic reading of the interpretation matrix, runs residuals.

    All keys are (variant, scope, lag). The three questions, in order:
      1. Is there dependence beyond innings-level composition at all?
         state-variant innings-scope lag-1 above its null band.
      2. Does it survive innings demeaning? (it must, if 1 is yes -- consistency)
      3. Is it carried by a partnership-level latent or by ball-adjacent order?
         partnership-demeaned lag-1 inside its band => partnership latent;
         above => genuine sequential dependence survives even within partnership.
    Plus the shape read: excess-over-null flat vs decaying across lags.
    """
    k1 = ("state", "innings", 1)
    m1, lo1, hi1 = band(null[k1])
    beyond_innings = obs[k1] > hi1
    het_share = float(np.clip(m1 / obs[k1], 0.0, 1.0)) if obs[k1] > 0 else float("nan")

    kd = ("state+innings", "innings", 1)
    md, lod, hid = band(null[kd])
    survives_inn_demean = obs[kd] > hid

    kp = ("state+partnership", "partnership", 1)
    mp, lop, hip = band(null[kp])
    survives_part_demean = obs[kp] > hip

    # shape: excess over null mean at each innings-scope lag, state variant
    excess = {
        lag: obs[("state", "innings", lag)] - band(null[("state", "innings", lag)])[0]
        for lag in LAGS
    }
    e1 = excess[1]
    tail = np.nanmean([excess[20], excess[30]])
    decays = bool(e1 > 0 and tail < 0.5 * e1)

    if not beyond_innings:
        label = "heterogeneity-only"
        text = (
            "All residual dependence is explained by innings-level composition "
            "(shared latent); no sequential structure beyond it."
        )
    elif survives_part_demean:
        label = "sequential"
        text = (
            "Genuine ball-adjacent sequential dependence survives even after "
            "removing partnership means; momentum-like correlation is real."
        )
    else:
        label = "partnership-latent"
        text = (
            "Dependence beyond innings composition exists but collapses once "
            "partnership means are removed: it is a partnership-scale shared "
            "latent (who is batting / how set they are), not ball-adjacent "
            "momentum."
        )
    return {
        "label": label,
        "text": text,
        "beyond_innings": beyond_innings,
        "survives_inn_demean": survives_inn_demean,
        "survives_part_demean": survives_part_demean,
        "het_share_lag1": het_share,
        "excess": excess,
        "decays": decays,
        "icc_inn": icc_inn,
        "icc_part": icc_part,
        "null_centre_lag1": m1,
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _table(obs: dict, null: dict, variant: str, scope: str, lags: list[int]) -> str:
    rows = [
        "| lag | observed | null mean | null 2.5% | null 97.5% | above band |",
        "|---|---|---|---|---|---|",
    ]
    for lag in lags:
        k = (variant, scope, lag)
        m, lo, hi = band(null[k])
        flag = "**yes**" if obs[k] > hi else ("below" if obs[k] < lo else "no")
        rows.append(f"| {lag} | {obs[k]:+.4f} | {m:+.4f} | {lo:+.4f} | {hi:+.4f} | {flag} |")
    return "\n".join(rows)


def main() -> int:
    df = pd.read_parquet(config.BALLS_PARQUET)
    train = df[df["split"] == "train"].copy()
    d = prep(train)

    icc = {
        ("runs", "innings"): icc_oneway(d["runs"], d["inn"]),
        ("runs", "partnership"): icc_oneway(d["runs"], d["part"]),
        ("wkt", "innings"): icc_oneway(d["wkt"], d["inn"]),
        ("wkt", "partnership"): icc_oneway(d["wkt"], d["part"]),
    }

    obs_r, null_r = observed_and_null(d["runs"], d["inn"], d["part"])
    obs_w, null_w = observed_and_null(d["wkt"], d["inn"], d["part"])
    v = verdict(obs_r, null_r, icc[("runs", "innings")], icc[("runs", "partnership")])

    L = [
        "# Dependence decomposition: serial correlation vs shared latents",
        "",
        f"Train split, {len(train):,} balls, {N_PERM} within-innings permutations. "
        f"Residual = value minus its (phase, w) state mean. The permutation null "
        f"PRESERVES innings-level composition (shuffling within an innings keeps "
        f"the innings' shared latent in every pair) while destroying ordering and "
        f"partnership structure -- so `observed above the null band` means "
        f"dependence BEYOND innings-level heterogeneity, and the null centre "
        f"estimates the heterogeneity component itself.",
        "",
    ]

    L += [
        "## Runs residuals -- within-innings lag profile",
        "",
        "### state-demeaned (the M3 construction)",
        "",
        _table(obs_r, null_r, "state", "innings", LAGS),
        "",
        "### + innings-demeaned",
        "",
        _table(obs_r, null_r, "state+innings", "innings", LAGS),
        "",
        "### + partnership-demeaned, within-partnership pairs",
        "",
        _table(obs_r, null_r, "state+partnership", "partnership", PART_LAGS),
        "",
    ]

    k_headline = ("state", "partnership", 1)
    m_h, lo_h, hi_h = band(null_r[k_headline])
    L += [
        "## Continuity with the M3 headline number",
        "",
        f"Within-partnership lag-1 autocorrelation of state-demeaned runs "
        f"residuals (the construction behind the +0.037 in "
        f"leverage_validation.md, here on train only): "
        f"**{obs_r[k_headline]:+.4f}**, permutation null "
        f"[{lo_h:+.4f}, {hi_h:+.4f}] centred at {m_h:+.4f}.",
        "",
    ]

    L += [
        "## Wicket residuals -- within-innings lag profile",
        "",
        "### state-demeaned",
        "",
        _table(obs_w, null_w, "state", "innings", LAGS),
        "",
        "### + innings-demeaned",
        "",
        _table(obs_w, null_w, "state+innings", "innings", LAGS),
        "",
    ]

    L += [
        "## Variance decomposition (ICC)",
        "",
        "| residual | innings-level ICC | partnership-level ICC |",
        "|---|---|---|",
        f"| runs | {icc[('runs', 'innings')]:.4f} | {icc[('runs', 'partnership')]:.4f} |",
        f"| wicket | {icc[('wkt', 'innings')]:.4f} | {icc[('wkt', 'partnership')]:.4f} |",
        "",
        f"Cross-check: the state-variant null-band centre at lag 1 "
        f"({v['null_centre_lag1']:+.4f}) should sit near the innings ICC "
        f"({v['icc_inn']:+.4f}) -- both estimate the innings-composition "
        f"component.",
        "",
        "Caveat: the wicket *partnership* ICC is mechanically confounded and "
        "should not be read as a latent -- every partnership ends in exactly one "
        "wicket, so its per-partnership wicket rate is 1/length by construction "
        "and the group-mean variance is a length artifact.",
        "",
    ]

    L += [
        "## Interpretation matrix",
        "",
        "| observation | verdict |",
        "|---|---|",
        "| profile decays, survives innings demeaning | genuine serial correlation |",
        "| flat profile, collapses under innings demeaning, innings ICC > 0 | match-level heterogeneity |",
        "| collapses under partnership demeaning, partnership ICC >> innings ICC | partnership-scale latent |",
        "| mixed | report the shares |",
        "",
    ]

    exc = ", ".join(f"lag {k}: {x:+.4f}" for k, x in v["excess"].items())
    L += [
        "## Verdict (computed, runs residuals)",
        "",
        f"- dependence beyond innings composition at lag 1: "
        f"**{'yes' if v['beyond_innings'] else 'no'}**",
        f"- survives innings demeaning: **{'yes' if v['survives_inn_demean'] else 'no'}**",
        f"- survives partnership demeaning (within-partnership pairs): "
        f"**{'yes' if v['survives_part_demean'] else 'no'}**",
        f"- innings-composition share of the lag-1 signal: **{100 * v['het_share_lag1']:.0f}%**",
        f"- excess over null by lag ({'decaying' if v['decays'] else 'flat-ish'}): {exc}",
        "",
        f"**{v['label']}** -- {v['text']}",
        "",
    ]

    # wickets get their own (asymmetric) verdict: read the short-lag cells
    wk1 = obs_w[("state", "innings", 1)]
    _, wlo1, whi1 = band(null_w[("state", "innings", 1)])
    wk_anti = wk1 < wlo1
    L += [
        "## Verdict (wicket residuals)",
        "",
        (
            f"Short-lag wicket dependence is **{'NEGATIVE' if wk_anti else 'not detected'}**: "
            f"lag-1 observed {wk1:+.4f} vs null [{wlo1:+.4f}, {whi1:+.4f}]. "
            + (
                "Wickets ANTI-cluster ball-adjacent -- immediately after state-mean "
                "adjustment, a wicket makes another wicket in the next few balls LESS "
                "likely than independence predicts (consolidation). The 'wicket "
                "clusters' half of the original mechanism story is not supported; "
                "the dependence that closes the calibration gap is run-scoring "
                "persistence. Note positive run-rate persistence alone explains the "
                "two-sided gap closure: bursts make hard chases more winnable, and "
                "the mirror-image droughts make easy chases more losable."
                if wk_anti
                else "No short-range wicket clustering or anti-clustering is resolved; "
                "the dependence signal lives in run scoring."
            )
        ),
        "",
    ]

    out = config.REPORTS / "dependence_decomposition.md"
    out.write_text("\n".join(L))
    print(f"verdict: {v['label']}")
    print(
        f"  lag-1 state obs {obs_r[('state', 'innings', 1)]:+.4f} vs null centre "
        f"{v['null_centre_lag1']:+.4f} (het share {100 * v['het_share_lag1']:.0f}%)"
    )
    print(f"  innings ICC {v['icc_inn']:+.4f}  partnership ICC {v['icc_part']:+.4f}")
    print(
        f"  survives innings demeaning: {v['survives_inn_demean']}  "
        f"partnership demeaning: {v['survives_part_demean']}"
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
