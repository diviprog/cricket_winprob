"""Constructive test of the calibration diagnosis: does honouring ball-to-ball
correlation close the WP calibration gap?

`tail_diagnostics.py` argued *by elimination* that the gap is ball correlation:
the one-ball marginals match empirical, yet the independence-assuming backward
induction misses E[y|b,w,r]. This module makes that constructive, in two parts.

PART A -- complete the elimination.
  tail_diagnostics only checked the boundary (4,6) and wicket marginals. Here we
  check the FULL one-ball distribution over {0..6,W} at every RRR slice via total
  variation distance model-vs-empirical. If TV ~ 0 everywhere, "the marginals are
  correct" is airtight, so any WP gap must be dependence, not marginals.

PART B -- constructive correlation, marginals held fixed.
  A forward innings simulator with a tunable dependence knob:
    * mode `model_iid` -- every ball drawn independently from the fitted outcome
      model p_o(state). This MUST reproduce the Markov DP WP (up to MC noise); it
      validates the simulator's mechanics and is the independence reference.
    * mode `block K` -- resample REAL consecutive outcome runs of length K,
      matched to the sim's current (rrr_bin, w, phase) cell at the block start.
      K=1 is an independent draw from the empirical cell marginal (~ model_iid,
      since marginals match). K>1 splices real consecutive balls, injecting the
      empirical ball-to-ball dependence WITHOUT changing the per-ball marginals.
    * mode `scattered K` -- DESIGNED as a heterogeneity-only control (donor
      innings' latent without local adjacency), but the results showed it is
      DEGENERATE and it must not be read as one: re-matching every ball to the
      sim's evolving cell within one donor turns the arm into a nearest-
      neighbour replay of empirical outcomes -- an E[y|state] estimator in
      disguise. Diagnostics: it closes MORE of the gap than whole real
      trajectories do (a >100% "share", self-refuting for heterogeneity), and it
      breaks its own marginal control (realized-marginal TV ~5-10x the block
      modes'). It is retained in the report as a negative methodological result;
      the mechanism question is answered by dependence_decomposition.md, which
      was built for it.

  Because every draw is matched to the current cell, all modes share the same
  one-ball marginals (verified via the realized-marginal fidelity check); the
  ONLY thing that varies is how much and what KIND of real dependence is
  carried. So the gap shrinking under consecutive blocks but not scattered ones
  is dependence -- specifically its sequential component -- closing the gap
  constructively.

  Every mode is replicated across SEEDS (the seed drives both the evaluated
  state sample and the MC draws), and the saturation claims are gated on the
  cross-seed spread rather than a single run.

Run:  .venv/bin/python -m src.correlation_experiment
Writes reports/correlation_experiment.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .outcome_model import RRR_BIN_EDGES, make_outcome_model, fit_rrr
from .wp_markov import solve_wp

OUT = config.OUTCOMES              # ["0".."6","W"]
RUNS = np.array([0, 1, 2, 3, 4, 5, 6, 0])  # run value per outcome index; W(idx7)=0 runs
W_IDX = 7
FINE_RRR_EDGES = [0, 6, 8, 10, 12, 14, 16, 18, 22, 40]
BLOCK_LENGTHS = [1, 3, 8, 20, 40, 80, 120]  # to saturation (120 = whole-innings replay)
SCATTER_K = 20             # donor-innings length for the scattered (mechanism) arm
N_SIMS = 400
PER_SLICE = 220            # evaluated real states sampled per RRR slice
SEED = 0
SEEDS = range(6)           # replication seeds; each drives sample AND MC draws
MIN_CELL = 150


# ---------------------------------------------------------------------------
# cell coding (must mirror outcome_model: (rrr_bin, w, phase))
# ---------------------------------------------------------------------------
def _phase_idx(b: np.ndarray) -> np.ndarray:
    over = (config.BALLS_PER_INNINGS - b) // 6
    return np.where(over < 6, 0, np.where(over < 15, 1, 2))


def _cell_id(b: np.ndarray, w: np.ndarray, r: np.ndarray) -> np.ndarray:
    rrr = 6.0 * r / np.maximum(b, 1)
    rb = np.searchsorted(RRR_BIN_EDGES, rrr, side="right")
    ph = _phase_idx(b)
    return (rb * 11 + w) * 3 + ph


def _rrr_slice(rrr: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(FINE_RRR_EDGES, rrr, side="right") - 1,
                  0, len(FINE_RRR_EDGES) - 2)


# ---------------------------------------------------------------------------
# PART A -- full one-ball marginal (total variation by RRR)
# ---------------------------------------------------------------------------
def full_marginal_tv(train: pd.DataFrame, dist) -> pd.DataFrame:
    d = train.copy()
    states = list(zip(d["b"], d["w"], d["r"]))
    model = np.array([[dd[o] for o in OUT] for dd in (dist(s) for s in states)])
    oc = d["outcome"].map({o: i for i, o in enumerate(OUT)}).to_numpy()
    emp1h = np.eye(len(OUT))[oc]
    d["_sl"] = _rrr_slice((6.0 * d["r"] / d["b"]).to_numpy())
    rows = []
    for sl, idx in d.groupby("_sl").groups.items():
        m = d.index.isin(idx)
        mp = model[m].mean(0)
        ep = emp1h[m].mean(0)
        tv = 0.5 * np.abs(mp - ep).sum()
        rows.append({"rrr_slice": _slice_label(sl), "n": int(m.sum()),
                     "tv": tv, "max_abs_dev": float(np.abs(mp - ep).max())})
    return pd.DataFrame(rows)


def _slice_label(i: int) -> str:
    return f"{FINE_RRR_EDGES[i]}-{FINE_RRR_EDGES[i + 1]}"


# ---------------------------------------------------------------------------
# PART B -- forward simulator
# ---------------------------------------------------------------------------
class BlockSampler:
    """Real train innings as one flat outcome-code array with innings bounds and a
    cell -> positions index, for matched block resampling."""

    def __init__(self, train: pd.DataFrame):
        t = train.sort_values(["match_id", "ball_index"])
        code = t["outcome"].map({o: i for i, o in enumerate(OUT)}).to_numpy()
        self.O = code.astype(np.int64)
        # innings stop (exclusive) and innings ordinal per global position
        stop = np.empty(len(t), dtype=np.int64)
        inn_of = np.empty(len(t), dtype=np.int64)
        pos, k = 0, 0
        for _, grp in t.groupby("match_id", sort=False):
            n = len(grp)
            stop[pos:pos + n] = pos + n
            inn_of[pos:pos + n] = k
            pos += n
            k += 1
        self.stop = stop
        self.inn_of = inn_of
        self.cell = _cell_id(t["b"].to_numpy(), t["w"].to_numpy(), t["r"].to_numpy())
        order = np.argsort(self.cell, kind="stable")
        self._order = order
        cells_sorted = self.cell[order]
        uniq, first = np.unique(cells_sorted, return_index=True)
        bounds = np.append(first, len(order))
        self.cell_pos = {int(c): order[bounds[i]:bounds[i + 1]]
                         for i, c in enumerate(uniq)}
        # (innings, cell) -> positions index, for the scattered mode. Cell ids are
        # < 1000 ((rb*11+w)*3+ph <= 197), so combo = inn*1000 + cell is unique.
        combo = inn_of * 1000 + self.cell
        ic_order = np.argsort(combo, kind="stable")
        self.ic_positions = ic_order
        cs = combo[ic_order]
        ic_uniq, ic_first = np.unique(cs, return_index=True)
        ic_bounds = np.append(ic_first, len(ic_order))
        self.ic_ptr = {int(c): (int(ic_first[i]), int(ic_bounds[i + 1] - ic_first[i]))
                       for i, c in enumerate(ic_uniq)}

    def draw(self, cell_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """A random real start position for each requested cell (grouped)."""
        out = np.empty(len(cell_ids), dtype=np.int64)
        for c in np.unique(cell_ids):
            mask = cell_ids == c
            pool = self.cell_pos.get(int(c))
            if pool is None:                       # unseen cell: nearest by |cell-c|
                keys = np.fromiter(self.cell_pos.keys(), dtype=np.int64)
                pool = self.cell_pos[int(keys[np.argmin(np.abs(keys - c))])]
            out[mask] = rng.choice(pool, size=int(mask.sum()))
        return out


def _terminal(b, w, r):
    """Vectorized terminal value; nan where live. Precedence: r<=0 win first."""
    val = np.full(b.shape, np.nan)
    val = np.where(r <= 0, 1.0, val)
    live = np.isnan(val)
    val = np.where(live & (w == 0), 0.0, val)
    live = np.isnan(val)
    val = np.where(live & (b == 0), np.where(r == 1, config.TIE_VALUE, 0.0), val)
    return val


def _sim_state(b0, w0, r0, mode, K, N, sampler, dist, cellprob, rng, acc=None, fb=None):
    """N Monte-Carlo continuations from one start state; return mean terminal WP.

    `cellprob` is a shared lazy cache of cell_id -> model prob vector, filled from
    `dist` on demand (the fitted model's fallback chain covers any reachable cell,
    including odd states real chases never visit). If `acc` (len-8 int array) is
    given, tallies every consumed outcome, for the realized-marginal fidelity check.
    In `scattered` mode, `fb` (len-2 int array) tallies [fallback draws, all draws]
    -- draws where the donor innings lacked the needed cell and the global pool
    substituted.
    """
    b = np.full(N, b0, dtype=np.int64)
    w = np.full(N, w0, dtype=np.int64)
    r = np.full(N, r0, dtype=np.int64)
    done = _terminal(b, w, r)
    alive = np.isnan(done)
    if mode == "block":
        cur = np.zeros(N, dtype=np.int64)
        left = np.zeros(N, dtype=np.int64)  # 0 => needs resample
    elif mode == "scattered":
        donor = np.zeros(N, dtype=np.int64)
        left = np.zeros(N, dtype=np.int64)  # 0 => pick a new donor innings
    for _ in range(config.BALLS_PER_INNINGS + 1):
        if not alive.any():
            break
        idx = np.where(alive)[0]
        if mode == "model_iid":
            cells = _cell_id(b[idx], w[idx], r[idx])
            oc = np.empty(len(idx), dtype=np.int64)
            for c in np.unique(cells):
                m = cells == c
                p = cellprob.get(int(c))
                if p is None:
                    j = idx[m][0]
                    pv = dist((int(b[j]), int(w[j]), int(r[j])))
                    p = np.array([pv[o] for o in OUT], dtype=float)
                    p /= p.sum()
                    cellprob[int(c)] = p
                oc[m] = rng.choice(len(OUT), size=int(m.sum()), p=p)
        elif mode == "block":
            need = idx[left[idx] == 0]
            if len(need):
                cells = _cell_id(b[need], w[need], r[need])
                start = sampler.draw(cells, rng)
                cur[need] = start
                left[need] = np.minimum(K, sampler.stop[start] - start)
            oc = sampler.O[cur[idx]]
            cur[idx] += 1
            left[idx] -= 1
        else:  # scattered: donor innings' latent without local adjacency
            need = idx[left[idx] == 0]
            if len(need):
                cells0 = _cell_id(b[need], w[need], r[need])
                start = sampler.draw(cells0, rng)  # donor = innings of a matched ball
                donor[need] = sampler.inn_of[start]
                left[need] = K
            cells = _cell_id(b[idx], w[idx], r[idx])
            combos = donor[idx] * 1000 + cells
            starts = np.zeros(len(idx), dtype=np.int64)
            lens = np.zeros(len(idx), dtype=np.int64)
            miss = []
            for j, cb in enumerate(combos):
                p = sampler.ic_ptr.get(int(cb))
                if p is None:
                    miss.append(j)
                else:
                    starts[j], lens[j] = p
            pick = starts + (rng.random(len(idx)) * lens).astype(np.int64)
            oc = sampler.O[sampler.ic_positions[pick]]
            if miss:
                mj = np.asarray(miss, dtype=np.int64)
                pos = sampler.draw(cells[mj], rng)  # global pool, nearest-cell fallback
                oc[mj] = sampler.O[pos]
            if fb is not None:
                fb[0] += len(miss)
                fb[1] += len(idx)
            left[idx] -= 1
        if acc is not None:
            np.add.at(acc, oc, 1)
        # apply outcome
        is_w = oc == W_IDX
        b[idx] -= 1
        w[idx] -= is_w.astype(np.int64)
        r[idx] -= RUNS[oc]
        tv = _terminal(b[idx], w[idx], r[idx])
        just = ~np.isnan(tv)
        done[idx[just]] = tv[just]
        alive[idx[just]] = False
    # any still "alive" after 120 balls are out of balls -> resolve as terminal
    rem = np.isnan(done)
    if rem.any():
        done[rem] = np.nan_to_num(_terminal(np.zeros(rem.sum(), dtype=np.int64),
                                            w[rem], r[rem]))
    return float(done.mean())


def run_block_experiment(
    train: pd.DataFrame, dist, wp, seed: int = SEED, sampler=None, cellprob=None,
    block_lengths=None, per_slice: int = PER_SLICE, n_sims: int = N_SIMS,
    scattered_k: int | None = SCATTER_K,
) -> tuple[pd.DataFrame, dict, float]:
    """One full replicate. `seed` drives BOTH the stratified state sample and the
    MC draws, so replicate spread covers both noise sources. Returns the per-slice
    table, the realized per-ball outcome distribution per mode (marginal-fidelity
    check), and the scattered arm's fallback rate."""
    rng = np.random.default_rng(seed)
    sampler = sampler if sampler is not None else BlockSampler(train)
    # lazy cache, filled from dist on demand; shareable across seeds (deterministic)
    cellprob = cellprob if cellprob is not None else {}
    block_lengths = list(BLOCK_LENGTHS if block_lengths is None else block_lengths)

    # Exact references from the FULL train split (model-free / table lookup, no MC
    # noise): empirical win rate and Markov WP per RRR slice.
    d = train.copy()
    d["_sl"] = _rrr_slice((6.0 * d["r"] / d["b"]).to_numpy())
    d["_mk"] = wp.predict(d["b"].to_numpy(), d["w"].to_numpy(), d["r"].to_numpy())
    full = d.groupby("_sl").agg(n_full=("y", "size"), emp=("y", "mean"),
                                markov=("_mk", "mean"))

    # stratified subsample for the (expensive) simulated modes
    picks = pd.concat([g.sample(min(per_slice, len(g)), random_state=seed)
                       for _, g in d.groupby("_sl")])
    states = list(zip(picks["b"].astype(int), picks["w"].astype(int), picks["r"].astype(int)))
    sl = picks["_sl"].to_numpy()

    modes = ["model_iid"] + [f"block_K{K}" for K in block_lengths]
    if scattered_k:
        modes.append(f"scattered_K{scattered_k}")
    cols = {m: np.empty(len(states)) for m in modes}
    accs = {m: np.zeros(len(OUT), dtype=np.int64) for m in modes}
    fb = np.zeros(2, dtype=np.int64)
    for i, (b, w, r) in enumerate(states):
        cols["model_iid"][i] = _sim_state(b, w, r, "model_iid", 0, n_sims, sampler,
                                          dist, cellprob, rng, accs["model_iid"])
        for K in block_lengths:
            cols[f"block_K{K}"][i] = _sim_state(b, w, r, "block", K, n_sims, sampler,
                                                dist, cellprob, rng, accs[f"block_K{K}"])
        if scattered_k:
            name = f"scattered_K{scattered_k}"
            cols[name][i] = _sim_state(b, w, r, "scattered", scattered_k, n_sims,
                                       sampler, dist, cellprob, rng, accs[name], fb)
    # realized per-ball outcome distribution each mode actually experienced
    realized = {m: a / a.sum() for m, a in accs.items()}
    fb_rate = float(fb[0] / fb[1]) if fb[1] else float("nan")

    agg = {"rrr_slice": [], "n": [], "emp": [], "markov": []}
    for name in cols:
        agg[name] = []
    for s in sorted(set(sl)):
        m = sl == s
        if int(full.loc[s, "n_full"]) < MIN_CELL:
            continue
        agg["rrr_slice"].append(_slice_label(int(s)))
        agg["n"].append(int(m.sum()))
        agg["emp"].append(float(full.loc[s, "emp"]))
        agg["markov"].append(float(full.loc[s, "markov"]))
        for name in cols:
            agg[name].append(float(cols[name][m].mean()))
    return pd.DataFrame(agg), realized, fb_rate


# ---------------------------------------------------------------------------
def _fmt(df: pd.DataFrame, floats: list[str], nd: int = 4) -> str:
    d = df.copy()
    for c in floats:
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:.{nd}f}")
    cols = list(d.columns)
    L = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        L.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(L)


def _mads(block: pd.DataFrame, modes: list[str]) -> dict[str, float]:
    """Mean absolute calibration gap per mode for one replicate's slice table."""
    return {m: float((block[m] - block["emp"]).abs().mean()) for m in modes}


def main() -> int:
    df = pd.read_parquet(config.BALLS_PARQUET)
    train = df[df["split"] == "train"].copy()
    r_max = int(df["r"].max()) + 1
    # Plain (non-era) model throughout: block resampling is flat empirical, so the
    # independence reference must share those flat marginals to isolate the
    # dependence effect. Era adjustment is an orthogonal level correction and
    # would only confound the marginal match (see report note).
    params = fit_rrr(train)
    dist = make_outcome_model("rrr", params)
    wp = solve_wp(dist, r_max=r_max)

    tv = full_marginal_tv(train, dist)
    max_tv = float(tv["tv"].max())

    # --- replication: every mode at every seed ------------------------------
    sampler = BlockSampler(train)
    cellprob: dict[int, np.ndarray] = {}
    scat = f"scattered_K{SCATTER_K}"
    modes = ["markov", "model_iid"] + [f"block_K{K}" for K in BLOCK_LENGTHS] + [scat]
    per_seed: dict[str, list[float]] = {m: [] for m in modes}
    fid_seed: dict[str, list[float]] = {}
    fb_rates: list[float] = []
    block0 = None
    for s in SEEDS:
        block, realized, fb_rate = run_block_experiment(
            train, dist, wp, seed=s, sampler=sampler, cellprob=cellprob)
        if block0 is None:
            block0 = block
        for m, v in _mads(block, modes).items():
            per_seed[m].append(v)
        fb_rates.append(fb_rate)
        base = realized["block_K1"]
        for m in realized:
            fid_seed.setdefault(m, []).append(
                float(0.5 * np.abs(realized[m] - base).sum()))

    mean_mad = {m: float(np.mean(per_seed[m])) for m in modes}
    lo_mad = {m: float(np.min(per_seed[m])) for m in modes}
    hi_mad = {m: float(np.max(per_seed[m])) for m in modes}
    max_fid = {m: float(np.max(v)) for m, v in fid_seed.items()}
    fb_rate = float(np.mean(fb_rates))
    n_seeds = len(list(SEEDS))

    # --- claim gates (pre-committed in the plan) -----------------------------
    kstar = min(BLOCK_LENGTHS, key=lambda K: mean_mad[f"block_K{K}"])
    closed = 100 * (mean_mad["block_K1"] - mean_mad[f"block_K{kstar}"]) / mean_mad["block_K1"]
    # gate 1: the K=20 < K=40 ordering holds in a majority of seeds
    n_order = sum(a < b for a, b in zip(per_seed["block_K20"], per_seed["block_K40"]))
    gate_order = n_order > n_seeds / 2
    # gate 2: the minimum is separated from K=1 beyond replicate spread
    gate_sep = mean_mad[f"block_K{kstar}"] < min(per_seed["block_K1"])
    saturation_resolved = gate_order and gate_sep
    # scattered arm: share of the consecutive-block closure it reproduces
    scat_share = ((mean_mad["block_K1"] - mean_mad[scat])
                  / max(mean_mad["block_K1"] - mean_mad[f"block_K{kstar}"], 1e-12))

    L = ["# Does honouring ball-to-ball dependence close the calibration gap?", "",
         f"Train split, {len(train):,} balls. Constructive test of the "
         f"tail_diagnostics conclusion. Uses the PLAIN (non-era) RRR Markov model so "
         f"every mode shares the same flat empirical marginals as the block "
         f"resampler -- era adjustment is an orthogonal level correction that would "
         f"otherwise confound the marginal match. {N_SIMS} sims/state, "
         f"{PER_SLICE}/slice, replicated over {n_seeds} seeds (each seed redraws "
         f"both the evaluated states and the MC paths).", ""]

    L += ["## Part A -- the full one-ball marginal matches empirical", "",
          "Total variation distance between the model's per-ball distribution over "
          "{0..6,W} and empirical, by RRR slice. Completes the elimination "
          "(tail_diagnostics only checked boundary + wicket).", "",
          _fmt(tv, ["tv", "max_abs_dev"]), "",
          f"Max TV across slices = **{max_tv:.4f}** (0 = identical distributions). The "
          f"full marginal is well estimated everywhere, so a WP gap cannot be a "
          f"marginal error.", ""]

    detail_modes = ["model_iid"] + [f"block_K{K}" for K in BLOCK_LENGTHS] + [scat]
    L += ["## Part B -- gap vs dependence (block length K), seed-0 detail", "",
          "Mean WP by mode vs empirical win rate, by RRR slice. `model_iid` draws "
          "each ball independently from the fitted model (independence reference, "
          "should track `markov`). `block_Kk` resamples real k-ball consecutive runs "
          "matched to the current cell -- same marginals, real dependence. "
          f"`{scat}` was designed as a heterogeneity-only control (cell-matched "
          "draws from random positions of one donor innings) but turned out "
          "DEGENERATE -- see the reading below; do not interpret it as a "
          "mechanism probe.", "",
          _fmt(block0[["rrr_slice", "n", "emp", "markov"] + detail_modes],
               ["emp", "markov"] + detail_modes), ""]

    L += ["## Replicated mean |gap| by mode (6 seeds; lower = better calibrated)", "",
          "`marginal_TV_vs_K1` = max across seeds of the total variation between the "
          "mode's realized per-ball outcome distribution and the independent K=1 "
          "mode's -- near 0 means the mode adds dependence WITHOUT changing the "
          "marginals, so gap movement is attributable to dependence alone.", "",
          "| mode | mean |gap| | min..max over seeds | marginal_TV_vs_K1 |",
          "|---|---|---|---|"]
    for m in modes:
        fid = f"{max_fid[m]:.4f}" if m in max_fid else "--"
        L.append(f"| {m} | {mean_mad[m]:.4f} | {lo_mad[m]:.4f}..{hi_mad[m]:.4f} | {fid} |")
    L += ["", f"Scattered-arm fallback rate (donor innings lacked the needed cell, "
          f"global pool substituted): **{100 * fb_rate:.1f}%** of draws.", ""]

    # --- conclusions, gated on the replicate spread --------------------------
    sat_txt = (
        f"The minimum at **K={kstar}** is stable across seeds: the K=20 < K=40 "
        f"ordering holds in {n_order}/{n_seeds} seeds, and the K={kstar} mean sits "
        f"below every seed's K=1 value. The saturation scale is consistent with the "
        f"**~3-5-ball sequential correlation range** measured in "
        f"dependence_decomposition.md: a block's extra cumulative variance grows "
        f"with K only until K is several times the correlation range, so a ~5-ball "
        f"range saturates near K~20."
        if saturation_resolved else
        f"The location of the minimum is NOT resolved beyond 'somewhere in the "
        f"8-40 range': the K=20 < K=40 ordering holds in only {n_order}/{n_seeds} "
        f"seeds. The reduction from K=1 itself is robust; the fine shape of the "
        f"curve is within replicate noise."
    )
    fid_blocks = max(v for m, v in max_fid.items() if m.startswith("block_K"))
    if scat_share < 0.4:
        scat_txt = (
            f"It reproduces only **{100 * scat_share:.0f}%** of the consecutive-block "
            f"closure, corroborating the decomposition: the gap is closed by LOCAL "
            f"sequential structure, not by innings-level heterogeneity.")
    elif scat_share <= 0.8:
        scat_txt = (
            f"It reproduces **{100 * scat_share:.0f}%** of the consecutive-block "
            f"closure -- a substantial innings-latent contribution, in tension with "
            f"the decomposition's 18% heterogeneity share; treat the mechanism split "
            f"as unresolved between the two probes.")
    else:
        scat_txt = (
            f"It closed **{100 * scat_share:.0f}%** of the consecutive-block closure "
            f"-- MORE than replaying whole real trajectories (mean |gap| "
            f"{mean_mad[scat]:.4f} vs block_K120 {mean_mad['block_K120']:.4f}), which "
            f"no heterogeneity injection could do, and it broke its own marginal "
            f"control (realized-marginal TV {max_fid[scat]:.4f} vs <= "
            f"{fid_blocks:.4f} for every block mode). The arm is DEGENERATE as "
            f"designed: re-matching every ball to the sim's evolving cell within one "
            f"donor makes it a nearest-neighbour replay of empirical outcomes -- an "
            f"E[y|state] estimator in disguise, not a dependence injection. It is "
            f"reported as a negative methodological result; the mechanism question "
            f"is settled by dependence_decomposition.md (sequential, ~18% "
            f"heterogeneity), whose permutation-null design does not have this "
            f"failure mode.")
    L += ["## Reading, with the cross-seed spread", "",
          f"Independence references agree at every seed: markov "
          f"{mean_mad['markov']:.4f} ~ model_iid {mean_mad['model_iid']:.4f} ~ "
          f"block_K1 {mean_mad['block_K1']:.4f} (validates the simulator and the "
          f"flat-marginal match). Injecting real dependence shrinks the mean |gap| "
          f"to **{mean_mad[f'block_K{kstar}']:.4f} at K={kstar}** -- a "
          f"**{closed:.0f}% reduction** versus independence, with marginal fidelity "
          f"<= TV {fid_blocks:.4f} across all block modes and seeds. The gap "
          f"closes **constructively**.", "",
          "Observations:",
          "- **It closes in both directions**, as over-dispersion predicts: "
          "dependence adds outcome variance, pulling WP toward 0.5 -- UP in hard "
          "chases (RRR>=8, where the model was too low) and DOWN in easy ones "
          "(RRR 0-6, where it was too high). Per dependence_decomposition.md the "
          "dependence is run-scoring persistence (wickets ANTI-cluster), whose "
          "burst/drought symmetry is exactly a two-sided variance effect.",
          f"- **Saturation.** {sat_txt}",
          f"- **The scattered arm.** {scat_txt}",
          f"- **{closed:.0f}% is a lower bound, and the block bootstrap cannot "
          f"measure more.** Even whole-innings replays (K={BLOCK_LENGTHS[-1]}) "
          f"splice segments that diverge from the sim's evolving state and break "
          f"dependence at block starts, so they extract no further clean signal. "
          f"Pinning down the full contribution needs a generative dependent model "
          f"(e.g. Markov-switching outcomes) -- the natural next step, and the one "
          f"that would forfeit the exact martingale.", ""]

    out = config.REPORTS / "correlation_experiment.md"
    out.write_text("\n".join(L))
    print(f"Part A: max one-ball marginal TV = {max_tv:.4f}")
    print(f"Part B replicated mean |gap| ({n_seeds} seeds):")
    for m in modes:
        print(f"  {m:16s} {mean_mad[m]:.4f}  [{lo_mad[m]:.4f}..{hi_mad[m]:.4f}]")
    print(f"kstar={kstar}  closed={closed:.0f}%  saturation_resolved={saturation_resolved}")
    print(f"scattered share of closure={100 * scat_share:.0f}%  fallback={100 * fb_rate:.1f}%")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
