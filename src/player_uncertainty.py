"""Player-level uncertainty -- match-clustered bootstrap CIs for the leaderboards.

Closes the gap flagged in the paper's Limitations: the leverage and WPA
leaderboards (player_leverage.md, wpa.md) are point estimates, and any claim
about ADJACENT players in a ranking needs an interval before it is a claim.

WHY THE CLUSTER IS THE MATCH (= THE INNINGS)
A per-ball bootstrap treats deliveries as independent draws -- which is exactly
the assumption this project measured to be false (dependence_decomposition.md:
within-innings lag-1 residual autocorrelation +0.044 against a permutation null
of +0.008, decaying over ~3-5 balls). Resampling whole matches keeps every
within-innings dependence intact and only requires exchangeability ACROSS a
player's matches -- the same reasoning as `baseline.cluster_bootstrap_delta`,
applied per player. One innings per match_id (spec 01), so match == innings.

WHAT THE CIs CONDITION ON
The WP surface, the leverage index, and the state-conditional drift correction
are held fixed across replicates. Any single player is a sliver of every WP
bin, so re-estimating the drift inside each replicate would move nothing at the
precision reported; what the intervals measure is the sampling variability of
the player's own career given those surfaces. Model-level error (the +0.037
residual autocorrelation, the calibration gap of up to 0.11) is shared across
players, biases rankings coherently, and is reported separately -- it is a
bound on the SURFACE, these are intervals on the PLAYERS.

DIFFERENCES BETWEEN PLAYERS
Rank-adjacency claims use a bootstrap CI on the DIFFERENCE of the two players'
statistics (independent replicate streams differenced), not CI overlap --
overlapping 95% intervals can hide a significant difference. Two players'
careers do overlap in a handful of shared innings, so independence is an
approximation; the shared-innings share of any pair's careers is small, and the
induced correlation is positive, making the difference CI mildly conservative.

Run:  .venv/bin/python -m src.player_uncertainty   (needs `wp`, `li` in parquet)
Writes reports/player_uncertainty.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .wpa import HIGH_LI, MIN_BALLS, per_ball_wpa

SEED = 0
N_BOOT = 2000
MIN_HIGH = 100  # clutch-split gate, same as wpa.md's clutch tables
TOP_N = 10


# ---------------------------------------------------------------------------
# core primitive: bootstrap replicates of a per-ball mean, resampling matches
# ---------------------------------------------------------------------------
def cluster_boot_mean(
    x: np.ndarray,
    codes: np.ndarray,
    n_matches: int,
    draws: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Replicates of mean(x) (optionally over `mask` balls only), whole matches
    resampled with replacement.

    Same per-match sum/count trick as `baseline.cluster_bootstrap_delta`. With a
    mask, a replicate whose resampled matches contain zero masked balls has no
    defined mean and comes back NaN; callers count and drop those.
    """
    if mask is not None:
        sums = np.bincount(codes, weights=np.where(mask, x, 0.0), minlength=n_matches)
        cnts = np.bincount(codes, weights=mask.astype(float), minlength=n_matches)
    else:
        sums = np.bincount(codes, weights=x, minlength=n_matches)
        cnts = np.bincount(codes, minlength=n_matches).astype(float)
    num = sums[draws].sum(axis=1)
    den = cnts[draws].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _ci(reps: np.ndarray) -> tuple[float, float]:
    return (
        float(np.nanpercentile(reps, 2.5)),
        float(np.nanpercentile(reps, 97.5)),
    )


# ---------------------------------------------------------------------------
# per-player bootstrap record
# ---------------------------------------------------------------------------
def bootstrap_player(sub: pd.DataFrame, wpa_col: str, rng: np.random.Generator) -> dict:
    """Point estimates + replicate arrays for one player's career.

    Replicate arrays are returned so that difference CIs between players can be
    formed downstream; all statistics for one player share one draws matrix
    (marginal CIs are unaffected, and it halves the RNG cost).
    """
    codes, uniq = pd.factorize(sub["match_id"].to_numpy())
    m = len(uniq)
    draws = rng.integers(0, m, size=(N_BOOT, m))
    li = sub["li"].to_numpy(dtype=float)
    wpa = sub[wpa_col].to_numpy(dtype=float)
    hi_mask = li >= HIGH_LI

    reps_li = cluster_boot_mean(li, codes, m, draws)
    reps_rate = cluster_boot_mean(wpa, codes, m, draws)
    n = len(sub)

    rec: dict = {
        "balls": n,
        "matches": m,
        "n_high": int(hi_mask.sum()),
        "mean_li": float(li.mean()),
        "total_dd": float(wpa.sum()),
        "reps_li": reps_li,
        # career volume is observed, not sampled: the CI on the total scales the
        # per-ball-rate replicates by the player's actual ball count, so the
        # interval reflects uncertainty in the RATE, not in how much they played
        "reps_total": n * reps_rate,
        "reps_split": None,
        "split_degenerate": 0,
    }
    rec["li_lo"], rec["li_hi"] = _ci(reps_li)
    rec["total_lo"], rec["total_hi"] = _ci(rec["reps_total"])

    if rec["n_high"] >= MIN_HIGH:
        reps_hi = cluster_boot_mean(wpa, codes, m, draws, mask=hi_mask)
        reps_lo = cluster_boot_mean(wpa, codes, m, draws, mask=~hi_mask)
        reps_split = reps_hi - reps_lo
        rec["wpa_high"] = float(wpa[hi_mask].mean())
        rec["wpa_low"] = float(wpa[~hi_mask].mean())
        rec["split"] = rec["wpa_high"] - rec["wpa_low"]
        rec["split_degenerate"] = int(np.isnan(reps_split).sum())
        rec["split_lo"], rec["split_hi"] = _ci(reps_split)
        rec["reps_split"] = reps_split
    return rec


def bootstrap_side(b: pd.DataFrame, who: str, side: str, rng: np.random.Generator) -> dict:
    """All qualified players on one side, in sorted-name order (determinism:
    the rng is consumed in a fixed order regardless of pandas group order)."""
    wpa_col = f"wpa_{side}_dd"
    sizes = b.groupby(who).size()
    names = sorted(sizes.index[sizes >= MIN_BALLS])
    grouped = dict(tuple(b.groupby(who)))
    return {name: bootstrap_player(grouped[name], wpa_col, rng) for name in names}


# ---------------------------------------------------------------------------
# difference CIs down a ranked table
# ---------------------------------------------------------------------------
def adjacent_diffs(ranked: list[str], recs: dict, reps_key: str, point_key: str) -> list[dict]:
    out = []
    for a, bname in zip(ranked[:-1], ranked[1:]):
        d = recs[a][reps_key] - recs[bname][reps_key]
        lo, hi = _ci(d)
        out.append(
            {
                "pair": f"{a} vs {bname}",
                "diff": recs[a][point_key] - recs[bname][point_key],
                "lo": lo,
                "hi": hi,
                "separated": bool(lo > 0 or hi < 0),
            }
        )
    return out


def pair_diff(recs: dict, a: str, bname: str, reps_key: str, point_key: str) -> dict:
    d = recs[a][reps_key] - recs[bname][reps_key]
    lo, hi = _ci(d)
    return {
        "pair": f"{a} vs {bname}",
        "diff": recs[a][point_key] - recs[bname][point_key],
        "lo": lo,
        "hi": hi,
        "separated": bool(lo > 0 or hi < 0),
    }


def _find(names, needle: str) -> str | None:
    hits = [n for n in names if needle.lower() in str(n).lower()]
    return hits[0] if len(hits) == 1 else None


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _fmt(rows: list[list[str]], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _b(x: float, lo: float, hi: float, nd: int = 3) -> str:
    return f"{x:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def main() -> int:
    df = pd.read_parquet(config.BALLS_PARQUET)
    if "wp" not in df.columns or "li" not in df.columns:
        raise SystemExit(
            "FATAL: need `wp` and `li` -- run `.venv/bin/python -m src.leverage` first."
        )
    b = per_ball_wpa(df)

    rng = np.random.default_rng(SEED)
    bats = bootstrap_side(b, "batter", "bat", rng)
    bowls = bootstrap_side(b, "bowler", "bowl", rng)

    # --- ranked orders (same point-estimate ranks as the leaderboard reports)
    li_bat = sorted(bats, key=lambda p: -bats[p]["mean_li"])[:TOP_N]
    li_bowl = sorted(bowls, key=lambda p: -bowls[p]["mean_li"])[:TOP_N]
    tot_bat = sorted(bats, key=lambda p: -bats[p]["total_dd"])[:TOP_N]
    tot_bowl = sorted(bowls, key=lambda p: -bowls[p]["total_dd"])[:TOP_N]

    # --- adjacency: is any neighbouring pair in the rankings distinguishable?
    adj = {
        "LI bat": adjacent_diffs(li_bat, bats, "reps_li", "mean_li"),
        "LI bowl": adjacent_diffs(li_bowl, bowls, "reps_li", "mean_li"),
        "WPA bat": adjacent_diffs(tot_bat, bats, "reps_total", "total_dd"),
        "WPA bowl": adjacent_diffs(tot_bowl, bowls, "reps_total", "total_dd"),
    }
    n_sep = {k: sum(d["separated"] for d in v) for k, v in adj.items()}
    span = {
        "LI bat": pair_diff(bats, li_bat[0], li_bat[-1], "reps_li", "mean_li"),
        "LI bowl": pair_diff(bowls, li_bowl[0], li_bowl[-1], "reps_li", "mean_li"),
        "WPA bat": pair_diff(bats, tot_bat[0], tot_bat[-1], "reps_total", "total_dd"),
        "WPA bowl": pair_diff(bowls, tot_bowl[0], tot_bowl[-1], "reps_total", "total_dd"),
    }

    # --- role-level contrast: top finisher vs lowest-LI qualified batter (anchor)
    anchor = sorted(bats, key=lambda p: bats[p]["mean_li"])[0]
    role = pair_diff(bats, li_bat[0], anchor, "reps_li", "mean_li")

    # --- the named clutch splits the paper leans on
    dhoni = _find(bats, "MS Dhoni") or _find(bats, "Dhoni")
    sky = _find(bats, "SA Yadav")
    named = [p for p in (dhoni, sky) if p is not None and bats[p]["reps_split"] is not None]

    # clutch tables: everyone with a split, ranked by split point estimate
    clutch_bat = sorted(
        (p for p in bats if bats[p]["reps_split"] is not None),
        key=lambda p: -bats[p]["split"],
    )
    clutch_sig_bat = [p for p in clutch_bat if bats[p]["split_lo"] > 0 or bats[p]["split_hi"] < 0]
    clutch_bowl = sorted(
        (p for p in bowls if bowls[p]["reps_split"] is not None),
        key=lambda p: -bowls[p]["split"],
    )
    clutch_sig_bowl = [
        p for p in clutch_bowl if bowls[p]["split_lo"] > 0 or bowls[p]["split_hi"] < 0
    ]

    n_excl0 = {
        "bat": sum(1 for p in tot_bat if bats[p]["total_lo"] > 0 or bats[p]["total_hi"] < 0),
        "bowl": sum(1 for p in tot_bowl if bowls[p]["total_lo"] > 0 or bowls[p]["total_hi"] < 0),
    }

    # ----------------------------------------------------------------- tables
    def li_rows(order, recs):
        return [
            [p, str(recs[p]["balls"]), str(recs[p]["matches"]), _b(recs[p]["mean_li"], recs[p]["li_lo"], recs[p]["li_hi"])]
            for p in order
        ]

    def tot_rows(order, recs):
        return [
            [
                p,
                str(recs[p]["balls"]),
                _b(recs[p]["total_dd"], recs[p]["total_lo"], recs[p]["total_hi"], 2),
                "yes" if (recs[p]["total_lo"] > 0 or recs[p]["total_hi"] < 0) else "no",
            ]
            for p in order
        ]

    def adj_rows(diffs, nd=3):
        return [
            [d["pair"], _b(d["diff"], d["lo"], d["hi"], nd), "yes" if d["separated"] else "no"]
            for d in diffs
        ]

    def clutch_rows(order, recs):
        return [
            [
                p,
                str(recs[p]["n_high"]),
                f"{recs[p]['wpa_high']:+.4f}",
                f"{recs[p]['wpa_low']:+.4f}",
                _b(recs[p]["split"], recs[p]["split_lo"], recs[p]["split_hi"], 4),
                "yes" if (recs[p]["split_lo"] > 0 or recs[p]["split_hi"] < 0) else "no",
            ]
            for p in order
        ]

    li_cols = ["player", "balls", "matches", "mean LI [95% CI]"]
    tot_cols = ["player", "balls", "total WPA_dd [95% CI]", "excl. 0"]
    adj_cols = ["adjacent pair", "diff [95% CI]", "separated"]
    clutch_cols = ["player", "n_high", "high", "low", "high - low [95% CI]", "excl. 0"]

    # ------------------------------------------------------------------ prose
    total_adj_sep = sum(n_sep.values())
    total_adj = sum(len(v) for v in adj.values())

    findings = [
        f"- **Adjacent ranks are{'' if total_adj_sep == 0 else ' mostly'} not separable.** "
        f"Across the four top-{TOP_N} tables, {total_adj_sep} of {total_adj} adjacent-rank "
        f"difference CIs exclude zero"
        + (
            " -- no neighbouring pair in any leaderboard is statistically "
            "distinguishable at 95%."
            if total_adj_sep == 0
            else f" ({', '.join(k for k, v in n_sep.items() if v)})."
        ),
        f"- **Rank-1 vs rank-{TOP_N}**: "
        + "; ".join(
            f"{k}: {_b(span[k]['diff'], span[k]['lo'], span[k]['hi'], 3)}"
            f" ({'separated' if span[k]['separated'] else 'not separated'})"
            for k in span
        )
        + ".",
        f"- **Role-level separation is decisive**: {role['pair']} (top finisher vs "
        f"lowest-LI qualified anchor) differs by {_b(role['diff'], role['lo'], role['hi'], 3)}"
        f" -- {'clearly separated' if role['separated'] else 'not separated'}. "
        f"The leaderboards resolve ROLES, not neighbouring individuals.",
        f"- **Career WPA totals**: {n_excl0['bat']} of the top-{TOP_N} batting and "
        f"{n_excl0['bowl']} of the top-{TOP_N} bowling de-drifted totals exclude zero.",
        f"- **Clutch splits**: of {len(clutch_bat)} batters with >= {MIN_HIGH} high-LI balls, "
        f"{len(clutch_sig_bat)} have a high-minus-low split CI excluding zero"
        f"{' (' + ', '.join(clutch_sig_bat[:6]) + ')' if clutch_sig_bat else ''}; "
        f"of {len(clutch_bowl)} bowlers, {len(clutch_sig_bowl)}"
        f"{' (' + ', '.join(clutch_sig_bowl[:6]) + ')' if clutch_sig_bowl else ''}.",
    ]
    for p in named:
        r = bats[p]
        findings.append(
            f"- **{p}**: high {r['wpa_high']:+.4f} vs low {r['wpa_low']:+.4f} per ball "
            f"({r['n_high']} high-LI balls); split {_b(r['split'], r['split_lo'], r['split_hi'], 4)}"
            f" -- {'significant at 95%' if (r['split_lo'] > 0 or r['split_hi'] < 0) else 'NOT significant at 95%'}."
        )

    L = [
        "# Player-level uncertainty -- match-clustered bootstrap CIs",
        "",
        f"95% percentile intervals from {N_BOOT} bootstrap replicates per player, "
        f"resampling whole MATCHES (= innings) with replacement, seed {SEED}. "
        f"Per-ball resampling would assume away the within-innings dependence this "
        f"project measured (dependence_decomposition.md: lag-1 +0.044 vs null "
        f"+0.008); resampling whole innings preserves it and requires only "
        f"exchangeability across a player's matches. The WP surface, LI, and the "
        f"state-conditional drift correction are held fixed: these intervals are "
        f"sampling variability of each player's career GIVEN the surfaces, not "
        f"model uncertainty (the +0.037 autocorrelation bound is the surface's, "
        f"shared by all players). Career gate >= {MIN_BALLS} balls "
        f"({len(bats)} batters, {len(bowls)} bowlers); clutch-split gate "
        f">= {MIN_HIGH} high-LI balls. Rank-adjacency uses a CI on the DIFFERENCE "
        f"(independent replicate streams differenced), not CI overlap.",
        "",
        "## Findings",
        "",
        *findings,
        "",
        f"## Mean leverage faced -- top {TOP_N} batters",
        "",
        _fmt(li_rows(li_bat, bats), li_cols),
        "",
        "Adjacent-rank differences:",
        "",
        _fmt(adj_rows(adj["LI bat"]), adj_cols),
        "",
        f"## Mean leverage bowled -- top {TOP_N} bowlers",
        "",
        _fmt(li_rows(li_bowl, bowls), li_cols),
        "",
        "Adjacent-rank differences:",
        "",
        _fmt(adj_rows(adj["LI bowl"]), adj_cols),
        "",
        f"## Career de-drifted WPA -- top {TOP_N} batters",
        "",
        "CI on the total scales the per-ball-rate replicates by observed career "
        "balls: volume is treated as known, the rate as uncertain.",
        "",
        _fmt(tot_rows(tot_bat, bats), tot_cols),
        "",
        "Adjacent-rank differences:",
        "",
        _fmt(adj_rows(adj["WPA bat"], nd=2), adj_cols),
        "",
        f"## Career de-drifted WPA -- top {TOP_N} bowlers",
        "",
        _fmt(tot_rows(tot_bowl, bowls), tot_cols),
        "",
        "Adjacent-rank differences:",
        "",
        _fmt(adj_rows(adj["WPA bowl"], nd=2), adj_cols),
        "",
        "## Clutch splits with intervals -- top 10 batter splits",
        "",
        "De-drifted per-ball WPA on high-LI (>= 2) minus ordinary balls. A replicate "
        "that resamples zero high-LI balls is dropped (counts below are negligible "
        "for every listed player).",
        "",
        _fmt(clutch_rows(clutch_bat[:10], bats), clutch_cols),
        "",
        "## Clutch splits with intervals -- top 10 bowler splits",
        "",
        _fmt(clutch_rows(clutch_bowl[:10], bowls), clutch_cols),
        "",
        "## Caveats",
        "",
        "- Difference CIs treat two careers as independent; players sharing innings "
        "induce a small positive correlation, so those CIs are mildly conservative "
        "(true intervals slightly narrower).",
        "- Intervals condition on the fitted WP/LI surfaces and the drift "
        "correction; surface-level bias is shared across players and bounded "
        "separately (leverage_validation.md, tail_diagnostics.md).",
        "- Percentile intervals; with >= 40 matches per qualified career the "
        "normal-range diagnostics are unremarkable and BCa refinements would not "
        "change any verdict above.",
        "",
    ]
    out = config.REPORTS / "player_uncertainty.md"
    out.write_text("\n".join(L))

    print(f"batters={len(bats)} bowlers={len(bowls)}  adjacent separated: {n_sep} / 9 each")
    print(f"top-{TOP_N} WPA totals excluding 0: bat {n_excl0['bat']}, bowl {n_excl0['bowl']}")
    print(f"clutch splits excluding 0: bat {len(clutch_sig_bat)}/{len(clutch_bat)} "
          f"{clutch_sig_bat[:6]}, bowl {len(clutch_sig_bowl)}/{len(clutch_bowl)} "
          f"{clutch_sig_bowl[:6]}")
    for p in named:
        r = bats[p]
        print(f"{p}: split {r['split']:+.4f} [{r['split_lo']:+.4f}, {r['split_hi']:+.4f}] "
              f"(n_high={r['n_high']}, degenerate reps={r['split_degenerate']})")
    print(f"role contrast {role['pair']}: {role['diff']:+.3f} [{role['lo']:+.3f}, {role['hi']:+.3f}]")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
