"""Go/no-go (the real gate): how much REGISTRATION ERROR can NBLAST tolerate?

The within-hemibrain NBLAST benchmark proved fragment-robustness. The remaining
question before building the registration pipeline: does NBLAST still recover type
once the query carries realistic registration error? nc82->template registration
achieves ~2 um on average (Bogovic et al. 2020). So sweep injected displacement
sigma and read recovery at ~2 um, combined with fragmentation.

Conservative model: per-node independent Gaussian displacement (um). Real
registration error is spatially smooth, so independent jitter is HARSHER than a
smooth warp of the same RMS — if NBLAST holds here, it holds for real warps.

GO if type recovery stays useful (top-5 well above chance) at sigma ~2 um with
fragmentation. Reuses cached hemibrain skeletons (no token/network).

Run:  uv run python scripts/bench_nblast_regerror.py
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")

SKEL_DIR = Path("/tmp/imajin_skel_cache")
META = Path("/tmp/imajin_bench_meta.json")
VOXEL_UM, RESAMPLE_UM, SEED = 0.008, 1.0, 0


def _load_df(bid):
    import pandas as pd

    z = np.load(SKEL_DIR / f"{bid}.npz")
    return pd.DataFrame({k: z[k] for k in ("rowId", "x", "y", "z", "radius", "link")})


def _subtree(df, frac, rng):
    children = defaultdict(list)
    for rid, link in zip(df.rowId.tolist(), df.link.tolist()):
        if link != -1:
            children[link].append(rid)
    rids = df.rowId.tolist()
    target = max(5, int(len(rids) * frac))
    best = None
    for _ in range(25):
        seed = int(rng.choice(rids))
        keep, stack = set(), [seed]
        while stack:
            r = stack.pop(); keep.add(r); stack.extend(children.get(r, []))
        if best is None or abs(len(keep) - target) < abs(len(best[1]) - target):
            best = (seed, keep)
        if 0.5 * target <= len(keep) <= 1.5 * target:
            break
    seed, keep = best
    sub = df[df.rowId.isin(keep)].copy()
    sub.loc[sub.rowId == seed, "link"] = -1
    return sub


def _dotprops(df, bid, *, jitter_um=0.0, rng=None):
    import navis

    d = df.rename(columns={"rowId": "node_id", "link": "parent_id"}).copy()
    for c in ("x", "y", "z", "radius"):
        d[c] = d[c].astype(float) * VOXEL_UM
    if jitter_um and rng is not None:
        for c in ("x", "y", "z"):
            d[c] = d[c] + rng.normal(0, jitter_um, size=len(d))
    try:
        n = navis.TreeNeuron(d, units="um", id=int(bid))
        n = navis.resample_skeleton(n, resample_to=RESAMPLE_UM)
        if n.nodes.shape[0] < 5:
            return None
        dp = navis.make_dotprops(n, k=5)
        dp.id = int(bid)
        return dp
    except Exception:
        return None


def _nblast(query_dps, ref_dps):
    import navis

    return navis.nblast(
        navis.NeuronList(query_dps), navis.NeuronList(ref_dps),
        scores="mean", smat="auto", progress=False, n_cores=1,
    )


def main():
    meta = json.loads(META.read_text())
    recs = [(int(b), t) for b, t in meta["recs"] if (SKEL_DIR / f"{b}.npz").exists()]
    rng = np.random.default_rng(SEED)
    ref = {b: _dotprops(_load_df(b), b) for b, _ in recs}
    ref = {b: dp for b, dp in ref.items() if dp is not None}
    recs = [(b, t) for b, t in recs if b in ref]
    bids = [b for b, _ in recs]
    types = {b: t for b, t in recs}
    ref_list = [ref[b] for b in bids]
    N = len(bids)
    counts = {t: list(types.values()).count(t) for t in set(types.values())}
    chance = sum(c * (c - 1) for c in counts.values()) / (N * (N - 1))
    print(f"=== {N} neurons, {len(counts)} types | chance top-1={chance:.3f} (top-5~{1-(1-chance)**5:.2f}) ===")
    print("nc82 registration is ~2 um on average; read the sigma=2 row.\n")

    def topk(row, q_bid, ks=(1, 5)):
        order = row.sort_values(ascending=False).index.tolist()
        ranked = [types[int(b)] for b in order if int(b) != q_bid]
        return {k: (types[q_bid] in ranked[:k]) for k in ks}

    print(f"{'reg-error sigma':>15} | {'full query':>18} | {'fragment ~50%':>18}")
    for sigma in (0.0, 1.0, 2.0, 4.0, 8.0):
        line = f"{sigma:>13.0f}um |"
        for frac in (None, 0.5):
            h1 = h5 = tot = 0
            for b in bids:
                df = _load_df(b)
                if frac is not None:
                    df = _subtree(df, frac, rng)
                dq = _dotprops(df, b, jitter_um=sigma, rng=rng)
                if dq is None:
                    continue
                r = topk(_nblast([dq], ref_list).iloc[0], b)
                h1 += r[1]; h5 += r[5]; tot += 1
            line += f"  t1={h1/tot:.2f} t5={h5/tot:.2f}    |"
        print(line)

    print("\nGO if sigma=2um row stays top-5 well above chance (esp. for full query);")
    print("NO-GO if recovery collapses by 2um (registration would be too fragile).")


if __name__ == "__main__":
    main()
