"""Go/no-go: is NBLAST fragmentation-robust where persistence collapsed?

Reuses the EXACT cached hemibrain skeletons + fragmentation regimes from
`bench_persistence_typeid.py` (same 72 neurons / 9 types in /tmp/imajin_skel_cache),
so NBLAST vs persistence is apples-to-apples. NBLAST is the gold standard the
registration pipeline (Stage C) would use; if it ALSO collapses on partial traces,
even registration won't rescue the real (partial confocal) use case.

Persistence baseline (same data): ceiling top-1 0.528 / top-5 0.750; fragment ~50%
0.208 / 0.472; fragment ~30% 0.139 / 0.417 (≈chance).

GO if NBLAST clearly beats that under fragmentation. NO-GO if it collapses too.

Run (no token needed — skeletons are cached):
  uv run python scripts/bench_nblast_typeid.py
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
VOXEL_UM = 0.008
RESAMPLE_UM = 1.0
SEED = 0


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
            r = stack.pop()
            keep.add(r)
            stack.extend(children.get(r, []))
        if best is None or abs(len(keep) - target) < abs(len(best[1]) - target):
            best = (seed, keep)
        if 0.5 * target <= len(keep) <= 1.5 * target:
            break
    seed, keep = best
    sub = df[df.rowId.isin(keep)].copy()
    sub.loc[sub.rowId == seed, "link"] = -1
    return sub


def _dotprops(df, bid, *, scale=1.0):
    import navis

    d = df.rename(columns={"rowId": "node_id", "link": "parent_id"}).copy()
    for c in ("x", "y", "z", "radius"):
        d[c] = d[c].astype(float) * VOXEL_UM * (scale if c != "radius" else 1.0)
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

    ref = {}
    for bid, _t in recs:
        dp = _dotprops(_load_df(bid), bid)
        if dp is not None:
            ref[bid] = dp
    recs = [(b, t) for b, t in recs if b in ref]
    bids = [b for b, _ in recs]
    types = {b: t for b, t in recs}
    ref_list = [ref[b] for b in bids]
    N = len(bids)
    counts = {t: list(types.values()).count(t) for t in set(types.values())}
    chance = sum(c * (c - 1) for c in counts.values()) / (N * (N - 1))
    print(f"=== {N} neurons, {len(counts)} types | chance top-1 = {chance:.3f} ===")

    # all-vs-all for the ceiling (leave-one-out)
    S = _nblast(ref_list, ref_list)  # rows=query, cols=target (bodyId index)

    def topk_from_scores(scores_row, q_bid, ks=(1, 5)):
        order = scores_row.sort_values(ascending=False).index.tolist()
        ranked = [types[int(b)] for b in order if int(b) != q_bid]
        return {k: (types[q_bid] in ranked[:k]) for k in ks}

    h1 = h5 = 0
    for b in bids:
        r = topk_from_scores(S.loc[b], b)
        h1 += r[1]; h5 += r[5]
    print(f"[CEILING] NBLAST leave-one-out: top-1={h1/N:.3f}  top-5={h5/N:.3f}")
    print("          (persistence baseline: 0.528 / 0.750)")

    def degrade(make_q, label):
        h1 = h5 = tot = 0
        for b in bids:
            dq = make_q(b)
            if dq is None:
                continue
            row = _nblast([dq], ref_list).iloc[0]
            r = topk_from_scores(row, b)
            h1 += r[1]; h5 += r[5]; tot += 1
        print(f"  {label:16s} top-1={h1/tot:.3f}  top-5={h5/tot:.3f}  (n={tot})")

    print("[DEGRADE — fragment, self excluded]")
    degrade(lambda b: _dotprops(_subtree(_load_df(b), 0.5, rng), b), "fragment ~50%")
    degrade(lambda b: _dotprops(_subtree(_load_df(b), 0.3, rng), b), "fragment ~30%")
    print("          (persistence fragment 50%: 0.208/0.472 ; 30%: 0.139/0.417 ≈chance)")
    print("\nGO if NBLAST fragment top-5 stays well above persistence + chance; else NO-GO.")


if __name__ == "__main__":
    main()
