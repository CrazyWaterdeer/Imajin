"""Benchmark: can registration-free persistence vectors recover neuron *type*?

Measures the ceiling and the degradation of the registration-free matching idea
BEFORE building any user-facing connectome lookup (Codex review #9). Token-only,
neuPrint EM data — no user data, no registration.

  1. CEILING: within hemibrain (same modality/scale), does persistence-vector NN
     recover type above chance? (leave-one-out)
  2. DEGRADE: how far does it fall when the query skeleton is fragmented /
     scale-perturbed / down-sampled — recomputed at the SKELETON level, and with
     the query's own bodyId excluded from the reference (no self-match).
  3. CONTROL: shuffled labels must collapse to chance.

Decision rule: ceiling >> chance AND survives fragmentation -> worth a caveated
browser; collapses under fragmentation -> too fragile for partial confocal traces,
pivot to registration.

Run:
  NEUPRINT_APPLICATION_CREDENTIALS="$(cat ~/.config/neuprint/token)" \
      uv run python scripts/bench_persistence_typeid.py
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")

DATASET = "hemibrain:v1.2.1"
FAMILIES = ["EPG", "PEN", "PFN", "Delta7", "ExR", "PEG", "EL", "FB4", "FS", "FR"]
MIN_PER_TYPE, MAX_PER_TYPE = 6, 8
RESAMPLE_UM, SAMPLES, SEED = 1.0, 64, 0
SKEL_DIR = Path("/tmp/imajin_skel_cache")
META = Path("/tmp/imajin_bench_meta.json")


def _voxel_um(client) -> float:
    try:
        vs = client.meta.get("voxelSize") or client.meta.get("voxelsize")
        nm = float(vs["x"]) if isinstance(vs, dict) else float(vs[0] if isinstance(vs, (list, tuple)) else vs)
        return nm / 1000.0
    except Exception:
        return 0.008


def _persistence(df, voxel_um, *, scale=1.0, resample_to=RESAMPLE_UM, jitter=0.0, rng=None):
    """Persistence vector from a neuPrint skeleton node-table (rowId,x,y,z,radius,link)."""
    import navis

    d = df.rename(columns={"rowId": "node_id", "link": "parent_id"}).copy()
    for c in ("x", "y", "z", "radius"):
        d[c] = d[c].astype(float) * voxel_um * (scale if c != "radius" else 1.0)
    if jitter and rng is not None:
        for c in ("x", "y", "z"):
            d[c] = d[c] + rng.normal(0, jitter, size=len(d))
    try:
        n = navis.TreeNeuron(d, units="um")
        n = navis.resample_skeleton(n, resample_to=resample_to)
        if n.nodes.shape[0] < 5:
            return None
        vec = np.asarray(navis.persistence_vectors(n, samples=SAMPLES)[0]).ravel()
        return [float(v) for v in vec] if vec.size == SAMPLES and np.all(np.isfinite(vec)) else None
    except Exception:
        return None


def _subtree(df, frac, rng):
    """Keep a connected distal subtree of roughly `frac` of the nodes (a fragment)."""
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


def collect():
    from neuprint import Client, NeuronCriteria as NC, fetch_neurons

    SKEL_DIR.mkdir(exist_ok=True)
    c = Client("neuprint.janelia.org", dataset=DATASET)
    vox = _voxel_um(c)
    print(f"voxel_um={vox}")
    rng = np.random.default_rng(SEED)
    recs = []  # (bodyId, type)
    for fam in FAMILIES:
        ndf, _ = fetch_neurons(NC(type=f"{fam}.*", regex=True, status="Traced", cropped=False))
        if ndf.empty:
            continue
        top = ndf["type"].value_counts().idxmax()
        sub = ndf[ndf["type"] == top]
        if len(sub) < MIN_PER_TYPE:
            continue
        bids = sub["bodyId"].tolist()
        rng.shuffle(bids)
        kept = 0
        for bid in bids:
            if kept >= MAX_PER_TYPE:
                break
            f = SKEL_DIR / f"{bid}.npz"
            if not f.exists():
                sk = c.fetch_skeleton(int(bid), format="pandas")
                np.savez_compressed(
                    f, rowId=sk.rowId.values, x=sk.x.values, y=sk.y.values,
                    z=sk.z.values, radius=sk.radius.values, link=sk.link.values,
                )
            recs.append((int(bid), top))
            kept += 1
        print(f"  {top:16s}: {kept}")
    META.write_text(json.dumps({"voxel_um": vox, "recs": recs}))
    return vox, recs


def _load_df(bid):
    import pandas as pd

    z = np.load(SKEL_DIR / f"{bid}.npz")
    return pd.DataFrame({k: z[k] for k in ("rowId", "x", "y", "z", "radius", "link")})


def _topk(ref_vecs, ref_types, ref_bids, q_vec, q_type, q_bid, ks=(1, 5)):
    d = np.linalg.norm(ref_vecs - q_vec, axis=1)
    ranked = [ref_types[j] for j in np.argsort(d) if ref_bids[j] != q_bid]
    return {k: (q_type in ranked[:k]) for k in ks}


def main():
    vox, recs = collect()
    if len(recs) < 20:
        print(f"Not enough neurons ({len(recs)})")
        return
    rng = np.random.default_rng(SEED)
    base = {}  # bid -> vector
    for bid, _t in recs:
        v = _persistence(_load_df(bid), vox)
        if v is not None:
            base[bid] = np.asarray(v, float)
    recs = [(b, t) for b, t in recs if b in base]
    bids = [b for b, _ in recs]
    types = [t for _, t in recs]
    V = np.vstack([base[b] for b in bids])
    N = len(recs)
    counts = {t: types.count(t) for t in set(types)}
    chance = sum(c * (c - 1) for c in counts.values()) / (N * (N - 1))
    print(f"\n=== {N} neurons, {len(counts)} types | chance top-1 = {chance:.3f} ===")

    def run(make_query, label):
        hit1 = hit5 = tot = 0
        for i, (bid, t) in enumerate(recs):
            qv = make_query(bid, V[i])
            if qv is None:
                continue
            r = _topk(V, types, bids, qv, t, bid)
            hit1 += r[1]; hit5 += r[5]; tot += 1
        print(f"  {label:22s} top-1={hit1/tot:.3f}  top-5={hit5/tot:.3f}  (n={tot})")

    print("[CEILING]")
    run(lambda b, v: v, "leave-one-out")
    print("[CONTROL]")
    shuf = list(types); rng.shuffle(shuf)
    # shuffle: reuse run but with shuffled ref types
    h1 = tot = 0
    for i, (bid, _t) in enumerate(recs):
        d = np.linalg.norm(V - V[i], axis=1)
        ranked = [shuf[j] for j in np.argsort(d) if bids[j] != bid]
        h1 += (shuf[i] in ranked[:1]); tot += 1
    print(f"  {'shuffled-labels':22s} top-1={h1/tot:.3f}  (≈chance)")

    print("[DEGRADE — recomputed at skeleton level, self excluded]")
    run(lambda b, v: _persistence(_load_df(b), vox, scale=0.8), "scale x0.8")
    run(lambda b, v: _persistence(_load_df(b), vox, scale=1.2), "scale x1.2")
    run(lambda b, v: _persistence(_load_df(b), vox, resample_to=3.0, jitter=0.5, rng=rng), "lowres+jitter")
    run(lambda b, v: _persistence(_subtree(_load_df(b), 0.5, rng), vox), "fragment ~50%")
    run(lambda b, v: _persistence(_subtree(_load_df(b), 0.3, rng), vox), "fragment ~30%")

    print("\nVERDICT: ceiling >> chance AND fragment top-5 stays useful -> caveated browser;")
    print("fragment collapses toward chance -> pivot to registration.")


if __name__ == "__main__":
    main()
