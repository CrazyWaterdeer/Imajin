"""Stage C — NBLAST a (registered) trace against connectome dotprops.

Thin wrapper around navis NBLAST. The query trace is assumed to be **already warped
into the connectome's space** (registration → template → bridge happens upstream);
this module only scores morphology. Validated by `scripts/bench_nblast_typeid.py`
and `bench_nblast_regerror.py` (NBLAST is robust to fragmentation and ~few-µm
registration error). `navis` / `flybrains` are the optional ``connectome`` extra and
are imported lazily; this module imports clean without them.

Outputs are **QC-gated candidates requiring review**, never definitive identity
(see the design spec): the caller adds the registration-QC gate, uncertainty
(score gap / null percentile), priors, and provenance.
"""
from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from imajin.analysis.morphology_nblast import navis_available


def connectome_backend_available() -> bool:
    """True when navis + flybrains (templates/bridging) are both importable."""
    return navis_available() and importlib.util.find_spec("flybrains") is not None


def download_bridge_assets(*, vnc: bool = False) -> None:
    """Download + register the flybrains bridging registrations (one-time, large).

    Required before :func:`warp_to_connectome_space` can bridge a trace into the
    hemibrain (JRCFIB2018F) space. **Finding (2026-06-15): the jefferislab CMTK set
    alone does NOT yield a JRC2018F→JRCFIB2018F path** — the JRC H5 inter-template
    transforms (large, possibly ~GB) are also needed, so this downloads both. The
    bridging transforms are an external asset supply chain (not bundled) — see the
    Stage C design spec; this can be slow and needs network + disk budget.
    """
    import flybrains

    flybrains.download_jefferislab_transforms()
    flybrains.download_jrc_transforms()
    if vnc:
        flybrains.download_jrc_vnc_transforms()
    flybrains.register_transforms()


def warp_to_connectome_space(
    points_um: np.ndarray,
    *,
    source: str = "JRC2018F",
    target: str = "JRCFIB2018F",
):
    """Bridge trace points from a template space into the connectome's space.

    ``source`` is the template the trace was registered into (JRC2018F for brain);
    ``target`` is the connectome space (JRCFIB2018F = hemibrain). Returns
    ``(warped_points | None, status)`` with a typed status: ``"ok"``,
    ``"needs_bridge_assets"`` (no bridging path registered — run
    :func:`download_bridge_assets`), or ``"backend_unavailable"``. The
    sample→template registration itself is upstream/external.
    """
    if not connectome_backend_available():
        return None, "backend_unavailable"
    import navis

    pts = np.asarray(points_um, dtype=float)
    try:
        out = navis.xform_brain(pts, source=source, target=target)
        return np.asarray(out, dtype=float), "ok"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        # No path between spaces, or no bridging registrations downloaded at all.
        if "nopath" in type(exc).__name__.lower() or "bridging" in msg:
            return None, "needs_bridge_assets"
        raise


def points_to_dotprops(
    points_um: np.ndarray,
    *,
    k: int = 5,
    neuron_id: Any = None,
):
    """Build navis Dotprops from an Nx3 array of micron-space trace points.

    Returns ``None`` when navis is unavailable or the point cloud is too small for
    a k-nearest-neighbour tangent estimate.
    """
    if not navis_available():
        return None
    pts = np.asarray(points_um, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < k + 1:
        return None
    try:
        import navis

        dp = navis.make_dotprops(pts, k=k)
        # navis NBLAST requires units; dotprops built from a raw array have none.
        dp.units = "um"
        if neuron_id is not None:
            dp.id = neuron_id
        return dp
    except Exception:
        return None


def nblast_candidates(
    query_dp: Any,
    reference: list[tuple[Any, Any, Any]],
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Rank connectome reference neurons by NBLAST similarity to a query.

    ``reference`` is ``[(id, type, dotprops), ...]`` (connectome neurons already in a
    common space). Returns up to ``top_k`` ``{id, type, nblast_score}`` sorted by
    NBLAST score (mean of forward/reverse, normalized; the fly score matrix is
    selected by ``smat='auto'`` for micron-scale data). These are *candidates*, not
    identifications.
    """
    if query_dp is None or not reference:
        return []
    import navis

    ref_dps = []
    meta: dict[Any, Any] = {}
    for rid, rtype, dp in reference:
        if dp is None:
            continue
        dp.id = rid
        ref_dps.append(dp)
        meta[rid] = rtype
    if not ref_dps:
        return []

    scores = navis.nblast(
        navis.NeuronList([query_dp]),
        navis.NeuronList(ref_dps),
        scores="mean",
        smat="auto",
        progress=False,
        n_cores=1,
    )
    row = scores.iloc[0].sort_values(ascending=False)
    out: list[dict[str, Any]] = []
    for rid in row.index[:top_k]:
        out.append(
            {
                "id": rid,
                "type": meta.get(rid),
                "nblast_score": float(row[rid]),
            }
        )
    return out
