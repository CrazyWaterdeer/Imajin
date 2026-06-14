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
