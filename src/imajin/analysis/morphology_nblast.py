"""NBLAST spatial morphology comparison (Tier-2 backend, optional).

Isolates every ``navis`` use so the rest of the app stays import-clean without the
optional ``connectome`` extra. NBLAST is calibrated for neurons in **microns** and
is **not** translation/rotation invariant — neurons must share a coordinate frame
(same acquisition, or registered into a common template) for scores to be
meaningful. Callers that have only unregistered, pixel-scale traces should prefer
the registration-free morphometric matcher (``morphology_match``).

``navis`` is imported lazily inside the functions; importing this module does not
require the extra.
"""
from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np


def navis_available() -> bool:
    """True iff the optional ``navis`` dependency is importable (no import side effects)."""
    return importlib.util.find_spec("navis") is not None


def backend_status() -> dict[str, Any]:
    available = navis_available()
    return {
        "backend": "navis-nblast",
        "available": available,
        "hint": None if available else "Install the connectome extra: uv sync --extra connectome",
    }


def _is_physical(units: Any) -> bool:
    if units is None:
        return False
    seq = units if isinstance(units, (list, tuple)) else [units]
    return bool(seq) and all(bool(u) for u in seq)


def _to_3d(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return pts.reshape(-1, 3) if pts.size == 0 else pts
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    return pts


def nblast_against_references(
    query_points: np.ndarray,
    query_units: Any,
    references: list[dict[str, Any]],
    *,
    k: int = 5,
    dotprops_k: int = 5,
) -> dict[str, Any]:
    """NBLAST ``query_points`` against labelled reference point clouds.

    Each reference is ``{"name", "label", "points" (Nx3 µm), "units"}``. Returns
    ``{status, ranked}`` where ``status`` is ``ok`` / ``backend_unavailable`` /
    ``needs_microns`` / ``no_reference``. NBLAST scores are normalized (1.0 = self).
    """
    if not navis_available():
        return {
            "status": "backend_unavailable",
            "ranked": [],
            "note": backend_status()["hint"],
        }
    # NBLAST scoring is calibrated for microns; refuse uncalibrated (pixel) data
    # rather than returning meaningless scores.
    if not _is_physical(query_units):
        return {
            "status": "needs_microns",
            "ranked": [],
            "note": (
                "NBLAST requires physical (micron) coordinates. Skeletonize from a "
                "layer with a physical scale, or use find_similar_neurons (morphometric)."
            ),
        }

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import navis

        def _dotprops(points: np.ndarray, neuron_id: str):
            dp = navis.make_dotprops(_to_3d(points), k=dotprops_k)
            dp.units = "um"
            dp.id = neuron_id
            return dp

        targets = []
        for ref in references:
            if not _is_physical(ref.get("units")):
                continue
            targets.append((ref, _dotprops(ref["points"], str(ref["name"]))))
        if not targets:
            return {"status": "no_reference", "ranked": [], "note": "No micron-scale references."}

        query = _dotprops(query_points, "__query__")
        scores = navis.nblast(
            navis.NeuronList([query]),
            navis.NeuronList([dp for _, dp in targets]),
            n_cores=1,
            progress=False,
        )
        row = scores.loc["__query__"]

    ranked = sorted(
        (
            {"name": str(ref["name"]), "label": ref.get("label"), "score": float(row[dp.id])}
            for ref, dp in targets
        ),
        key=lambda r: -r["score"],
    )[: max(1, int(k))]
    return {"status": "ok", "ranked": ranked}
