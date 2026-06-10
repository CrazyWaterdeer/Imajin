"""Morphometric nearest-neighbour matching against a reference library.

Standardizes features (so counts and ratios are comparable) and ranks reference
neurons by Euclidean distance in feature space. When the query and the library do
not agree on physical units, it restricts to the scale-invariant feature subset so
a pixel-scale neuron is never compared to a micron-scale one on absolute lengths.

Pure ``scikit-learn`` (already a dependency) + numpy/pandas — no new dependency,
no registration, no network.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler


def _confidence(distances: np.ndarray, order: np.ndarray) -> float:
    """Separation of the nearest match from the runner-up, in [0, 1]."""
    if len(order) < 2:
        return 1.0
    d1 = float(distances[order[0]])
    d2 = float(distances[order[1]])
    if d2 <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - d1 / d2))


def match_against_library(
    query_fv: dict[str, Any], library: Any, *, k: int = 5
) -> dict[str, Any]:
    """Rank ``library`` neurons by morphometric similarity to ``query_fv``.

    ``query_fv`` is the dict from ``extract_feature_vector``; ``library`` is a
    ``ReferenceLibrary``. Returns ``{status, predicted, confidence, ranked,
    features_used, invariant_only}``.
    """
    query_features = query_fv.get("features", {})
    query_physical = bool(query_fv.get("units_physical", False))
    invariant_only = not (query_physical and library.all_physical)

    candidate = query_fv.get("invariant_keys", []) if invariant_only else list(query_features)
    # keep only columns the library actually has, the query provides, and that are
    # fully populated in the library (a mixed library leaves absolute columns blank)
    cols = sorted(
        c
        for c in candidate
        if c in library.feature_columns
        and c in query_features
        and not library.frame[c].isna().any()
    )
    if not cols:
        return {
            "status": "no_features",
            "predicted": None,
            "confidence": None,
            "ranked": [],
            "features_used": [],
            "invariant_only": invariant_only,
        }

    X = library.frame[cols].to_numpy(dtype=float)
    q = np.array([[float(query_features[c]) for c in cols]], dtype=float)

    scaler = StandardScaler().fit(X)  # zero-variance columns are handled (scale→1)
    distances = np.linalg.norm(scaler.transform(X) - scaler.transform(q), axis=1)
    order = np.argsort(distances)

    names = library.names
    labels = library.labels
    k_eff = max(1, min(int(k), len(order)))
    ranked = [
        {"name": names[i], "label": labels[i], "distance": float(distances[i])}
        for i in order[:k_eff]
    ]

    return {
        "status": "ok",
        "predicted": ranked[0]["label"],
        "confidence": _confidence(distances, order),
        "ranked": ranked,
        "features_used": cols,
        "invariant_only": invariant_only,
    }
