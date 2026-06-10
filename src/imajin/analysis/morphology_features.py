"""Registration-free morphometric feature vectors for neuron classification.

Turns the dict produced by ``compute_morphology_descriptors`` (plus an optional
per-branch table) into a feature vector for similarity/classification. Features are
split into two groups:

- **scale-invariant** (counts, ratios, tortuosity, occupancy): independent of
  whether the skeleton is measured in pixels or microns. Always emitted.
- **absolute** (lengths, bounding-box size): only meaningful in physical units, so
  they are emitted *only* when the source is in microns (``length_unit == "um"``).

This split is the guard against silently comparing pixel-scale and micron-scale
neurons (the units-mismatch bug class): callers that mix the two should restrict to
``invariant_keys``.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _safe_div(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    return float(numerator) / denominator if denominator else 0.0


def _mean_tortuosity(branch_df: Any) -> float | None:
    """Mean branch tortuosity (path length / euclidean distance), if derivable.

    Scale-invariant: both columns scale identically with spacing.
    """
    if branch_df is None:
        return None
    if "branch_length" not in branch_df.columns or "euclidean_distance" not in branch_df.columns:
        return None
    length = branch_df["branch_length"].to_numpy(dtype=float)
    straight = branch_df["euclidean_distance"].to_numpy(dtype=float)
    valid = straight > 0
    if not valid.any():
        return None
    return float(np.mean(length[valid] / straight[valid]))


def extract_feature_vector(
    descriptors: dict[str, Any], branch_df: Any = None
) -> dict[str, Any]:
    """Build a morphometric feature vector from descriptor output.

    Returns ``{"features": {...}, "units_physical": bool, "invariant_keys": [...]}``.
    ``invariant_keys`` lists the subset usable when comparing across mixed units.
    """
    units_physical = descriptors.get("length_unit") == "um"
    n_branches = max(int(descriptors.get("n_branches", 0) or 0), 0)
    n_junctions = int(descriptors.get("n_junctions", 0) or 0)
    n_endpoints = int(descriptors.get("n_endpoints", 0) or 0)

    features: dict[str, float] = {
        # topology counts — invariant to the pixel-vs-micron unit choice
        "n_branches": float(n_branches),
        "n_endpoints": float(n_endpoints),
        "n_junctions": float(n_junctions),
        "n_components": float(descriptors.get("n_components", 0) or 0),
        # dimensionless ratios — invariant
        "terminal_fraction": _safe_div(descriptors.get("n_terminal_branches", 0), n_branches),
        "internal_fraction": _safe_div(descriptors.get("n_internal_branches", 0), n_branches),
        "endpoints_per_junction": _safe_div(n_endpoints, max(n_junctions, 1)),
        "mean_to_median_length": _safe_div(
            descriptors.get("mean_branch_length", 0),
            descriptors.get("median_branch_length", 0),
        ),
        "volume_occupancy": float(descriptors.get("skeleton_volume_occupancy", 0.0) or 0.0),
    }

    tortuosity = _mean_tortuosity(branch_df)
    if tortuosity is not None:
        features["mean_tortuosity"] = tortuosity

    invariant_keys = sorted(features)

    if units_physical:
        bbox = descriptors.get("bbox_scaled") or ()
        features["total_length_um"] = float(descriptors.get("total_length", 0.0) or 0.0)
        features["mean_branch_length_um"] = float(
            descriptors.get("mean_branch_length", 0.0) or 0.0
        )
        features["bbox_diagonal_um"] = (
            float(np.linalg.norm(np.asarray(bbox, dtype=float))) if len(bbox) else 0.0
        )

    return {
        "features": features,
        "units_physical": units_physical,
        "invariant_keys": invariant_keys,
    }
