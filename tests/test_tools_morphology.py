"""Characterization net for neural morphology matching (Tier 1).

Pins (a) the current stub contract of ``classify_neuron_type`` /
``query_connectome`` so the change in N4/N6 is observable, and (b) the
``compute_morphology_descriptors`` output contract that the Tier-1 feature
extractor (N1) consumes. Also provides the labelled toy-skeleton fixture that the
feature / reference-library / classifier commits reuse.

Everything here passes on the current code with only existing dependencies.
"""
from __future__ import annotations

import numpy as np
import pytest

from imajin.tools import trace


# --------------------------------------------------------------------------- #
# Toy shapes — chosen to be morphometrically distinct so downstream
# classification/similarity tests are meaningful (straight < Y-branch < bushy in
# branch/junction/endpoint counts).
# --------------------------------------------------------------------------- #
def _straight_mask() -> np.ndarray:
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:55, 32] = 1
    return img


def _branched_mask() -> np.ndarray:
    """Y-shape: vertical trunk + two diagonal branches in the upper half."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:55, 31:33] = 1
    for i in range(20):
        img[10 + i, 31 - i] = 1
        img[10 + i, 32 + i] = 1
    return img


def _bushy_mask() -> np.ndarray:
    """Trunk with several short spurs on alternating sides."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:55, 32] = 1
    for y in (18, 28, 38, 48):
        img[y, 32:40] = 1
    for y in (23, 33, 43):
        img[y, 25:32] = 1
    return img


# Keys the Tier-1 feature extractor (N1) reads from compute_morphology_descriptors.
_FEATURE_SOURCE_KEYS = {
    "total_length",
    "length_unit",
    "mean_branch_length",
    "median_branch_length",
    "n_branches",
    "n_endpoints",
    "n_junctions",
    "n_components",
    "n_terminal_branches",
    "n_internal_branches",
    "bbox_scaled",
    "skeleton_volume_occupancy",
}


@pytest.fixture
def labeled_morphology_samples(viewer) -> list[dict]:
    """Skeletonize the toy shapes and return their labelled descriptors.

    Reused by the feature-extractor / reference-library / classifier commits.
    """
    trace.reset_skeletons()
    samples: list[dict] = []
    for label, mask, name in [
        ("linear", _straight_mask(), "m_linear"),
        ("branched", _branched_mask(), "m_branched"),
        ("bushy", _bushy_mask(), "m_bushy"),
    ]:
        viewer.add_labels(mask, name=name)
        skel_id = trace.skeletonize(name)["skeleton_id"]
        samples.append(
            {
                "label": label,
                "name": name,
                "skeleton_id": skel_id,
                "descriptors": trace.compute_morphology_descriptors(skel_id),
            }
        )
    return samples


# --------------------------------------------------------------------------- #
# Current stub contract (pinned so N4/N6 changes are observable)
# --------------------------------------------------------------------------- #
def test_classify_neuron_type_is_currently_stub(viewer) -> None:
    res = trace.classify_neuron_type("any_id")
    assert res["status"] == "not_implemented"


def test_query_connectome_is_currently_stub(viewer) -> None:
    res = trace.query_connectome("any_id", db="neuprint", k=5)
    assert res["status"] == "not_implemented"
    assert res["matches"] == []


# --------------------------------------------------------------------------- #
# Descriptor contract that the Tier-1 feature extractor depends on
# --------------------------------------------------------------------------- #
def test_descriptor_output_exposes_feature_source_keys(
    labeled_morphology_samples,
) -> None:
    for sample in labeled_morphology_samples:
        missing = _FEATURE_SOURCE_KEYS - set(sample["descriptors"])
        assert not missing, f"{sample['label']} missing descriptor keys: {missing}"


def test_toy_shapes_are_morphometrically_discriminable(
    labeled_morphology_samples,
) -> None:
    by_label = {s["label"]: s["descriptors"] for s in labeled_morphology_samples}

    # a straight line has no junctions; branched/bushy do, increasingly so
    assert by_label["linear"]["n_junctions"] == 0
    assert by_label["branched"]["n_junctions"] >= 1
    assert by_label["bushy"]["n_junctions"] > by_label["branched"]["n_junctions"]

    # endpoint count tracks branchiness too
    assert by_label["bushy"]["n_endpoints"] > by_label["linear"]["n_endpoints"]
