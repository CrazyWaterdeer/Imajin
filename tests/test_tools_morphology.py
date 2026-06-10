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

from imajin.analysis.morphology_features import extract_feature_vector
from imajin.analysis.morphology_match import match_against_library
from imajin.analysis.morphology_reference import (
    append_reference,
    load_reference_library,
)
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


def _branched_variant_mask() -> np.ndarray:
    """A different Y-shape (held-out query): same topology, different placement."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[14:50, 29:31] = 1
    for i in range(16):
        img[14 + i, 29 - i] = 1
        img[14 + i, 30 + i] = 1
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


# --------------------------------------------------------------------------- #
# N1: feature extractor + unit guard
# --------------------------------------------------------------------------- #
def _descriptor_dict(length_unit: str, scale: float = 1.0) -> dict:
    """Same topology at a given length scale; lengths multiply by ``scale``."""
    return {
        "total_length": 100.0 * scale,
        "length_unit": length_unit,
        "mean_branch_length": 20.0 * scale,
        "median_branch_length": 18.0 * scale,
        "n_branches": 5,
        "n_endpoints": 4,
        "n_junctions": 2,
        "n_components": 1,
        "n_terminal_branches": 3,
        "n_internal_branches": 2,
        "bbox_scaled": (40.0 * scale, 30.0 * scale, 0.0),
        "skeleton_volume_occupancy": 0.05,
    }


def test_invariant_features_identical_across_units() -> None:
    # the M1 guard: the same shape at pixel scale vs 0.5 um/px must yield
    # identical scale-invariant features
    fv_px = extract_feature_vector(_descriptor_dict("pixels", scale=1.0))
    fv_um = extract_feature_vector(_descriptor_dict("um", scale=0.5))

    assert fv_px["units_physical"] is False
    assert fv_um["units_physical"] is True
    assert fv_px["invariant_keys"] == fv_um["invariant_keys"]
    for key in fv_px["invariant_keys"]:
        assert fv_px["features"][key] == pytest.approx(fv_um["features"][key]), key


def test_absolute_features_gated_on_physical_units() -> None:
    fv_px = extract_feature_vector(_descriptor_dict("pixels"))
    fv_um = extract_feature_vector(_descriptor_dict("um"))

    for absolute in ("total_length_um", "mean_branch_length_um", "bbox_diagonal_um"):
        assert absolute not in fv_px["features"]
        assert absolute in fv_um["features"]
    assert fv_um["features"]["bbox_diagonal_um"] == pytest.approx((40**2 + 30**2) ** 0.5)


def test_feature_vector_from_real_descriptors(labeled_morphology_samples) -> None:
    for sample in labeled_morphology_samples:
        fv = extract_feature_vector(sample["descriptors"])
        assert set(fv["invariant_keys"]).issubset(fv["features"])
        assert all(np.isfinite(v) for v in fv["features"].values())


# --------------------------------------------------------------------------- #
# N2: reference library I/O
# --------------------------------------------------------------------------- #
def test_reference_library_round_trip(labeled_morphology_samples, tmp_path) -> None:
    lib_path = tmp_path / "refs.csv"
    for sample in labeled_morphology_samples:
        fv = extract_feature_vector(sample["descriptors"])
        append_reference(lib_path, fv, label=sample["label"], name=sample["name"])

    lib = load_reference_library(lib_path)
    assert len(lib) == 3
    assert set(lib.labels) == {"linear", "branched", "bushy"}
    assert "n_branches" in lib.feature_columns


def test_load_missing_library_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_reference_library(tmp_path / "nope.csv")


def test_load_empty_library_raises(tmp_path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("name,label,n_branches\n")  # header only, no rows
    with pytest.raises(ValueError):
        load_reference_library(p)


def test_load_library_requires_label_column(tmp_path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("name,n_branches\nfoo,3\n")
    with pytest.raises(ValueError, match="label"):
        load_reference_library(p)


def test_append_records_units_and_detects_mixed(tmp_path) -> None:
    lib_path = tmp_path / "mixed.csv"
    append_reference(
        lib_path, extract_feature_vector(_descriptor_dict("um")), label="a", name="n1"
    )
    assert load_reference_library(lib_path).all_physical is True

    append_reference(
        lib_path, extract_feature_vector(_descriptor_dict("pixels")), label="b", name="n2"
    )
    assert load_reference_library(lib_path).all_physical is False


# --------------------------------------------------------------------------- #
# N3: matcher core
# --------------------------------------------------------------------------- #
def test_match_classifies_held_out_branched(
    labeled_morphology_samples, viewer, tmp_path
) -> None:
    lib_path = tmp_path / "refs.csv"
    for sample in labeled_morphology_samples:
        append_reference(
            lib_path,
            extract_feature_vector(sample["descriptors"]),
            label=sample["label"],
            name=sample["name"],
        )
    library = load_reference_library(lib_path)

    # a held-out Y-shape (not in the library) should classify as branched
    viewer.add_labels(_branched_variant_mask(), name="q_branch")
    qid = trace.skeletonize("q_branch")["skeleton_id"]
    query_fv = extract_feature_vector(trace.compute_morphology_descriptors(qid))

    res = match_against_library(query_fv, library, k=3)
    assert res["status"] == "ok"
    assert res["predicted"] == "branched"
    assert res["ranked"][0]["label"] == "branched"


def test_match_single_row_library_no_div_by_zero(tmp_path) -> None:
    lib_path = tmp_path / "one.csv"
    append_reference(
        lib_path, extract_feature_vector(_descriptor_dict("pixels")), label="solo", name="n1"
    )
    library = load_reference_library(lib_path)

    res = match_against_library(extract_feature_vector(_descriptor_dict("pixels")), library, k=5)
    assert res["status"] == "ok"
    assert res["predicted"] == "solo"
    assert 0.0 <= res["confidence"] <= 1.0


def test_match_falls_back_to_invariant_on_mixed_units(tmp_path) -> None:
    lib_path = tmp_path / "phys.csv"
    append_reference(
        lib_path, extract_feature_vector(_descriptor_dict("um")), label="a", name="n1"
    )
    append_reference(
        lib_path, extract_feature_vector(_descriptor_dict("um", scale=3.0)), label="b", name="n2"
    )
    library = load_reference_library(lib_path)
    assert library.all_physical is True

    res = match_against_library(extract_feature_vector(_descriptor_dict("pixels")), library, k=2)
    assert res["status"] == "ok"
    assert res["invariant_only"] is True
    # absolute (micron) features end with "_um"; none should be used here
    assert not any(c.endswith("_um") for c in res["features_used"])
