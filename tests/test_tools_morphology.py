"""Characterization net for neural morphology matching (Tier 1).

Pins (a) the current stub contract of ``classify_neuron_type`` /
``query_connectome`` so the change in N4/N6 is observable, and (b) the
``compute_morphology_descriptors`` output contract that the Tier-1 feature
extractor (N1) consumes. Also provides the labelled toy-skeleton fixture that the
feature / reference-library / classifier commits reuse.

Everything here passes on the current code with only existing dependencies.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from imajin import session as state
from imajin.analysis import morphology_nblast
from imajin.analysis.morphology_features import extract_feature_vector
from imajin.analysis.morphology_match import match_against_library
from imajin.analysis.morphology_nblast import nblast_against_references
from imajin.analysis.morphology_reference import (
    append_reference,
    load_reference_library,
)
from imajin.tools import trace


requires_navis = pytest.mark.skipif(
    importlib.util.find_spec("navis") is None,
    reason="navis not installed (uv sync --extra connectome)",
)


def _um_points(viewer, mask, name, scale=(0.4, 0.4)):
    """Skeletonize a mask at a physical (micron) scale; return (points_um, units)."""
    viewer.add_labels(mask, name=name, scale=scale)
    skel_id = trace.skeletonize(name)["skeleton_id"]
    entry = trace._entry(skel_id)
    points = np.asarray(entry.skel.coordinates, dtype=float) * np.asarray(entry.record.spacing)
    return points, entry.record.units


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
def test_classify_neuron_type_no_reference_is_graceful(viewer, tmp_path) -> None:
    # missing library ⇒ no_reference, and (H3) no KeyError on a bogus skeleton id
    res = trace.classify_neuron_type("any_id", reference=str(tmp_path / "none.csv"))
    assert res["status"] == "no_reference"
    assert res["predicted_type"] is None


def test_query_connectome_neuprint_degrades(viewer) -> None:
    res = trace.query_connectome("any_id", db="neuprint", k=5)
    assert res["status"] in {"backend_unavailable", "needs_token", "needs_registration"}
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


# --------------------------------------------------------------------------- #
# N4: classify_neuron_type tool (real path + H2 distinct QC key)
# --------------------------------------------------------------------------- #
def test_classify_neuron_type_real_and_distinct_qc(
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

    viewer.add_labels(_branched_variant_mask(), name="q_classify")
    qid = trace.skeletonize("q_classify")["skeleton_id"]
    res = trace.classify_neuron_type(qid, reference=str(lib_path))

    assert res["status"] == "ok"
    assert res["predicted_type"] == "branched"

    # H2: classification QC under a distinct key; morphology QC under the bare id
    assert state.get_qc_record(f"{qid}::classification").metrics["kind"] == "neural_classification"
    assert state.get_qc_record(qid).metrics["kind"] == "neural_morphology"


# --------------------------------------------------------------------------- #
# N5: add_reference_neuron + find_similar_neurons (full offline loop)
# --------------------------------------------------------------------------- #
def test_build_library_and_find_similar_round_trip(viewer, tmp_path) -> None:
    trace.reset_skeletons()
    lib_path = str(tmp_path / "lib.csv")

    out = None
    for label, mask, name in [
        ("linear", _straight_mask(), "r_lin"),
        ("branched", _branched_mask(), "r_br"),
        ("bushy", _bushy_mask(), "r_bush"),
    ]:
        viewer.add_labels(mask, name=name)
        sid = trace.skeletonize(name)["skeleton_id"]
        out = trace.add_reference_neuron(sid, label=label, library_path=lib_path)
        assert out["status"] == "ok"
    assert out["n_references"] == 3

    viewer.add_labels(_branched_variant_mask(), name="q_find")
    qid = trace.skeletonize("q_find")["skeleton_id"]
    res = trace.find_similar_neurons(qid, reference=lib_path, k=3)

    assert res["status"] == "ok"
    assert res["matches"][0]["label"] == "branched"


def test_find_similar_no_reference_is_graceful(viewer, tmp_path) -> None:
    res = trace.find_similar_neurons("any_id", reference=str(tmp_path / "none.csv"))
    assert res["status"] == "no_reference"
    assert res["matches"] == []


# --------------------------------------------------------------------------- #
# N6: specialist exposure + query_connectome honesty
# --------------------------------------------------------------------------- #
def test_specialist_exposes_new_morphology_tools() -> None:
    from imajin.tools.registry import tools_for_anthropic

    names = {t["name"] for t in tools_for_anthropic("neural_tracer")}
    assert {"add_reference_neuron", "find_similar_neurons", "classify_neuron_type"} <= names


def test_query_connectome_rejects_mouse_databases(viewer) -> None:
    for db in ("microns", "allen"):
        res = trace.query_connectome("any_id", db=db)
        assert res["status"] == "off_domain"
    # neuprint is a real Drosophila DB → a backend/credential degradation, never off_domain
    assert trace.query_connectome("any_id", db="neuprint")["status"] != "off_domain"


def test_specialist_prompt_advertises_local_classification() -> None:
    from imajin.agent.specialists.neural_tracer import NEURAL_TRACER_PROMPT

    assert "add_reference_neuron" in NEURAL_TRACER_PROMPT
    assert "stubbed for now" not in NEURAL_TRACER_PROMPT


# --------------------------------------------------------------------------- #
# Tier 2: NBLAST adapter (navis optional; gated/degraded)
# --------------------------------------------------------------------------- #
def test_nblast_backend_unavailable_is_graceful(monkeypatch) -> None:
    # forced absent backend ⇒ typed status, never an exception (runs without navis)
    monkeypatch.setattr(morphology_nblast, "navis_available", lambda: False)
    res = morphology_nblast.nblast_against_references(
        np.zeros((5, 3)), ("um", "um", "um"), []
    )
    assert res["status"] == "backend_unavailable"


@requires_navis
def test_nblast_refuses_pixel_scale_data() -> None:
    # NBLAST is micron-calibrated; pixel-scale (units None) must be refused
    res = nblast_against_references(
        np.zeros((10, 3)),
        None,
        [{"name": "r", "label": "x", "points": np.zeros((10, 3)), "units": ("um",)}],
    )
    assert res["status"] == "needs_microns"


@requires_navis
def test_nblast_self_match_ranks_highest(viewer) -> None:
    trace.reset_skeletons()
    pts_branched, units = _um_points(viewer, _branched_mask(), "nb_branched")
    pts_linear, _ = _um_points(viewer, _straight_mask(), "nb_linear")
    references = [
        {"name": "branched", "label": "branched", "points": pts_branched, "units": units},
        {"name": "linear", "label": "linear", "points": pts_linear, "units": units},
    ]

    res = nblast_against_references(pts_branched, units, references, k=2)

    assert res["status"] == "ok"
    # the co-located self (branched) must score highest
    assert res["ranked"][0]["name"] == "branched"


# --------------------------------------------------------------------------- #
# Tier 2: neuPrint backend readiness (credential-gated, deterministic via patch)
# --------------------------------------------------------------------------- #
def test_neuprint_backend_unavailable(monkeypatch) -> None:
    from imajin.analysis import connectome_neuprint as cn

    monkeypatch.setattr(cn, "navis_available", lambda: False)
    res = cn.query_neuprint(None, None)
    assert res["status"] == "backend_unavailable"


def test_neuprint_needs_token_when_backend_present(monkeypatch) -> None:
    from imajin.analysis import connectome_neuprint as cn

    monkeypatch.setattr(cn, "navis_available", lambda: True)
    monkeypatch.setattr(cn, "neuprint_available", lambda: True)
    monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
    res = cn.query_neuprint(None, None)
    assert res["status"] == "needs_token"


def test_neuprint_with_token_needs_registration(monkeypatch) -> None:
    from imajin.analysis import connectome_neuprint as cn

    monkeypatch.setattr(cn, "navis_available", lambda: True)
    monkeypatch.setattr(cn, "neuprint_available", lambda: True)
    monkeypatch.setenv("NEUPRINT_APPLICATION_CREDENTIALS", "fake-token")
    res = cn.query_neuprint(None, None)
    assert res["status"] == "needs_registration"


def test_query_connectome_routes_neuprint_without_skeleton_lookup(viewer, monkeypatch) -> None:
    from imajin.analysis import connectome_neuprint as cn

    monkeypatch.setattr(cn, "navis_available", lambda: False)
    # bogus skeleton id must not raise — readiness is resolved before any lookup
    res = trace.query_connectome("any_id", db="neuprint")
    assert res["db"] == "neuprint"
    assert res["status"] == "backend_unavailable"


# --------------------------------------------------------------------------- #
# Option B: topological persistence features (registration-free)
# --------------------------------------------------------------------------- #
def test_persistence_unavailable_returns_none(monkeypatch, tmp_path) -> None:
    import imajin.analysis.morphology_persistence as mp

    monkeypatch.setattr(mp, "navis_available", lambda: False)
    assert mp.persistence_features_from_swc(tmp_path / "x.swc") is None


@requires_navis
def test_persistence_features_from_real_skeleton(viewer, tmp_path) -> None:
    from imajin.analysis.morphology_persistence import persistence_features_from_swc
    from imajin.tools._trace_export import _write_swc

    trace.reset_skeletons()
    viewer.add_labels(_branched_mask(), name="p_branch", scale=(0.4, 0.4))
    skel_id = trace.skeletonize("p_branch")["skeleton_id"]
    swc = tmp_path / "p_branch.swc"
    _write_swc(trace._entry(skel_id), swc)

    feats = persistence_features_from_swc(swc, samples=32)
    assert feats is not None
    assert len(feats) == 32
    assert all(k.startswith("pers_") for k in feats)
    assert all(np.isfinite(v) for v in feats.values())


@requires_navis
def test_persistence_is_translation_rotation_invariant(tmp_path) -> None:
    # the headline property: persistence is invariant to rigid motion (NBLAST is not)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import navis

        from imajin.analysis.morphology_persistence import persistence_features_from_swc

        base = navis.example_neurons(1, kind="skeleton")
        moved = base.copy()
        theta = 0.7
        rot = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )
        coords = moved.nodes[["x", "y", "z"]].to_numpy() @ rot.T + np.array([1e4, -5e3, 2e3])
        moved.nodes[["x", "y", "z"]] = coords

        navis.write_swc(base, tmp_path / "base.swc")
        navis.write_swc(moved, tmp_path / "moved.swc")

    f_base = persistence_features_from_swc(tmp_path / "base.swc", samples=48)
    f_moved = persistence_features_from_swc(tmp_path / "moved.swc", samples=48)
    assert f_base is not None and f_moved is not None
    for key in f_base:
        assert f_base[key] == pytest.approx(f_moved[key], abs=1e-6)


@requires_navis
def test_persistence_enriches_physical_matching(viewer, tmp_path) -> None:
    trace.reset_skeletons()
    lib_path = str(tmp_path / "lib_um.csv")
    for label, mask, name in [
        ("branched", _branched_mask(), "pe_br"),
        ("linear", _straight_mask(), "pe_lin"),
    ]:
        viewer.add_labels(mask, name=name, scale=(0.4, 0.4))  # physical → microns
        sid = trace.skeletonize(name)["skeleton_id"]
        trace.add_reference_neuron(sid, label=label, library_path=lib_path)

    # the library now carries persistence columns (computed because navis is present)
    library = load_reference_library(lib_path)
    assert any(c.startswith("pers_") for c in library.feature_columns)

    viewer.add_labels(_branched_variant_mask(), name="pe_q", scale=(0.4, 0.4))
    qid = trace.skeletonize("pe_q")["skeleton_id"]
    res = trace.classify_neuron_type(qid, reference=lib_path)

    assert res["status"] == "ok"
    # both query and library are micron-scale ⇒ persistence + absolute features used
    assert res["invariant_only"] is False
    assert res["predicted_type"] == "branched"
