from __future__ import annotations

from pathlib import Path
from typing import Any

from imajin import session as state
from imajin.analysis.morphology_features import extract_feature_vector
from imajin.analysis.morphology_match import match_against_library
from imajin.analysis.morphology_reference import append_reference, load_reference_library
from imajin.paths import normalize_user_path
from imajin.tools._trace_export import _write_swc
from imajin.tools._trace_store import _entry
from imajin.tools._trace_tables import _branch_summary
from imajin.tools.registry import tool
# Cross-family: _skeleton_feature_vector calls compute_morphology_descriptors.
# Import it from the morphometry SUBMODULE (not the package, which is mid-import).
from imajin.tools.trace.morphometry import compute_morphology_descriptors


def _reference_library_path(reference: str) -> Path:
    """Resolve a reference-library path. 'default' → <results_root>/morphology_reference.csv."""
    if reference.strip() in ("", "default"):
        from imajin.results import results_root

        return results_root() / "morphology_reference.csv"
    return normalize_user_path(reference)


def _load_reference_or_none(reference: str):
    """Load the reference library, or None if it is missing/empty/malformed."""
    path = _reference_library_path(reference)
    if not path.exists():
        return None
    try:
        return load_reference_library(path)
    except (FileNotFoundError, ValueError):
        return None


def _add_persistence_features(feature_vector: dict[str, Any], entry: Any) -> None:
    """Enrich the feature vector with navis persistence features, when available.

    Persistence is rotation/translation-invariant but scale-sensitive, so the
    features are added to ``features`` but NOT to ``invariant_keys`` — the matcher
    uses them only when query and library share physical (micron) units. A no-op
    without the connectome extra or on a degenerate skeleton.
    """
    import tempfile

    from imajin.analysis.morphology_persistence import (
        persistence_available,
        persistence_features_from_swc,
    )

    if not persistence_available():
        return
    with tempfile.TemporaryDirectory() as tmp:
        swc = Path(tmp) / "skeleton.swc"
        _write_swc(entry, swc)
        features = persistence_features_from_swc(swc)
    if features:
        feature_vector["features"].update(features)


def _skeleton_feature_vector(skeleton_id: str) -> dict[str, Any]:
    """Descriptor → feature vector for one registered skeleton (incl. tortuosity)."""
    entry = _entry(skeleton_id)
    descriptors = compute_morphology_descriptors(skeleton_id)
    branch_df = _branch_summary(entry.skel, entry.record.spacing)
    fv = extract_feature_vector(descriptors, branch_df)
    _add_persistence_features(fv, entry)
    return fv


@tool(
    description="Classify a skeleton's neuron type by morphometric similarity to a "
    "labelled reference library (build one with add_reference_neuron). Local, "
    "offline, registration-free. Returns status 'no_reference' when no library is "
    "configured. (Spatial NBLAST / connectome lookup is a separate Tier-2 backend.)",
    phase="6B",
    subagent="neural_tracer",
)
def classify_neuron_type(
    skeleton_id: str,
    reference: str = "default",
) -> dict[str, Any]:
    # H3: resolve the reference library BEFORE touching the skeleton registry, so a
    # missing library returns a graceful status rather than KeyError on a bad id.
    library = _load_reference_or_none(reference)
    if library is None:
        return {
            "skeleton_id": skeleton_id,
            "reference": reference,
            "predicted_type": None,
            "confidence": None,
            "status": "no_reference",
            "note": (
                "No morphology reference library found. Build one by labelling your "
                "own traced neurons: add_reference_neuron(skeleton_id, label)."
            ),
        }

    fv = _skeleton_feature_vector(skeleton_id)
    res = match_against_library(fv, library, k=5)
    runner_up = res["ranked"][1]["label"] if len(res["ranked"]) > 1 else None

    # H2: distinct QC key — do NOT reuse the bare skeleton_id, which holds the
    # neural_morphology record written by compute_morphology_descriptors.
    state.put_qc_record(
        f"{skeleton_id}::classification",
        status="pass",
        metrics={
            "kind": "neural_classification",
            "predicted_type": res["predicted"],
            "confidence": res["confidence"],
            "invariant_only": res["invariant_only"],
        },
    )
    return {
        "skeleton_id": skeleton_id,
        "reference": reference,
        "predicted_type": res["predicted"],
        "confidence": res["confidence"],
        "runner_up": runner_up,
        "ranked": res["ranked"],
        "invariant_only": res["invariant_only"],
        "status": res["status"],
        "note": "Morphometric (feature-vector) match — registration-free, local.",
    }


@tool(
    description="Add the current skeleton to a labelled morphology reference library "
    "(CSV) so future neurons can be classified against it. Builds the library from "
    "your own traced + labelled neurons; fully local/offline.",
    phase="6B",
    subagent="neural_tracer",
)
def add_reference_neuron(
    skeleton_id: str,
    label: str,
    library_path: str = "default",
) -> dict[str, Any]:
    # adding a reference requires a real skeleton, so the lookup (KeyError on a bad
    # id) is the correct behaviour here
    fv = _skeleton_feature_vector(skeleton_id)
    path = _reference_library_path(library_path)
    library = append_reference(path, fv, label=label.strip(), name=skeleton_id)
    return {
        "skeleton_id": skeleton_id,
        "label": label.strip(),
        "library_path": str(path),
        "n_references": len(library),
        "units_physical": fv["units_physical"],
        "status": "ok",
    }


@tool(
    description="Find the k most morphometrically similar neurons in a labelled "
    "reference library (local, offline, registration-free). Returns status "
    "'no_reference' when no library is configured. (External connectome lookup is "
    "query_connectome, a separate Tier-2 backend.)",
    phase="6B",
    subagent="neural_tracer",
)
def find_similar_neurons(
    skeleton_id: str,
    reference: str = "default",
    k: int = 10,
) -> dict[str, Any]:
    # H3 ordering: reference first, skeleton lookup only when a library exists
    library = _load_reference_or_none(reference)
    if library is None:
        return {
            "skeleton_id": skeleton_id,
            "reference": reference,
            "matches": [],
            "status": "no_reference",
            "note": (
                "No morphology reference library found. Build one with "
                "add_reference_neuron(skeleton_id, label)."
            ),
        }

    fv = _skeleton_feature_vector(skeleton_id)
    res = match_against_library(fv, library, k=k)
    return {
        "skeleton_id": skeleton_id,
        "reference": reference,
        "matches": res["ranked"],
        "invariant_only": res["invariant_only"],
        "status": res["status"],
    }
