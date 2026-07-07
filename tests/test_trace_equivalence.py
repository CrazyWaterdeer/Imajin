"""Characterization / equivalence guard for the trace tool-module split.

Safety net for the trace.py -> trace/ package split (same template as the segment
split, issue #3). trace.py is a pure @tool wrapper layer, so this pins the
*contract* the split must not change:

* the registry entry (flags), ``inspect.signature`` and json-schema ``required``
  for every tool -- all are ``subagent="neural_tracer"`` specialist tools, so a
  silent flag flip would drop them from the neural-tracer agent's tool set;
* the public + re-exported surface on ``imajin.tools.trace`` (the tools plus
  ``reset_skeletons`` / ``_entry`` / ``list_trace_records`` -- the last is a
  ``report.py`` source import, not just a test read);
* that all of them stay in ``tools_for_anthropic("neural_tracer")``;
* a skeletonize -> analyze pipeline smoke (the existing test_tools_trace /
  test_tools_morphology suites pin the numeric detail; this guards that the moved
  tools still run end-to-end and the no-op statuses are unchanged).

Asserts observable contract only -- never that a helper lives at a path (that
moves in the split).
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from imajin.tools import trace
from imajin.tools.registry import get_tool, tools_for_anthropic

TOOLS = [
    "enhance_neural_processes",
    "segment_neural_processes",
    "skeletonize",
    "extract_branch_metrics",
    "prune_skeleton",
    "set_branch_qc",
    "set_soma_location",
    "assign_neural_region",
    "compute_sholl_analysis",
    "compute_morphology_descriptors",
    "export_neural_trace",
    "propose_filament_bridges",
    "build_rooted_tree",
    "measure_filament_diameter",
    "compute_tree_topology",
    "query_connectome",
    "classify_neuron_type",
    "add_reference_neuron",
    "find_similar_neurons",
]

# Only these run on a worker thread; the rest are worker=False. (All are
# phase="6B", subagent="neural_tracer", manual=False, llm=True.)
WORKER_TRUE = {
    "enhance_neural_processes",
    "segment_neural_processes",
    "skeletonize",
    "prune_skeleton",
}

# (signature string, sorted json-schema required) per tool.
SIG_GOLDEN: dict[str, tuple[str, list[str]]] = {
    "enhance_neural_processes": ("(layer: 'str', method: 'str' = 'tubeness', sigma: 'float | tuple[float, ...] | None' = None, background: 'str | None' = 'rolling_ball', normalize: 'bool' = True) -> 'dict[str, Any]'", ["layer"]),
    "segment_neural_processes": ("(layer: 'str', threshold: 'str | float' = 'otsu', min_size_um3: 'float | None' = None, fill_holes: 'bool' = False, keep_largest: 'bool' = False) -> 'dict[str, Any]'", ["layer"]),
    "skeletonize": ("(layer: 'str', min_branch_length: 'float' = 0.0, threshold: 'float | None' = None) -> 'dict[str, Any]'", ["layer"]),
    "extract_branch_metrics": ("(skeleton_id: 'str') -> 'dict[str, Any]'", ["skeleton_id"]),
    "prune_skeleton": ("(skeleton_id: 'str', min_branch_length_um: 'float', remove_isolated: 'bool' = True) -> 'dict[str, Any]'", ["min_branch_length_um", "skeleton_id"]),
    "set_branch_qc": ("(skeleton_id: 'str', branch_ids: 'list[int]', status: 'str', reason: 'str | None' = None) -> 'dict[str, Any]'", ["branch_ids", "skeleton_id", "status"]),
    "set_soma_location": ("(skeleton_id: 'str', point_layer: 'str | None' = None, mask_layer: 'str | None' = None) -> 'dict[str, Any]'", ["skeleton_id"]),
    "assign_neural_region": ("(skeleton_id: 'str', region_layer: 'str') -> 'dict[str, Any]'", ["region_layer", "skeleton_id"]),
    "compute_sholl_analysis": ("(skeleton_id: 'str', center: 'str' = 'soma', radius_step_um: 'float' = 5.0, max_radius_um: 'float | None' = None) -> 'dict[str, Any]'", ["skeleton_id"]),
    "compute_morphology_descriptors": ("(skeleton_id: 'str') -> 'dict[str, Any]'", ["skeleton_id"]),
    "export_neural_trace": ("(skeleton_id: 'str', output_path: 'str', format: 'str' = 'swc') -> 'dict[str, Any]'", ["output_path", "skeleton_id"]),
    "propose_filament_bridges": ("(skeleton_id: 'str', max_gap_um: 'float', support_layer: 'str | None' = None, min_support: 'float' = 0.2, max_tangent_angle_deg: 'float' = 60.0) -> 'dict[str, Any]'", ["max_gap_um", "skeleton_id"]),
    "build_rooted_tree": ("(skeleton_id: 'str', apply_bridges: 'bool' = True) -> 'dict[str, Any]'", ["skeleton_id"]),
    "measure_filament_diameter": ("(skeleton_id: 'str', mask_layer: 'str | None' = None) -> 'dict[str, Any]'", ["skeleton_id"]),
    "compute_tree_topology": ("(skeleton_id: 'str') -> 'dict[str, Any]'", ["skeleton_id"]),
    "query_connectome": ("(skeleton_id: 'str', db: 'str' = 'neuprint', k: 'int' = 10) -> 'dict[str, Any]'", ["skeleton_id"]),
    "classify_neuron_type": ("(skeleton_id: 'str', reference: 'str' = 'default') -> 'dict[str, Any]'", ["skeleton_id"]),
    "add_reference_neuron": ("(skeleton_id: 'str', label: 'str', library_path: 'str' = 'default') -> 'dict[str, Any]'", ["label", "skeleton_id"]),
    "find_similar_neurons": ("(skeleton_id: 'str', reference: 'str' = 'default', k: 'int' = 10) -> 'dict[str, Any]'", ["skeleton_id"]),
}


@pytest.mark.parametrize("name", TOOLS)
def test_registry_signature_and_schema_golden(name: str) -> None:
    e = get_tool(name)
    assert e.phase == "6B"
    assert e.subagent == "neural_tracer"
    assert e.manual is False
    assert e.llm is True
    assert e.worker is (name in WORKER_TRUE)
    sig, required = SIG_GOLDEN[name]
    assert str(inspect.signature(e.func)) == sig
    schema = e.input_model.model_json_schema()
    assert sorted(schema.get("required", [])) == sorted(required)


def test_public_and_reexported_surface() -> None:
    for name in TOOLS:
        assert getattr(trace, name) is get_tool(name).func
    # Re-exported names read via the module: reset_skeletons / _entry (tests),
    # list_trace_records (report.py source import).
    for alias in ("reset_skeletons", "_entry", "list_trace_records"):
        assert hasattr(trace, alias), alias


def test_all_neural_tracer_tools_in_set() -> None:
    names = {t["name"] for t in tools_for_anthropic(subagent="neural_tracer")}
    assert set(TOOLS) <= names
    # TOOLS must list exactly the neural_tracer tools — no accidental add/drop.
    assert names == set(TOOLS)


def _branched_mask() -> np.ndarray:
    """Y-shape: vertical trunk + two diagonal branches (from test_tools_trace)."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:55, 31:33] = 1
    for i in range(20):
        img[10 + i, 31 - i] = 1
        img[10 + i, 32 + i] = 1
    return img


def test_pipeline_smoke_end_to_end(viewer, tmp_path) -> None:
    """skeletonize -> analyze end-to-end: proves the moved tools still run and the
    degraded (no-backend / no-reference) statuses are unchanged. Numeric detail is
    pinned by test_tools_trace / test_tools_morphology."""
    trace.reset_skeletons()
    viewer.add_labels(_branched_mask(), name="ymask")

    skel = trace.skeletonize("ymask")
    sid = skel["skeleton_id"]
    assert sid.startswith("skel_")
    assert skel["n_paths"] >= 3

    assert trace.extract_branch_metrics(sid)["n_branches"] >= 3
    assert "descriptors" in trace.compute_morphology_descriptors(sid) or \
        trace.compute_morphology_descriptors(sid)  # non-empty dict
    assert trace.compute_sholl_analysis(sid)["skeleton_id"] == sid

    swc = tmp_path / "trace.swc"
    exp = trace.export_neural_trace(sid, str(swc), format="swc")
    assert swc.exists() and exp.get("format") == "swc"

    # Optional-extra tools degrade to typed statuses (deterministic without the
    # connectome extra / a reference library).
    assert trace.query_connectome(sid)["status"] in {
        "backend_unavailable", "needs_token", "needs_registration"
    }
    assert trace.classify_neuron_type(sid, reference=str(tmp_path / "none.csv"))["status"] == "no_reference"
    assert trace.find_similar_neurons(sid, reference=str(tmp_path / "none.csv"))["status"] == "no_reference"

    # _entry (re-exported) resolves the registered skeleton.
    assert trace._entry(sid).record is not None
