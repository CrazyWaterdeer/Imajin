"""Neural-morphology tools, split by family into submodules (GitHub issue #3).

The 15 ``@tool`` functions (all ``subagent="neural_tracer"``) live in the family
submodules (``enhance``, ``skeleton``, ``morphometry``, ``export``, ``connectome``,
``classify``); the numeric cores are in ``analysis/morphology_*`` and the
``tools/_trace_*`` helper modules. Importing the submodules here registers the tools
and re-exports their names on ``imajin.tools.trace`` so the public path is unchanged.
"""
from __future__ import annotations

# Public tools -- importing each submodule registers it via @tool and re-exports
# the tool name on ``imajin.tools.trace``.
from imajin.tools.trace.enhance import (
    enhance_neural_processes,
    segment_neural_processes,
)
from imajin.tools.trace.skeleton import (
    assign_neural_region,
    extract_branch_metrics,
    prune_skeleton,
    set_branch_qc,
    set_soma_location,
    skeletonize,
)
from imajin.tools.trace.morphometry import (
    compute_morphology_descriptors,
    compute_sholl_analysis,
)
from imajin.tools.trace.export import export_neural_trace
from imajin.tools.trace.tracer import (
    build_rooted_tree,
    propose_filament_bridges,
)
from imajin.tools.trace.connectome import query_connectome
from imajin.tools.trace.classify import (
    add_reference_neuron,
    classify_neuron_type,
    find_similar_neurons,
)

# Re-exported names read via ``imajin.tools.trace.<name>``: reset_skeletons / _entry
# (tests) and list_trace_records (a report.py source import).
from imajin.tools._trace_store import (  # noqa: F401
    _entry,
    list_trace_records,
    reset_skeletons,
)

__all__ = [
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
    "query_connectome",
    "classify_neuron_type",
    "add_reference_neuron",
    "find_similar_neurons",
]
