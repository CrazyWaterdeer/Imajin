from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imajin.analysis.morphology_features import extract_feature_vector
from imajin.analysis.morphology_match import match_against_library
from imajin.analysis.morphology_reference import append_reference, load_reference_library
from imajin import session as state
from imajin.agent.qt_dispatch import call_on_main
from imajin.session import get_layer
from imajin.paths import normalize_user_path
from imajin.tools._trace_export import _swc_coordinates, _write_swc
from imajin.tools._trace_image import (
    _binary_from_layer_data,
    _component_labels,
    _materialize,
    _normalize_image,
    _rolling_ball_subtract,
)
from imajin.tools._trace_store import (
    _BRANCH_QC_STATUSES,
    _SKELETON_REGISTRY,
    _SkeletonEntry,
    _TRACE_STATUSES,
    _entry,
    _register_skeleton,
    _store_graph_tables,
    NeuralTraceQC,
    NeuralTraceRecord,
    get_skeleton,
    get_trace_record,
    list_trace_records,
    reset_skeletons,
)
from imajin.tools._trace_tables import (
    _BRANCH_TYPES,
    _branch_summary,
    _component_table,
    _edge_table,
    _node_table,
    _normalize_branch_df,
    _put_table,
    _scale_is_physical,
    _scale_tuple,
)
from imajin.tools.napari_ops import (
    add_image_from_worker,
    add_labels_from_worker,
    snapshot_layer,
)
from imajin.tools.registry import tool

# Tool families live in submodules; import them for @tool registration and to
# re-export the public tool names on ``imajin.tools.trace``.
from imajin.tools.trace.enhance import (  # noqa: F401,E402
    enhance_neural_processes,
    segment_neural_processes,
)
from imajin.tools.trace.skeleton import (  # noqa: F401,E402
    assign_neural_region,
    extract_branch_metrics,
    prune_skeleton,
    set_branch_qc,
    set_soma_location,
    skeletonize,
)
from imajin.tools.trace.morphometry import (  # noqa: F401,E402
    compute_morphology_descriptors,
    compute_sholl_analysis,
)
from imajin.tools.trace.export import export_neural_trace  # noqa: F401,E402
from imajin.tools.trace.connectome import query_connectome  # noqa: F401,E402
from imajin.tools.trace.classify import (  # noqa: F401,E402
    add_reference_neuron,
    classify_neuron_type,
    find_similar_neurons,
)
