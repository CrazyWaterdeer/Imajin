"""The 20-tool core set advertised to LOCAL (Ollama) models only.

Measured on this machine, and stated at the size it actually reproduces. The
fully-reproduced win is VRAM: qwen3.5:9b at the full set's num_ctx of 57,344
needs 10.58 GB against a ~9.96 GB usable budget on a 10 GB card and spills to
76% GPU at 33 tok/s; with this core set num_ctx falls to 24,576, the model is
9.37 GB and 100% resident, and the spill is gone.

The decision-loop effect is real but smaller than first reported. Across four
harness runs (rich and opaque canned tool results, 8/10/14-step budgets, 3
prompts) the 107-tool condition terminated on 2 of 3 prompts and detoured
through 3-4 non-core orientation tools per run; the same model on these 20
terminated 3/3, in 2-3 fewer steps. An earlier note here claimed 3/3
non-termination with a tight segment -> list_layers loop -- that did NOT
reproduce under repeat testing and has been corrected. What does hold is that
the wasted calls go to tools this set removes (list_sample_annotations,
get_batch_progress, list_experiment, list_registered_files).

Cloud backends are unaffected -- Anthropic, OpenAI, the claude-agent and
codex-agent subscription paths all still advertise the full registry, unchanged.
Only the local-model wiring (imajin.ui.chat_dock's ollama branch) reaches for
this module.

The 20 names are exactly the tools the system prompt's own named pipelines call
by name: list_layers first (always the first move), then
segmentation/correction, then measurement/analysis, then plotting/comparison
and output, plus the onboarding/registration/channel-resolution helpers those
pipelines lean on. CORE_TOOL_NAMES' order is that pipeline order, not
alphabetical -- core_tools() below preserves it regardless of the order tools
happen to be registered or iterated in.

Deliberately dependency-free: no `imajin.tools` import at module scope, so this
stays cheap and side-effect-free to import from a UI module (imajin.tools pulls
in every tool module -- torch, cellpose, the works). A test
(tests/test_tool_subset.py), not this module, checks these 20 names against the
live registry, so a tool rename fails loudly in CI instead of silently shrinking
the local-model core set.
"""
from __future__ import annotations

from typing import Any

CORE_TOOL_NAMES: tuple[str, ...] = (
    "list_layers",
    "load_file",
    "rolling_ball_background",
    "segment_target_objects",
    "auto_segment_target",
    "correct_roi",
    "analyze_target_cells",
    "measure_intensity",
    "plot_group_distribution",
    "save_result_bundle",
    "summarize_table",
    "get_help",
    "register_files",
    "auto_contrast",
    "gaussian_denoise",
    "segment_3d_cells_auto",
    "cellpose_sam",
    "measure_projected_intensity",
    "compare_groups",
    "resolve_channel",
)


def core_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter an Anthropic-style tool spec list down to CORE_TOOL_NAMES, in
    CORE_TOOL_NAMES' pipeline order -- not `tools`' order, so the tool list a
    local model sees reads the same way regardless of the live registry's own
    (insertion-order-dependent) iteration order.

    A name in CORE_TOOL_NAMES that is absent from `tools` is skipped rather
    than raised: a KeyError here, at chat-dock provider-construction time,
    would take down the whole local-model path over a naming drift that
    missing_core_names() below is what's meant to catch -- loudly, in a test
    and a startup warning, not as a crash.
    """
    by_name = {t.get("name"): t for t in tools}
    return [by_name[name] for name in CORE_TOOL_NAMES if name in by_name]


def missing_core_names(tools: list[dict[str, Any]]) -> list[str]:
    """CORE_TOOL_NAMES entries not present in `tools`, in CORE_TOOL_NAMES order.

    Empty against the live registry in normal operation -- asserted directly in
    tests/test_tool_subset.py. Non-empty means a core tool was renamed or
    removed without updating this module; chat_dock surfaces that as a startup
    warning (fewer than 20 tools reaching a local model) instead of only a
    quietly shorter core set.
    """
    present = {t.get("name") for t in tools}
    return [name for name in CORE_TOOL_NAMES if name not in present]
