from __future__ import annotations

from imajin.agent.prompts import SYSTEM_PROMPT
from imajin.agent.prompts import build_system_prompt


def test_system_prompt_guides_bundle_recipe_reuse() -> None:
    assert "import_recipe_from_bundle" in SYSTEM_PROMPT
    assert "recipe_params" in SYSTEM_PROMPT
    assert "file scope" in SYSTEM_PROMPT
    assert "channel roles" in SYSTEM_PROMPT
    assert "run_context" in SYSTEM_PROMPT


# The pinned 20-name local-model core set (see tool_subset.py's CORE_TOOL_NAMES,
# owned by a different slice of the same change). Duplicated here as a literal,
# rather than imported, so this file's tests don't depend on that module landing
# first -- both lists are checked against the same spec, so they must agree.
_CORE_TOOL_NAMES = frozenset(
    {
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
    }
)


def test_build_system_prompt_default_matches_explicit_none() -> None:
    assert build_system_prompt() == build_system_prompt(None)


def test_build_system_prompt_none_is_byte_identical_to_today() -> None:
    # available_tools=None must reproduce the pre-reduction prompt exactly -- every
    # existing caller (chat_dock.py's cloud and local paths alike) relies on this, so
    # a silent drift here would change every cloud backend's behaviour too, not just
    # the local-Ollama path the available_tools parameter exists for.
    result = build_system_prompt(None)
    assert result.startswith(SYSTEM_PROMPT)
    assert "reduced tool set is active" not in result.lower()


def test_build_system_prompt_reduced_drops_non_core_tool_names() -> None:
    # The two tools BOTH test models called despite never being advertised, once the
    # tool list alone was cut to the 20-tool core (see prompts.py's _REDUCTION_SPANS
    # comment) -- plus a broader sample spanning every dropped block, so a future
    # edit that only half-applies the reduction still fails here.
    reduced = build_system_prompt(_CORE_TOOL_NAMES)
    non_core_sample = (
        "list_sample_annotations",
        "list_registered_files",
        "get_batch_progress",
        "advance_to_file",
        "manders_coefficients",
        "track_roi_over_time",
        "annotate_sample",
        "export_table",
    )
    for name in non_core_sample:
        assert name not in reduced, f"{name!r} still named in the reduced prompt"


def test_build_system_prompt_reduced_keeps_core_guidance_and_notice() -> None:
    reduced = build_system_prompt(_CORE_TOOL_NAMES)
    for name in ("list_layers", "segment_target_objects", "measure_intensity", "get_help"):
        assert name in reduced, f"{name!r} (a core tool) missing from the reduced prompt"

    # The notice itself (not just get_help's other, unrelated mentions elsewhere in
    # the prompt) must point at get_help -- sliced against its own known neighbour,
    # the section header prompts.py always places right after it, not a magic length.
    notice_start = reduced.lower().find("reduced tool set is active")
    notice_end = reduced.find("# Bias to action")
    assert notice_start != -1, "reduced-tool-set notice is missing"
    assert notice_end != -1 and notice_start < notice_end
    assert "get_help" in reduced[notice_start:notice_end]


def test_build_system_prompt_reduced_is_not_trivially_short() -> None:
    # A regression that emptied (or nearly emptied) the prompt would ALSO satisfy
    # "no non-core name appears" -- this catches that failure mode specifically.
    reduced = build_system_prompt(_CORE_TOOL_NAMES)
    assert len(reduced) > 8000
