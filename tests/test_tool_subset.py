"""Tests for the 20-tool core set advertised to local (Ollama) models.

CORE_TOOL_NAMES is a hand-picked subset of imajin.tools' live registry (see
tool_subset.py's module docstring for why: with the full 107-tool registry
advertised, qwen3.5:4b never terminated a multi-step task). The registry checks
here are the guard: they fail loudly the moment someone renames or removes one
of these 20 tools, rather than silently shrinking the local-model core set.
"""
from __future__ import annotations

from imajin.agent import local_models as lm
from imajin.agent.tool_subset import CORE_TOOL_NAMES, core_tools, missing_core_names

# -- CORE_TOOL_NAMES itself ---------------------------------------------------


def test_core_tool_names_all_exist_in_the_live_registry() -> None:
    from imajin.tools import tools_for_anthropic

    registered = {t["name"] for t in tools_for_anthropic()}
    missing = [name for name in CORE_TOOL_NAMES if name not in registered]
    assert missing == [], (
        f"{missing} not found in the live tool registry -- a core tool was "
        "renamed or removed; update CORE_TOOL_NAMES in tool_subset.py"
    )


def test_core_tool_names_has_twenty_unique_entries() -> None:
    # The whole point is a SMALL core; a silent duplicate would make it 19.
    assert len(CORE_TOOL_NAMES) == 20
    assert len(set(CORE_TOOL_NAMES)) == 20


# -- core_tools ----------------------------------------------------------------


def test_core_tools_preserves_core_tool_names_order_not_input_order() -> None:
    # Deliberately reversed input order -- core_tools must still come out in
    # CORE_TOOL_NAMES' pipeline order, not whatever order the registry (or a
    # provider's tools_for_anthropic) happens to iterate in.
    reversed_specs = [{"name": name} for name in reversed(CORE_TOOL_NAMES)]
    result = core_tools(reversed_specs)
    assert [t["name"] for t in result] == list(CORE_TOOL_NAMES)


def test_core_tools_skips_absent_names_without_raising() -> None:
    # A renamed tool must not crash the chat dock (see tool_subset.py) -- it
    # should just be missing from the filtered list; missing_core_names is
    # what surfaces that loudly elsewhere.
    specs = [{"name": name} for name in CORE_TOOL_NAMES if name != "cellpose_sam"]
    result = core_tools(specs)
    assert "cellpose_sam" not in [t["name"] for t in result]
    assert len(result) == len(CORE_TOOL_NAMES) - 1


def test_core_tools_returns_empty_list_for_empty_input() -> None:
    assert core_tools([]) == []


def test_core_tools_ignores_tools_outside_the_core_set() -> None:
    specs = [{"name": name} for name in CORE_TOOL_NAMES] + [{"name": "cellpose_gui"}]
    result = core_tools(specs)
    assert "cellpose_gui" not in [t["name"] for t in result]
    assert len(result) == len(CORE_TOOL_NAMES)


def test_core_tools_preserves_the_full_spec_dict_not_just_the_name() -> None:
    # Downstream code (OllamaProvider.stream) reads description/input_schema
    # off each entry -- core_tools must hand back the original dict, not a
    # name-only stub.
    specs = [{"name": "list_layers", "description": "d", "input_schema": {"type": "object"}}]
    [result] = core_tools(specs)
    assert result == specs[0]


# -- missing_core_names ---------------------------------------------------------


def test_missing_core_names_reports_absent_entries_in_core_order() -> None:
    specs = [
        {"name": name} for name in CORE_TOOL_NAMES if name not in {"get_help", "load_file"}
    ]
    missing = missing_core_names(specs)
    # CORE_TOOL_NAMES order: load_file is index 1, get_help is index 11.
    assert missing == ["load_file", "get_help"]


def test_missing_core_names_empty_when_everything_present() -> None:
    specs = [{"name": name} for name in CORE_TOOL_NAMES]
    assert missing_core_names(specs) == []


def test_missing_core_names_against_the_live_registry_is_empty() -> None:
    # Same guard as test_core_tool_names_all_exist_in_the_live_registry, phrased
    # through the function this module actually exposes for it -- this is the
    # "test that fails loudly when someone renames a core tool" from
    # missing_core_names' own docstring.
    from imajin.tools import tools_for_anthropic

    assert missing_core_names(tools_for_anthropic()) == []


# -- integration: the real, measured token/num_ctx effect ----------------------


def test_subset_tools_reduce_the_measured_prompt_token_estimate() -> None:
    """The primary, already-realized win: advertising 20 tools instead of 107
    cuts the tool-schema JSON that rides on every local-model turn, which is
    what estimate_prompt_tokens counts. Measured against the live registry and
    the current build_system_prompt() output, not a hand-picked number.
    """
    from imajin.agent.local_models import estimate_prompt_tokens
    from imajin.agent.prompts import build_system_prompt
    from imajin.tools import tools_for_anthropic

    full = tools_for_anthropic()
    subset = core_tools(full)
    prompt = build_system_prompt()

    est_full = estimate_prompt_tokens(prompt, full, [])
    est_subset = estimate_prompt_tokens(prompt, subset, [])

    assert len(subset) == 20
    assert est_subset < est_full
    # Loose bound (measured on this registry: ~34.7K -> ~16.4K, roughly
    # half) -- tight enough to catch core_tools silently no-op'ing (returning
    # `full`) or over-filtering to near-nothing, loose enough not to flake as
    # individual tool schemas drift.
    assert est_full * 0.3 < est_subset < est_full * 0.7


def _unbounded_model() -> lm.LocalModel:
    return lm.LocalModel(
        name="unbounded",
        context_length=None,
        capabilities=frozenset(),
        parameter_size=None,
        size_bytes=None,
    )


def test_choose_num_ctx_for_the_tool_count_reduction_alone() -> None:
    """Isolates slice 1's own contribution: same (full-length,
    available_tools=None) system prompt, fewer tool schemas.

    choose_num_ctx's ceil-to-8192 rounding -- not its floor -- is what lands
    both estimates inside their own multiple-of-8192 buckets here: full ->
    57344, subset -> 32768. floor=16384 vs. the old 32768 default makes NO
    difference to this specific number (24,6xx already rounds up past the
    3*8192=24576 boundary to 32768 either way); see
    test_choose_num_ctx_end_to_end_with_the_reduced_prompt below for where the
    floor change actually bites now that prompts.py's available_tools trim
    has *also* landed.
    """
    from imajin.agent.local_models import choose_num_ctx, estimate_prompt_tokens
    from imajin.agent.prompts import build_system_prompt
    from imajin.tools import tools_for_anthropic

    full = tools_for_anthropic()
    subset = core_tools(full)
    prompt = build_system_prompt()  # available_tools=None -> unreduced, full length

    est_full = estimate_prompt_tokens(prompt, full, [])
    est_subset = estimate_prompt_tokens(prompt, subset, [])

    unbounded = _unbounded_model()
    assert choose_num_ctx(unbounded, est_full) == 57344
    assert choose_num_ctx(unbounded, est_subset, floor=16384) == 32768
    assert choose_num_ctx(unbounded, est_subset, floor=16384) < choose_num_ctx(
        unbounded, est_full
    )


def test_choose_num_ctx_end_to_end_with_the_reduced_prompt() -> None:
    """The full, real, TODAY number: 20 tools AND prompts.py's
    available_tools trim (a separate, parallel slice against the same pinned
    contract) both in effect -- this is exactly what
    chat_dock._make_provider computes for a local-model turn.

    Here floor=16384 is load-bearing: the OLD default (32768) would force the
    estimate's natural sub-32768 rounding right back up to twice what's
    needed -- this is a live demonstration of the "floor swallows the saving"
    failure the call-site change fixes, not a synthetic one (see
    test_local_models.py for that in isolation, with a hand-picked estimate,
    independent of prompts.py).
    """
    from imajin.agent.local_models import choose_num_ctx, estimate_prompt_tokens
    from imajin.agent.prompts import build_system_prompt
    from imajin.tools import tools_for_anthropic

    full = tools_for_anthropic()
    subset = core_tools(full)
    reduced_prompt = build_system_prompt(available_tools=set(CORE_TOOL_NAMES))

    est = estimate_prompt_tokens(reduced_prompt, subset, [])

    unbounded = _unbounded_model()
    old_default_floor_result = choose_num_ctx(unbounded, est)
    new_floor_result = choose_num_ctx(unbounded, est, floor=16384)

    assert new_floor_result % 8192 == 0
    assert new_floor_result < 32768
    assert new_floor_result < old_default_floor_result
