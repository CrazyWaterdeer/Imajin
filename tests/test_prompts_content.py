"""Regression guard for the time-course pipeline guidance in the system prompt.

The reported bug (Korean, verbatim): "live imaging tiff를 넣으면 roi를 단 한 장의
사진에 대해서만 분석을 하는 바람에 위치가 흔들리면 측정이 달라지는 문제" -- an ROI
drawn/segmented on one frame got measured against every frame of the movie, so
sample drift silently changed the measurement. The tool-side fix is a shape/axis
contract in measure.py; this file pins the *prompt*-side fix, because the agent
caused the bug by treating a single-frame ROI as good enough to measure directly
(prompts.py's old "if an ROI/Labels layer already exists, invoke
measure_intensity_over_time" step). A future edit that quietly restores that
shortcut, or drops the new tool names, should fail here before it ships.
"""
from __future__ import annotations

from imajin.agent.prompts import build_system_prompt

_NEW_TOOLS = ("track_roi_over_time", "resegment_roi_over_time")


def _time_course_pipeline_block(prompt: str) -> str:
    """Slice the Pipeline "time course" block out of the prompt (up to the next
    Pipeline block), so assertions can't accidentally match unrelated sections."""
    start = prompt.index('Pipeline "time course"')
    end = prompt.index('Pipeline "representative image', start)
    return prompt[start:end]


def _time_course_intent_bullet(prompt: str) -> str:
    """Slice the "Intent -> pipeline mappings" bullet for the same trigger set
    (a second, independent place the old bug's guidance used to live)."""
    start = prompt.index('- **"intensity over time"**')
    end = prompt.index("\n\n", start)
    return prompt[start:end]


def test_new_time_course_tools_are_named_in_the_prompt() -> None:
    prompt = build_system_prompt()
    for name in _NEW_TOOLS:
        assert name in prompt, f"{name!r} is never mentioned in build_system_prompt()"


def test_time_course_pipeline_no_longer_terminates_at_extract_timepoint() -> None:
    prompt = build_system_prompt()
    block = _time_course_pipeline_block(prompt)

    # The exact old shortcut: hand back a representative frame and stop, leaving
    # measurement to reuse that single frame against the whole movie.
    assert "then continue once ROIs exist" not in block

    for name in _NEW_TOOLS:
        assert name in block, f"{name!r} missing from the time-course pipeline block"

    # Carrying the ROI through the movie must happen BEFORE measurement, not after.
    assert "measure_intensity_over_time" in block
    assert block.index("track_roi_over_time") < block.index("measure_intensity_over_time")


def test_time_course_pipeline_forbids_static_roi_against_movie() -> None:
    prompt = build_system_prompt()
    block = _time_course_pipeline_block(prompt).lower()
    assert "never measure" in block or "do not measure" in block


def test_time_course_pipeline_states_track_tool_limits() -> None:
    prompt = build_system_prompt()
    block = _time_course_pipeline_block(prompt)

    # track_roi_over_time is 2D+T only; a 4D movie needs projection or resegmentation.
    assert "2D+T" in block
    assert "resegment_roi_over_time" in block
    # Gated/undetected frames are a deliberate gap, never a fabricated measurement.
    assert "no row" in block.lower() or "background (0)" in block


def test_time_course_intent_bullet_matches_the_pipeline_fix() -> None:
    prompt = build_system_prompt()
    bullet = _time_course_intent_bullet(prompt)

    # The old bullet's terminal step -- create a reference frame and stop.
    assert "create a reference frame first" not in bullet
    assert "track_roi_over_time" in bullet


def test_time_course_pipeline_recommends_structural_channel_tracking() -> None:
    """Slice R3: track_roi_over_time gates by signal CONTRAST, not drift, so on
    a movie whose signal itself dips toward baseline the surviving trace reads
    biased high (measured: +15.2% on an isolated single-ROI case). The fix
    correct_sparse's own docstring already names ("an activity-independent
    landmark") must actually reach the agent, not stay a fact only the
    analysis-layer docstring knows."""
    prompt = build_system_prompt()
    block = _time_course_pipeline_block(prompt).lower()

    assert "structural" in block
    assert "activity-independent" in block
    # The bias must be named as a bias, not left as a silent coverage number.
    assert "bias" in block
    assert "missed a few" in block
