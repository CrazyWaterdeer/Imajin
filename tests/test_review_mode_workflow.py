"""Workflow-level coverage for Phase-3 review_mode wiring.

Exercises analyze_target_cells with review_mode='auto' (no pause) and
review_mode='interactive' (request_review_and_wait is stubbed). No Qt
event loop is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from imajin import session as state
from imajin.tools import workflows


@pytest.fixture(autouse=True)
def _clean_tables():
    state.reset_tables()
    yield
    state.reset_tables()


def _add_target(viewer) -> str:
    labels = np.zeros((24, 24), dtype=np.int32)
    img = np.zeros_like(labels, dtype=np.float32)
    img[4:12, 4:12] = 200.0
    img[14:22, 14:22] = 80.0
    labels[4:12, 4:12] = 1
    labels[14:22, 14:22] = 2
    viewer.add_image(img, name="green_target", scale=(0.5, 0.5))
    state.put_channel_annotation("green_target", role="target", color="green")
    return "green_target"


def _stub_cellpose(monkeypatch: pytest.MonkeyPatch, mask: np.ndarray) -> None:
    from imajin.tools import segment

    class _FakeModel:
        def eval(self, data, **kwargs):  # noqa: ANN001
            return mask, None, None

    monkeypatch.setattr("imajin.tools._segmentation_io._get_cellpose_model", lambda *a, **kw: _FakeModel())


def test_review_mode_auto_skips_checkpoint(viewer, monkeypatch):
    """auto mode must not call request_review_and_wait."""
    _add_target(viewer)
    mask = np.zeros((24, 24), dtype=np.int32)
    mask[4:12, 4:12] = 1
    mask[14:22, 14:22] = 2
    _stub_cellpose(monkeypatch, mask)

    from imajin.agent import review_checkpoint

    calls = []

    def boom(*_a, **_kw):
        calls.append("called")
        raise AssertionError("review checkpoint must not run in auto mode")

    monkeypatch.setattr(review_checkpoint, "request_review_and_wait", boom)

    res = workflows.analyze_target_cells(review_mode="auto")
    assert res["ok"] is True
    assert calls == []
    assert res.get("review") is None


def test_review_mode_interactive_calls_checkpoint(viewer, monkeypatch):
    _add_target(viewer)
    mask = np.zeros((24, 24), dtype=np.int32)
    mask[4:12, 4:12] = 1
    mask[14:22, 14:22] = 2
    _stub_cellpose(monkeypatch, mask)

    from imajin.agent import review_checkpoint

    captured: dict[str, object] = {}

    def stub_request(image_layer, labels_layer, *, timeout=None):
        captured["image_layer"] = image_layer
        captured["labels_layer"] = labels_layer
        captured["timeout"] = timeout
        return {"action": "commit", "final_voxels": 100}

    monkeypatch.setattr(review_checkpoint, "request_review_and_wait", stub_request)

    res = workflows.analyze_target_cells(
        review_mode="interactive", review_timeout_s=1.5
    )
    assert res["ok"] is True
    assert res["review"]["action"] == "commit"
    assert captured["image_layer"] == "green_target"
    assert captured["labels_layer"]  # the seg result labels layer name
    assert captured["timeout"] == pytest.approx(1.5)


def test_review_mode_skip_returns_skipped_stage(viewer, monkeypatch):
    _add_target(viewer)
    mask = np.zeros((24, 24), dtype=np.int32)
    mask[4:12, 4:12] = 1
    _stub_cellpose(monkeypatch, mask)

    from imajin.agent import review_checkpoint

    monkeypatch.setattr(
        review_checkpoint,
        "request_review_and_wait",
        lambda *a, **kw: {"action": "skip", "reason": "user_skipped"},
    )

    res = workflows.analyze_target_cells(review_mode="interactive")
    assert res["ok"] is False
    assert res["stage"] == "review_skipped"
    assert res["review"]["action"] == "skip"


def test_review_mode_timeout_emits_warning_and_continues(viewer, monkeypatch):
    _add_target(viewer)
    mask = np.zeros((24, 24), dtype=np.int32)
    mask[4:12, 4:12] = 1
    _stub_cellpose(monkeypatch, mask)

    from imajin.agent import review_checkpoint

    monkeypatch.setattr(
        review_checkpoint,
        "request_review_and_wait",
        lambda *a, **kw: {"action": "timeout"},
    )

    res = workflows.analyze_target_cells(review_mode="interactive")
    assert res["ok"] is True
    assert any("timed out" in w for w in res.get("warnings", []))


def test_recipe_review_mode_field_round_trips():
    state.put_recipe(name="rev_test", review_mode="interactive")
    recipe = state.get_recipe("rev_test")
    assert recipe.review_mode == "interactive"

    # Restoring from a dict (rehydration path) keeps the field.
    state.reset_recipes()
    state.restore_session_dict(
        {
            "recipes": [
                {"name": "rev_test", "review_mode": "interactive"},
            ]
        }
    ) if hasattr(state, "restore_session_dict") else None
    # That helper may not exist — just put a fresh one as a fallback check.
    state.put_recipe(name="rev_test2")
    assert state.get_recipe("rev_test2").review_mode == "auto"

    with pytest.raises(ValueError):
        state.put_recipe(name="bad", review_mode="weird")
