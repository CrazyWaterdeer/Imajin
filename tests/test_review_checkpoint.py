"""Headless tests for the review checkpoint plumbing.

The Qt dock is stubbed out — we only exercise the worker-thread
``request_review_and_wait`` ↔ main-thread ``notify_*`` handshake.
"""

from __future__ import annotations

import threading
import time

import pytest

from imajin.agent import review_checkpoint as rc


def _reset_state() -> None:
    # Forcefully reset the module-level state between tests.
    rc._EVENT.clear()
    rc._RESULT.clear()
    with rc._LOCK:
        rc._ACTIVE = False


def test_commit_releases_waiting_worker():
    _reset_state()
    captured: dict[str, object] = {}

    def fake_open(image_layer: str, labels_layer: str) -> None:
        captured["image_layer"] = image_layer
        captured["labels_layer"] = labels_layer

    out: dict[str, object] = {}

    def worker():
        out["result"] = rc.request_review_and_wait(
            "img",
            "labels",
            open_dock=fake_open,
            dispatcher=lambda fn, *a: fn(*a),
        )

    t = threading.Thread(target=worker)
    t.start()
    # Give the worker a moment to enter the wait.
    for _ in range(50):
        if rc.is_review_active():
            break
        time.sleep(0.01)
    assert rc.is_review_active()
    rc.notify_review_committed(labels_layer="labels", final_objects=3)
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert out["result"]["action"] == "commit"
    assert out["result"]["final_objects"] == 3
    assert captured == {"image_layer": "img", "labels_layer": "labels"}


def test_skip_returns_skip_action():
    _reset_state()

    def fake_open(_i: str, _l: str) -> None:
        return None

    out: dict[str, object] = {}

    def worker():
        out["result"] = rc.request_review_and_wait(
            "img",
            "labels",
            open_dock=fake_open,
            dispatcher=lambda fn, *a: fn(*a),
        )

    t = threading.Thread(target=worker)
    t.start()
    for _ in range(50):
        if rc.is_review_active():
            break
        time.sleep(0.01)
    rc.notify_review_skipped(reason="user_skipped")
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert out["result"]["action"] == "skip"
    assert out["result"]["reason"] == "user_skipped"


def test_timeout_returns_timeout_action():
    _reset_state()
    result = rc.request_review_and_wait(
        "img",
        "labels",
        open_dock=lambda _i, _l: None,
        dispatcher=lambda fn, *a: fn(*a),
        timeout=0.1,
    )
    assert result == {"action": "timeout"}
    assert not rc.is_review_active()


def test_concurrent_review_raises():
    _reset_state()
    # Hold a review open on a worker thread.
    started = threading.Event()

    def hold_review():
        rc.request_review_and_wait(
            "img",
            "labels",
            open_dock=lambda _i, _l: started.set(),
            dispatcher=lambda fn, *a: fn(*a),
            timeout=2.0,
        )

    t = threading.Thread(target=hold_review)
    t.start()
    started.wait(timeout=1.0)
    assert rc.is_review_active()
    with pytest.raises(rc.ReviewAlreadyActiveError):
        rc.request_review_and_wait(
            "other_img",
            "other_labels",
            open_dock=lambda _i, _l: None,
            dispatcher=lambda fn, *a: fn(*a),
            timeout=0.05,
        )
    rc.notify_review_committed()
    t.join(timeout=2.0)
    assert not rc.is_review_active()


def test_active_resets_on_timeout():
    _reset_state()
    rc.request_review_and_wait(
        "img",
        "labels",
        open_dock=lambda _i, _l: None,
        dispatcher=lambda fn, *a: fn(*a),
        timeout=0.05,
    )
    assert not rc.is_review_active()
    # Should be able to start a new review after the timeout cleared state.
    out: dict[str, object] = {}

    def worker():
        out["result"] = rc.request_review_and_wait(
            "img2",
            "labels2",
            open_dock=lambda _i, _l: None,
            dispatcher=lambda fn, *a: fn(*a),
        )

    t = threading.Thread(target=worker)
    t.start()
    for _ in range(50):
        if rc.is_review_active():
            break
        time.sleep(0.01)
    rc.notify_review_committed()
    t.join(timeout=2.0)
    assert out["result"]["action"] == "commit"
