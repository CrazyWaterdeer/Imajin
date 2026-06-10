"""Worker-thread checkpoint that blocks on an interactive ROI review.

Phase 3 of the SNR/ROI initiative. The batch workflow runs on a worker
thread; the napari review dock lives on the Qt main thread. This module
is the bridge:

* :func:`request_review_and_wait` is called from the worker. It asks the
  main thread to open/raise the review dock for the requested
  ``(image_layer, labels_layer)`` and then blocks on a
  :class:`threading.Event` until the user commits, skips, or the
  optional timeout elapses.
* :func:`notify_review_committed` / :func:`notify_review_skipped` are
  called from the main thread (typically wired to ``ReviewDock`` Qt
  signals). They wake the worker.

The module has no Qt or napari imports at module load time so it can be
exercised in headless tests by stubbing the dispatcher (the
``open_dock`` callable defaults to a real GUI helper but can be
overridden).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


_LOCK = threading.Lock()
_EVENT = threading.Event()
_RESULT: dict[str, Any] = {}
_ACTIVE = False


class ReviewAlreadyActiveError(RuntimeError):
    """Raised when a second review is requested while one is already in flight."""


def _default_open_dock(image_layer: str, labels_layer: str) -> Any:
    from imajin.session import get_viewer
    from imajin.ui.main import _show_review_panel

    viewer = get_viewer()
    if viewer is None:
        return None
    dock_widget = _show_review_panel(viewer)
    if dock_widget is None:
        return None
    dock_widget.request_layers(image_layer, labels_layer)
    return dock_widget


def request_review_and_wait(
    image_layer: str,
    labels_layer: str,
    *,
    timeout: float | None = None,
    open_dock: Callable[[str, str], Any] | None = None,
    dispatcher: Callable[[Callable[..., Any], Any], Any] | None = None,
) -> dict[str, Any]:
    """Open the review dock for ``(image_layer, labels_layer)`` and wait.

    Returns a dict like ``{"action": "commit", ...}`` /
    ``{"action": "skip", ...}`` / ``{"action": "timeout"}``. The worker
    thread is blocked until one of these notifications fires.

    ``open_dock`` defaults to :func:`_default_open_dock` (real GUI). Pass
    a stub in tests. ``dispatcher`` lets you marshal the ``open_dock``
    call onto a specific thread; when omitted, :func:`call_on_main` from
    :mod:`imajin.agent.qt_dispatch` is used so the dock is created on the
    Qt main thread.
    """
    global _ACTIVE

    open_dock_fn = open_dock or _default_open_dock

    with _LOCK:
        if _ACTIVE:
            raise ReviewAlreadyActiveError(
                "another interactive review is already pending"
            )
        _ACTIVE = True
        _EVENT.clear()
        _RESULT.clear()
    try:
        if dispatcher is not None:
            dispatcher(open_dock_fn, image_layer, labels_layer)
        else:
            from imajin.agent.qt_dispatch import call_on_main

            call_on_main(open_dock_fn, image_layer, labels_layer)
        if not _EVENT.wait(timeout=timeout):
            return {"action": "timeout"}
        return dict(_RESULT)
    finally:
        with _LOCK:
            _ACTIVE = False
            _EVENT.clear()
            _RESULT.clear()


def notify_review_committed(**info: Any) -> None:
    """Signal commit from the main thread; releases the worker."""
    _RESULT.clear()
    _RESULT.update({"action": "commit", **info})
    _EVENT.set()


def notify_review_skipped(**info: Any) -> None:
    """Signal skip from the main thread; releases the worker."""
    _RESULT.clear()
    _RESULT.update({"action": "skip", **info})
    _EVENT.set()


def is_review_active() -> bool:
    with _LOCK:
        return _ACTIVE
