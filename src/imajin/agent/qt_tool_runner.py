"""Run tool calls on the main Qt thread, even when invoked from a worker.

The agent runner streams LLM responses inside a `napari.qt.thread_worker`
(non-main thread). Tools that touch the napari viewer (e.g. `viewer.add_labels`
in `cellpose_sam`) create Qt objects, which must happen on the main thread or
Qt prints `QObject::setParent: Cannot set parent, new parent is in a different
thread` and the UI eventually deadlocks.

This module provides a `MainThreadToolRunner(QObject)` that, when parented to a
widget on the main thread, marshals tool calls back to the main thread via a
`BlockingQueuedConnection`. The worker blocks until the tool returns (or
raises), preserving the synchronous semantics the runner expects.
"""
from __future__ import annotations

from typing import Any

from qtpy.QtCore import QObject, Qt, QThread, Signal, Slot


class MainThreadToolRunner(QObject):
    """Bridges tool dispatch from worker thread → main thread.

    Construct on the main thread (typically `parent=chat_dock`). Workers call
    `.call(name, **kwargs)`; the call is forwarded to `_handle` via a
    `BlockingQueuedConnection`, so the emitter blocks until the slot returns.
    """

    _request = Signal(object)  # payload dict, mutated in-place by the slot

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._request.connect(self._handle, Qt.ConnectionType.BlockingQueuedConnection)
        from imajin.agent.qt_dispatch import set_dispatcher

        set_dispatcher(self)

    @Slot(object)
    def _handle(self, payload: dict[str, Any]) -> None:
        from imajin.tools import call_tool

        try:
            if "func" in payload:
                payload["result"] = payload["func"](*payload["args"], **payload["kwargs"])
            else:
                payload["result"] = call_tool(payload["name"], **payload["kwargs"])
        except Exception as e:
            payload["error"] = e

    def invoke(self, func, *args: Any, **kwargs: Any) -> Any:
        if QThread.currentThread() == self.thread():
            return func(*args, **kwargs)

        payload: dict[str, Any] = {
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "result": None,
            "error": None,
        }
        self._request.emit(payload)
        if payload["error"] is not None:
            raise payload["error"]
        return payload["result"]

    def call(self, name: str, **kwargs: Any) -> Any:
        from imajin.tools import call_tool
        from imajin.tools.registry import get_tool

        # Same thread? Just call directly — no marshalling needed.
        if QThread.currentThread() == self.thread():
            return call_tool(name, **kwargs)

        try:
            entry = get_tool(name)
        except KeyError:
            entry = None
        if entry is not None and entry.worker:
            return call_tool(name, **kwargs)

        payload: dict[str, Any] = {
            "name": name,
            "kwargs": kwargs,
            "result": None,
            "error": None,
        }
        self._request.emit(payload)  # blocks until _handle returns
        if payload["error"] is not None:
            raise payload["error"]
        return payload["result"]
