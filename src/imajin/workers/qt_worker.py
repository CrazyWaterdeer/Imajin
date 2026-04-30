from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CancellationToken:
    _flag: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self._flag.set()

    def is_cancelled(self) -> bool:
        return self._flag.is_set()

    def raise_if_cancelled(self) -> None:
        if self._flag.is_set():
            raise CancelledError("Tool execution cancelled by user.")

    def reset(self) -> None:
        self._flag.clear()


class CancelledError(RuntimeError):
    pass


def run_in_worker(
    func: Callable[..., Any],
    *args: Any,
    progress_cb: Callable[[float], None] | None = None,
    **kwargs: Any,
):
    from napari.qt import thread_worker

    @thread_worker
    def _runner():
        return func(*args, **kwargs)

    worker = _runner()
    if progress_cb is not None:
        worker.yielded.connect(progress_cb)
    return worker
