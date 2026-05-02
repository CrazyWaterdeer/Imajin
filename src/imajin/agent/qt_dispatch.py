from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qtpy.QtCore import QThread

_DISPATCHER: Any | None = None


def set_dispatcher(dispatcher: Any | None) -> None:
    global _DISPATCHER
    _DISPATCHER = dispatcher


def call_on_main(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a small UI-bound callable on the Qt main thread.

    When no dispatcher is registered (tests, scripts, manual direct calls), this
    simply calls the function inline.
    """

    if _DISPATCHER is None:
        return func(*args, **kwargs)
    try:
        if QThread.currentThread() == _DISPATCHER.thread():
            return func(*args, **kwargs)
    except RuntimeError:
        return func(*args, **kwargs)
    return _DISPATCHER.invoke(func, *args, **kwargs)
