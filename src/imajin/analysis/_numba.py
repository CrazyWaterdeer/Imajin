"""Lazy, cached compilation of numba kernels.

Analysis modules keep a fast import by compiling their numba kernels only on first
use. This centralises that boilerplate: ``lazy_kernel(build)`` returns a getter
that compiles + caches the kernel on first call (``build(njit)`` must return the
compiled function) and returns ``None`` when numba is unavailable, so callers fall
back to their numpy path.
"""

from __future__ import annotations

from typing import Any, Callable


def lazy_kernel(build: Callable[[Any], Any]) -> Callable[[], Any]:
    """Return a getter that lazily compiles + caches a numba kernel.

    ``build`` receives the ``numba.njit`` decorator and returns the compiled
    kernel. The getter returns that kernel, or ``None`` if numba is unavailable
    (the caller then uses its numpy fallback).
    """
    cache: dict[str, Any] = {}

    def get() -> Any:
        if "fn" not in cache:
            try:
                from numba import njit

                cache["fn"] = build(njit)
            except Exception:  # pragma: no cover - numba is a declared dependency
                cache["fn"] = None
        return cache["fn"]

    return get
