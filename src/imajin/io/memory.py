from __future__ import annotations

import os
from typing import Any

import numpy as np

_GIB = 1024**3
_MIN_MEMORY_HEADROOM = 512 * 1024**2
_MAX_MEMORY_HEADROOM = 2 * _GIB


def available_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int):
            return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        pass
    return None


def array_nbytes(shape: tuple[int, ...], dtype: Any) -> int:
    return int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize)


def memory_headroom(estimated_nbytes: int) -> int:
    return max(
        _MIN_MEMORY_HEADROOM,
        min(_MAX_MEMORY_HEADROOM, estimated_nbytes // 2),
    )


def should_load_into_memory(
    estimated_nbytes: int, available_bytes: int | None = None
) -> bool:
    if available_bytes is None:
        available_bytes = available_memory_bytes()
    if available_bytes is None:
        # Prefer the fast path when memory cannot be probed; MemoryError callers
        # can still fall back to a disk-backed path.
        return True
    return estimated_nbytes + memory_headroom(estimated_nbytes) <= available_bytes
