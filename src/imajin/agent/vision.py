from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np


def encode_layer_thumbnail(
    layer: Any, max_size: int = 256, projection: str = "active_slice"
) -> str | None:
    if layer is None:
        return None
    data = layer.data
    try:
        arr = np.asarray(data.compute() if hasattr(data, "compute") else data)
    except Exception:
        return None

    if arr.ndim == 3:
        if projection == "mip":
            arr = arr.max(axis=0)
        else:
            arr = arr[arr.shape[0] // 2]
    elif arr.ndim != 2:
        return None

    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    arr = (arr * 255).astype(np.uint8)

    h, w = arr.shape
    scale = max_size / max(h, w)
    if scale < 1.0:
        from PIL import Image

        img = Image.fromarray(arr)
        img = img.resize((int(w * scale), int(h * scale)))
    else:
        from PIL import Image

        img = Image.fromarray(arr)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def thumbnail_block_anthropic(b64: str) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": b64},
    }
