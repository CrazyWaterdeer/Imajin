from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


def _swc_coordinates(coords: np.ndarray) -> np.ndarray:
    if coords.shape[1] == 2:
        y = coords[:, 0]
        x = coords[:, 1]
        z = np.zeros(len(coords), dtype=float)
        return np.column_stack([x, y, z])
    z = coords[:, 0]
    y = coords[:, 1]
    x = coords[:, 2]
    return np.column_stack([x, y, z])


def _write_swc(entry: Any, path: Path) -> None:
    graph = entry.skel.graph.tocsr()
    n = graph.shape[0]
    coords = np.asarray(entry.skel.coordinates, dtype=float) * np.asarray(entry.record.spacing)
    swc_coords = _swc_coordinates(coords)

    degrees = np.asarray(entry.skel.degrees)
    endpoints = np.where(degrees == 1)[0]
    if entry.record.soma is not None:
        soma = np.asarray(entry.record.soma, dtype=float)
        root = int(np.argmin(np.linalg.norm(coords - soma, axis=1)))
    elif len(endpoints):
        root = int(endpoints[0])
    else:
        root = 0

    parent = np.full(n, -2, dtype=int)
    parent[root] = -1
    queue: deque[int] = deque([root])
    while queue:
        node = queue.popleft()
        start, end = graph.indptr[node], graph.indptr[node + 1]
        for nb in graph.indices[start:end]:
            nb = int(nb)
            if parent[nb] != -2:
                continue
            parent[nb] = node
            queue.append(nb)
    disconnected = np.where(parent == -2)[0]
    for node in disconnected:
        parent[node] = -1

    lines = [
        "# imajin SWC export",
        "# type 3 is used for all process nodes; radius is a placeholder.",
        "# If no soma was annotated, the root is the first endpoint or node.",
    ]
    for node_id in range(n):
        x, y, z = swc_coords[node_id]
        lines.append(
            f"{node_id + 1} 3 {x:.6f} {y:.6f} {z:.6f} 0.5 "
            f"{-1 if parent[node_id] < 0 else parent[node_id] + 1}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
