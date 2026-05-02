from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from imajin.agent.state import get_layer, get_viewer, put_table
from imajin.tools.registry import tool


def _materialize(arr) -> np.ndarray:
    return np.asarray(arr.compute() if hasattr(arr, "compute") else arr)


@tool(
    description="Cell tracking across a time series. Input is a Labels layer with a "
    "leading T axis (TYX for 2D+T, TZYX for 3D+T). Uses Bayesian tracking (btrack) to "
    "link instance masks across frames and produces a Tracks layer plus a per-detection "
    "table (track_id, t, z, y, x). search_radius is in pixels.",
    phase="6",
)
def track_cells(
    labels_layer: str,
    search_radius: float = 50.0,
    max_lost: int = 5,
) -> dict[str, Any]:
    L = get_layer(labels_layer)
    data = _materialize(L.data)
    if data.ndim not in (3, 4):
        raise ValueError(
            f"track_cells expects 3D (TYX) or 4D (TZYX) labels; got shape {data.shape}"
        )

    import btrack
    from btrack import datasets, utils

    objects = utils.segmentation_to_objects(data, num_workers=1)
    if not objects:
        raise ValueError("no objects found in labels layer — segmentation may be empty")

    config_file = datasets.cell_config()
    with btrack.BayesianTracker(verbose=False) as tracker:
        tracker.configure(config_file)
        tracker.max_search_radius = float(search_radius)
        if hasattr(tracker, "max_lost"):
            tracker.max_lost = int(max_lost)
        tracker.append(objects)
        if data.ndim == 4:
            tracker.volume = (
                (0, data.shape[-1]),
                (0, data.shape[-2]),
                (0, data.shape[-3]),
            )
        else:
            tracker.volume = ((0, data.shape[-1]), (0, data.shape[-2]), (-1e5, 1e5))
        tracker.track()
        tracker.optimize()
        tracks_data, properties, graph = tracker.to_napari()

    tracks_arr = np.asarray(tracks_data)
    if tracks_arr.shape[1] == 4:
        cols = ["track_id", "t", "y", "x"]
    else:
        cols = ["track_id", "t", "z", "y", "x"]
    df = pd.DataFrame(tracks_arr, columns=cols)
    df["track_id"] = df["track_id"].astype(int)
    df["t"] = df["t"].astype(int)
    if isinstance(properties, dict):
        for key, vals in properties.items():
            arr = np.asarray(vals)
            if arr.shape[0] == len(df):
                df[key] = arr

    table_name = put_table(
        f"{L.name}_tracks",
        df,
        spec={"op": "track_cells", "source_layer": L.name, "search_radius": float(search_radius)},
    )

    viewer = get_viewer()
    new_layer = viewer.add_tracks(
        tracks_arr,
        name=f"{L.name}_tracks",
        graph=graph if isinstance(graph, dict) else {},
    )

    return {
        "tracks_layer": new_layer.name,
        "tracks_table": table_name,
        "n_tracks": int(df["track_id"].nunique()),
        "n_detections": int(len(df)),
    }
