from __future__ import annotations

import json
from typing import Any

import numpy as np


def _sample_array(data: Any, max_points: int = 65_536) -> tuple[np.ndarray, bool]:
    shape = tuple(int(s) for s in getattr(data, "shape", ()))
    if not shape:
        arr = np.asarray(data.compute() if hasattr(data, "compute") else data)
        return arr, False

    total = int(np.prod(shape, dtype=np.int64))
    if total <= max_points:
        arr = data.compute() if hasattr(data, "compute") else data
        return np.asarray(arr), False

    per_axis = max(1, int(round(max_points ** (1 / max(1, len(shape))))))
    slices = tuple(slice(None, None, max(1, int(np.ceil(s / per_axis)))) for s in shape)
    sample = data[slices]
    sample = sample.compute() if hasattr(sample, "compute") else sample
    return np.asarray(sample), True


def _layer_summary(layer: Any) -> dict[str, Any]:
    data = layer.data
    shape = tuple(int(s) for s in getattr(data, "shape", ()))
    dtype = str(getattr(data, "dtype", "?"))

    md_raw = getattr(layer, "metadata", None)
    md = dict(md_raw) if isinstance(md_raw, dict) else {}

    scale_raw = getattr(layer, "scale", None)
    try:
        scale = tuple(float(s) for s in scale_raw) if scale_raw is not None else ()
    except TypeError:
        scale = ()

    info: dict[str, Any] = {
        "name": layer.name,
        "kind": getattr(layer, "kind", type(layer).__name__.lower()),
        "shape": shape,
        "dtype": dtype,
        "scale": scale,
    }
    if "axes" in md:
        info["axes"] = md["axes"]
    if "voxel_size_um" in md:
        info["voxel_size_um"] = md["voxel_size_um"]

    if info["kind"] == "image" and shape:
        try:
            sample, sampled = _sample_array(data)
            if sample.size > 0:
                info["intensity"] = {
                    "min": float(sample.min()),
                    "max": float(sample.max()),
                    "mean": float(sample.mean()),
                    "p1": float(np.percentile(sample, 1)),
                    "p99": float(np.percentile(sample, 99)),
                    "sampled": sampled,
                }
        except Exception:
            pass
    elif info["kind"] == "labels" and shape:
        try:
            sample, sampled = _sample_array(data)
            info["n_labels_sample"] = int(sample.max())
            info["sampled"] = sampled
        except Exception:
            pass

    return info


def summarize_viewer_state(
    *,
    max_layers: int = 12,
    max_tables: int = 12,
    max_samples: int = 20,
    max_channels: int = 20,
) -> str:
    from imajin.session import (
        list_channel_annotations,
        list_samples,
        list_tables,
        viewer_or_none,
    )

    viewer = viewer_or_none()
    if viewer is None:
        samples = list_samples()
        channels = list_channel_annotations()
        return json.dumps(
            {
                "layers": [],
                "tables": [],
                "samples": samples[:max_samples],
                "channels": channels[:max_channels],
                "omitted": {
                    "samples": max(0, len(samples) - max_samples),
                    "channels": max(0, len(channels) - max_channels),
                },
                "note": "viewer not initialized",
            }
        )

    layer_list = list(viewer.layers)
    layers = [_layer_summary(L) for L in layer_list[:max_layers]]
    tables = list_tables()
    samples = list_samples()
    channels = list_channel_annotations()
    return json.dumps(
        {
            "layers": layers,
            "tables": tables[:max_tables],
            "samples": samples[:max_samples],
            "channels": channels[:max_channels],
            "omitted": {
                "layers": max(0, len(layer_list) - max_layers),
                "tables": max(0, len(tables) - max_tables),
                "samples": max(0, len(samples) - max_samples),
                "channels": max(0, len(channels) - max_channels),
            },
        },
        default=str,
    )


def _canon_key(key: Any) -> str:
    s = str(key)
    if "/" in s or "\\" in s:
        try:
            from imajin.tools.files import _canonical_path_text

            return _canonical_path_text(s)
        except Exception:
            return s
    return s


def _key_label(key: str, fallback: str) -> str:
    if "/" in key or "\\" in key:
        import os.path

        return os.path.splitext(os.path.basename(key))[0]
    return fallback


def batch_progress_data() -> dict[str, Any]:
    """Structured analysed/pending/failed progress, normalised to canonical keys so
    interactive runs (keyed by source path) and batch runs (keyed by registered
    file_id) resolve to one universe. Source of truth: the session `AnalysisRun`s +
    the file registry."""
    from imajin.session import iter_file_records, list_runs

    runs = list_runs()
    records = list(iter_file_records())

    universe: dict[str, str] = {}          # canon_key -> registered file_id (label)
    fid_to_canon: dict[str, str] = {}
    for rec in records:
        path = getattr(rec, "path", None)
        ck = _canon_key(path) if path else _canon_key(getattr(rec, "file_id", ""))
        universe[ck] = getattr(rec, "file_id", ck)
        fid_to_canon[getattr(rec, "file_id", ck)] = ck

    complete: dict[str, str | None] = {}   # canon_key -> result table
    failed: set[str] = set()
    for r in runs:
        ck = fid_to_canon.get(r.get("file_id"), _canon_key(r.get("file_id")))
        table = (r.get("table_names") or [None])[0]
        if r.get("status") == "complete":
            complete[ck] = table
        elif r.get("status") == "failed":
            failed.add(ck)
    failed -= set(complete)

    universe_known = bool(universe)
    pending = [k for k in universe if k not in complete] if universe_known else []

    def _entry(keys: list[str]) -> list[dict[str, Any]]:
        return [
            {"label": _key_label(k, universe.get(k, k)), "table": complete.get(k), "key": k}
            for k in keys
        ]

    # The "current" file(s): loaded image files (by source_path) not yet analysed.
    # Read inline + headless-safe so this never requires a viewer and never imports
    # imajin.tools (which would register all tools during agent-context import).
    current: list[dict[str, Any]] = []
    try:
        from imajin.session import viewer_or_none

        viewer = viewer_or_none()
        if viewer is not None:
            seen: set[str] = set()
            for layer in list(viewer.layers):
                md = getattr(layer, "metadata", None)
                sp = (md.get("source_path") or md.get("path")) if isinstance(md, dict) else None
                if not sp:
                    continue
                ck = _canon_key(sp)
                if ck in complete or ck in seen:
                    continue
                seen.add(ck)
                current.append({"label": _key_label(ck, universe.get(ck, ck)), "key": ck})
    except Exception:
        current = []

    return {
        "analysed": _entry(list(complete)),
        "pending": _entry(pending),
        "failed": _entry(list(failed)),
        "current": current,
        "universe_known": universe_known,
        "n_universe": len(universe),
        "next_pending": (_key_label(pending[0], universe.get(pending[0], pending[0]))
                         if pending else None),
    }


def summarize_batch_progress(max_labels: int = 8, max_chars: int = 600) -> str | None:
    """Compact one-block ledger for the per-turn system prompt; `None` when there is
    no batch state to show (single-shot use)."""
    data = batch_progress_data()
    if (
        not data["analysed"]
        and not data["failed"]
        and not data["pending"]
        and not data.get("current")
    ):
        return None

    n_a, n_f, n_p = len(data["analysed"]), len(data["failed"]), len(data["pending"])
    if data["universe_known"]:
        head = (
            f"Batch progress: analysed {n_a}/{data['n_universe']}, "
            f"failed {n_f}, pending {n_p}."
        )
    else:
        head = (
            f"Batch progress: analysed {n_a}, failed {n_f}, "
            "pending unknown (call register_files to track the batch)."
        )
    lines = [head]

    cur = data.get("current") or []
    if cur:
        labels = ", ".join(c["label"] for c in cur[:max_labels])
        lines.append(f"  current (loaded, not yet analysed): {labels}")

    def _fmt(entries: list[dict[str, Any]]) -> str:
        shown = entries[:max_labels]
        parts = [
            f"{e['label']} [{e['table']}]" if e.get("table") else e["label"]
            for e in shown
        ]
        if len(entries) > max_labels:
            parts.append(f"(+{len(entries) - max_labels} more)")
        return ", ".join(parts)

    if data["analysed"]:
        lines.append("  analysed: " + _fmt(data["analysed"]))
    if data["pending"]:
        lines.append("  pending: " + _fmt(data["pending"]))
    if data["failed"]:
        lines.append("  failed: " + _fmt(data["failed"]))
    lines.append(
        "  Re-analyse a listed file only when the user asks to rerun or changes "
        "parameters; pick the next pending file when continuing a batch."
    )

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text
