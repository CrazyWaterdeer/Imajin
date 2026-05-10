from __future__ import annotations

import contextlib
import contextvars
import shutil
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from imajin.analysis.arrays import materialize_array
from imajin.agent.qt_dispatch import call_on_main
from imajin.agent.state import get_table
from imajin.paths import normalize_user_path
from imajin.results import read_bundle_metadata, write_bundle_metadata


_active_bundle: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "imajin_active_bundle", default=None
)
_active_sample_slug: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "imajin_active_sample_slug", default=None
)


def current_bundle() -> Path | None:
    return _active_bundle.get()


def current_sample_slug() -> str | None:
    return _active_sample_slug.get()


@contextlib.contextmanager
def with_active_bundle(path: Path | str) -> Iterator[Path]:
    p = Path(path)
    token = _active_bundle.set(p)
    try:
        yield p
    finally:
        _active_bundle.reset(token)


@contextlib.contextmanager
def with_active_sample_slug(slug: str | None) -> Iterator[str | None]:
    token = _active_sample_slug.set(slug)
    try:
        yield slug
    finally:
        _active_sample_slug.reset(token)


def materialize_result_array(arr: Any) -> np.ndarray:
    return materialize_array(arr)


def label_output_dtype(data: np.ndarray) -> np.dtype:
    max_label = int(np.nanmax(data)) if data.size else 0
    if max_label <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def write_label_layer(bundle: Path, tier: str, sample_slug: str, layer_name: str) -> str:
    """Snapshot a label layer and write it to bundle/labels/<tier>/<slug>.tif."""
    import tifffile
    from imajin.tools.napari_ops import snapshot_layer

    layer = call_on_main(snapshot_layer, layer_name)
    data = materialize_result_array(layer.data)
    labels = data.astype(label_output_dtype(data), copy=False)
    rel = Path("labels") / tier / f"{sample_slug}.tif"
    out = bundle / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError(
            f"{rel} already exists in bundle {bundle.name}; "
            "sample_slug collision suspected"
        )
    tifffile.imwrite(out, labels)
    return rel.as_posix()


def copy_qc_png(bundle: Path, qc_png: str, sample_slug: str) -> str | None:
    src = normalize_user_path(qc_png).resolve()
    if not src.exists():
        return None
    rel = Path("qc") / f"{sample_slug}.png"
    dst = bundle / rel
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return rel.as_posix()


def populate_sample_outputs(
    bundle: Path,
    sample_slug: str,
    *,
    labels_cells: str | None = None,
    labels_domain: str | None = None,
    qc_png: str | None = None,
) -> dict[str, str | None]:
    out: dict[str, str | None] = {
        "labels_cells": None,
        "labels_domain": None,
        "qc_png": None,
    }
    if labels_cells:
        out["labels_cells"] = write_label_layer(
            bundle, "cells", sample_slug, labels_cells
        )
    if labels_domain:
        out["labels_domain"] = write_label_layer(
            bundle, "domain", sample_slug, labels_domain
        )
    if qc_png:
        out["qc_png"] = copy_qc_png(bundle, qc_png, sample_slug)
    return out


def write_combined_csv(bundle: Path, table_names: list[str]) -> Path:
    import pandas as pd

    frames: list[pd.DataFrame] = []
    for name in table_names:
        try:
            frame = get_table(name)
        except KeyError:
            continue
        if frame is None or frame.empty:
            continue
        frames.append(frame)
    out = bundle / "tables" / "combined.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame()
    combined.to_csv(out, index=False)
    return out


def _environment_from_flat_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in ("python_version", "imajin_version", "deps", "git_commit")
        if metadata.get(key) is not None
    }


def _normalize_bundle_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata.get("schema_version") == 2:
        return {
            "schema_version": 2,
            "recipe_params": dict(metadata.get("recipe_params") or {}),
            "run_context": dict(metadata.get("run_context") or {}),
            "environment": dict(metadata.get("environment") or {}),
        }

    run_context_keys = {
        "kind",
        "tier",
        "name",
        "status",
        "created_at",
        "samples",
        "n_samples",
        "n_complete",
        "n_failed",
        "tables",
    }
    run_context = {
        key: metadata[key]
        for key in run_context_keys
        if key in metadata
    }
    return {
        "schema_version": 2,
        "recipe_params": dict(metadata.get("recipe") or metadata.get("recipe_params") or {}),
        "run_context": run_context,
        "environment": _environment_from_flat_metadata(metadata),
    }


def read_bundle_metadata_normalized(bundle: Path | str) -> dict[str, Any]:
    """Read bundle metadata as the schema-v2 logical shape.

    Older flat metadata files are mapped into recipe_params/run_context/environment
    so reuse tools and downstream stats do not need version-specific branches.
    """
    return _normalize_bundle_metadata(read_bundle_metadata(bundle))


def finalize_bundle_metadata(
    bundle: Path,
    *,
    samples: list[dict[str, Any]],
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    seed = read_bundle_metadata(bundle)
    normalized = _normalize_bundle_metadata(seed)
    extra = dict(extra or {})
    run_context_extras = dict(extra.pop("run_context_extras", {}) or {})

    recipe_params = (
        extra.pop("recipe_params", None)
        or normalized.get("recipe_params")
        or {}
    )
    environment = {
        **dict(normalized.get("environment") or {}),
        **dict(extra.pop("environment", {}) or {}),
    }
    samples_list = list(samples)
    run_context = {
        **dict(normalized.get("run_context") or {}),
        "status": status,
        "samples": samples_list,
        "n_samples": len(samples_list),
        "n_complete": sum(1 for s in samples_list if s.get("status") == "complete"),
        "n_failed": sum(1 for s in samples_list if s.get("status") == "failed"),
        "tables": {"combined": "tables/combined.csv"},
        **run_context_extras,
        **extra,
    }
    write_bundle_metadata(
        bundle,
        {
            "schema_version": 2,
            "recipe_params": dict(recipe_params),
            "run_context": run_context,
            "environment": environment,
        },
    )
