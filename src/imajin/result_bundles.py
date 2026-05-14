from __future__ import annotations

import contextlib
import contextvars
import shutil
import threading
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

_process_bundle_lock = threading.Lock()
_process_bundle: Path | None = None


def reset_process_bundle() -> None:
    """Drop the process-global ad-hoc bundle slot. Intended for tests."""
    global _process_bundle
    with _process_bundle_lock:
        _process_bundle = None


def ensure_active_bundle() -> Path:
    """Return the active bundle, creating a process-wide ad-hoc one if needed."""
    global _process_bundle
    ctx_bundle = _active_bundle.get()
    if ctx_bundle is not None:
        return ctx_bundle
    from imajin.results import create_result_bundle, user_results_root

    root = user_results_root()
    with _process_bundle_lock:
        if _process_bundle is None:
            _process_bundle = create_result_bundle(
                name="adhoc",
                kind="adhoc",
                root=root,
            )
        return _process_bundle


def bundle_output_path(category: str, filename: str) -> Path:
    """Resolve <bundle>/<category>/<filename>, lazily creating the bundle and parent."""
    bundle = ensure_active_bundle()
    out = bundle / category / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def register_output(
    kind: str,
    path: Path | str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append an entry to the active bundle's metadata.json `outputs` index.

    `path` must already live inside the active bundle; it is recorded as a
    POSIX-relative path. Writes are flushed immediately so consumers can read
    a partial bundle.
    """
    from datetime import UTC, datetime as _datetime

    bundle = ensure_active_bundle()
    target = Path(path).resolve()
    bundle_resolved = bundle.resolve()
    try:
        rel = target.relative_to(bundle_resolved)
    except ValueError as exc:
        raise ValueError(
            f"output {target} is outside the active bundle {bundle_resolved}"
        ) from exc

    record = {
        "kind": kind,
        "path": rel.as_posix(),
        "created_at": _datetime.now(UTC).isoformat(),
        "metadata": dict(metadata or {}),
    }
    seed = read_bundle_metadata(bundle)
    outputs = list(seed.get("outputs") or [])
    outputs.append(record)
    seed["outputs"] = outputs
    if "schema_version" not in seed:
        seed["schema_version"] = 3
    write_bundle_metadata(bundle, seed)


def register_table_spec(table_name: str, spec: dict[str, Any]) -> None:
    bundle = ensure_active_bundle()
    seed = read_bundle_metadata(bundle)
    table_specs = dict(seed.get("table_specs") or {})
    table_specs[str(table_name)] = dict(spec)
    seed["table_specs"] = table_specs
    if "schema_version" not in seed:
        seed["schema_version"] = 3
    write_bundle_metadata(bundle, seed)


def current_bundle() -> Path | None:
    ctx = _active_bundle.get()
    if ctx is not None:
        return ctx
    with _process_bundle_lock:
        return _process_bundle


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
    if max_label <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_label <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def write_label_tiff(path: Path, labels: np.ndarray) -> None:
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tifffile.imwrite(path, labels, compression="zlib", photometric="minisblack")
    except Exception:  # noqa: BLE001
        tifffile.imwrite(path, labels, photometric="minisblack")


def write_label_layer(bundle: Path, tier: str, sample_slug: str, layer_name: str) -> str:
    """Snapshot a label layer and write it to bundle/labels/<tier>/<slug>.tif."""
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
    write_label_tiff(out, labels)
    return rel.as_posix()


def copy_qc_png(bundle: Path, qc_png: str, sample_slug: str) -> str | None:
    src = normalize_user_path(qc_png).resolve()
    if not src.exists():
        return None
    rel = Path("qc") / f"{sample_slug}.png"
    dst = bundle / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
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
    sv = metadata.get("schema_version")
    if sv in (2, 3):
        return {
            "schema_version": sv,
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
    samples_list = [_redact_sample(s) for s in samples]
    run_context = {
        **dict(normalized.get("run_context") or {}),
        "status": status,
        "finalized_at": _kst_now_iso(),
        "samples": samples_list,
        "n_samples": len(samples_list),
        "n_complete": sum(1 for s in samples_list if s.get("status") == "complete"),
        "n_failed": sum(1 for s in samples_list if s.get("status") == "failed"),
        **run_context_extras,
        **extra,
    }
    # Drop fields that schema_v3 no longer carries.
    run_context.pop("tables", None)

    write_bundle_metadata(
        bundle,
        {
            "schema_version": 3,
            "recipe_params": dict(recipe_params),
            "run_context": run_context,
            "environment": environment,
            "table_specs": dict(seed.get("table_specs") or {}),
            "outputs": list(seed.get("outputs") or []),
        },
    )


def _redact_sample(sample: dict[str, Any]) -> dict[str, Any]:
    out = dict(sample)
    out.pop("outputs", None)  # filesystem mirror removed
    summary = dict(out.get("summary") or {})
    summary.pop("qc_warnings", None)
    out["summary"] = summary
    return out


def _kst_now_iso() -> str:
    from imajin.results import _kst_now

    return _kst_now().isoformat()


def start_analysis(
    name: str,
    *,
    kind: str = "single",
    tier: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Create a named bundle, write seed metadata.json (schema_v3, status='in_progress'),
    and set it as the active bundle for the calling context."""
    from imajin.results import create_result_bundle, user_results_root

    root = user_results_root()
    bundle = create_result_bundle(
        name=name,
        kind=kind,
        tier=tier,
        metadata=metadata,
        root=root,
    )
    # Rewrite as schema_v3 seed so all downstream readers see schema_version=3.
    seed = read_bundle_metadata(bundle)
    normalized = _normalize_bundle_metadata(seed)
    run_context = dict(normalized.get("run_context") or {})
    environment = dict(normalized.get("environment") or {})
    write_bundle_metadata(
        bundle,
        {
            "schema_version": 3,
            "recipe_params": dict(normalized.get("recipe_params") or {}),
            "run_context": run_context,
            "environment": environment,
            "table_specs": {},
            "outputs": [],
        },
    )
    # Promote the bundle into the process-global slot so cross-call tool writes
    # share it without a containing with-block.
    global _process_bundle
    with _process_bundle_lock:
        _process_bundle = bundle
    return bundle


# Per-kind dedup keys; rows with the same key replace earlier rows.
_STATS_KEY_FIELDS = {
    "describe": ("value_col", "level", "sample_aggregation", "group"),
    "compare": ("value_col", "test", "data_level", "group_a", "group_b"),
    "timecourse_features": ("value_col", "sample_name", "label"),
}


def register_stats_rows(
    *,
    kind: str,
    table: str,
    rows: list[dict[str, Any]],
) -> None:
    """Merge stats rows into `<bundle>/stats/<kind>__<table>.csv` (long format).

    Rows are deduplicated by the kind-specific key fields; later rows replace
    earlier ones with the same key. The destination CSV is rewritten on every
    call so partial bundles are readable.
    """
    if kind not in _STATS_KEY_FIELDS:
        raise ValueError(f"unsupported stats kind {kind!r}")
    if not rows:
        return

    import pandas as pd

    from imajin.results import slugify_result_name

    bundle = ensure_active_bundle()
    target = bundle / "stats" / f"{kind}__{slugify_result_name(table)}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(rows)
    # Normalize empty strings to NaN in key columns so they round-trip through
    # CSV identically (pandas read_csv reads empty fields as NaN).
    key_fields = _STATS_KEY_FIELDS[kind]
    for col in key_fields:
        if col in new_df.columns:
            new_df[col] = new_df[col].replace("", float("nan"))

    if target.exists():
        existing = pd.read_csv(target)
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df

    key_cols = [c for c in key_fields if c in combined.columns]
    if key_cols:
        combined = combined.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)

    combined.to_csv(target, index=False)
    register_output(
        f"stats_{kind}",
        target,
        {"table": table, "n_rows": int(len(combined))},
    )


def finalize_analysis(
    *,
    status: str = "complete",
    samples: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Finalize the currently active bundle. Writes schema_v3 metadata.json with the
    final status and clears the process slot."""
    bundle = ensure_active_bundle()
    finalize_bundle_metadata(
        bundle,
        samples=list(samples or []),
        status=status,
        extra=dict(extra or {}),
    )
    reset_process_bundle()
    return bundle
