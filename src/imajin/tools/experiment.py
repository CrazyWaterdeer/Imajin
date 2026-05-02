from __future__ import annotations

from pathlib import Path
from typing import Any

from imajin.agent.state import list_samples, put_sample
from imajin.tools.registry import tool


@tool(
    description="Annotate a sample/replicate with its experimental group. Use this "
    "when the user says which files or layers belong to control, treatment, genotype, "
    "condition, or replicate groups. These annotations are used by reports and future "
    "batch summaries.",
    phase="1.5",
)
def annotate_sample(
    sample_name: str,
    group: str,
    layers: list[str] | None = None,
    files: list[str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if not group or not group.strip():
        raise ValueError("group must not be empty for annotate_sample()")
    normalized_files = [str(Path(f).expanduser()) for f in (files or [])]
    name = put_sample(
        sample_name=sample_name,
        group=group,
        layers=list(layers or []),
        files=normalized_files,
        notes=notes,
    )
    return {
        "sample_name": name,
        "group": group,
        "layers": list(layers or []),
        "files": normalized_files,
        "notes": notes,
    }


@tool(
    description="List current sample/group annotations for this analysis session. "
    "Use before group-level summaries or reports.",
    phase="1.5",
)
def list_sample_annotations() -> list[dict[str, Any]]:
    return list_samples()


_SUPPORTED_EXTS = {".lsm", ".czi", ".tif", ".tiff", ".ome.tif", ".ome.tiff"}


def _classify_extension(name: str) -> tuple[bool, str | None]:
    lower = name.lower()
    for ext in sorted(_SUPPORTED_EXTS, key=len, reverse=True):
        if lower.endswith(ext):
            return True, ext.lstrip(".")
    return False, None


@tool(
    description="Register one or more imaging files with the experiment without "
    "loading them into napari. Use this when the user names files or folders to "
    "include in a batch analysis. Returns one record per file with file_id, "
    "supported/missing flags, and any cheap metadata. Filenames are NOT parsed "
    "into condition/replicate/tissue — call annotate_samples for that.",
    phase="3",
)
def register_files(paths: list[str]) -> dict[str, Any]:
    from imajin.agent.state import put_file

    out: list[dict[str, Any]] = []
    n_unsupported = 0
    n_missing = 0
    for raw in paths:
        p = Path(raw).expanduser()
        original_name = p.name
        supported, file_type = _classify_extension(original_name)
        exists = p.exists()
        if not supported:
            n_unsupported += 1
        if not exists:
            n_missing += 1
        file_id = put_file(
            path=str(p.resolve() if exists else p),
            original_name=original_name,
            file_type=file_type,
        )
        out.append(
            {
                "file_id": file_id,
                "path": str(p.resolve() if exists else p),
                "original_name": original_name,
                "file_type": file_type,
                "supported": supported,
                "exists": exists,
                "load_status": "unloaded",
            }
        )
    return {
        "n_registered": len(out),
        "n_supported": len(out) - n_unsupported,
        "n_unsupported": n_unsupported,
        "n_missing": n_missing,
        "files": out,
    }


def _resolve_files_for_sample(
    files: list[str] | None,
    file_ids: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Return (file_paths, file_ids). Either input can be empty.
    Paths are matched against registered FileRecords; unmatched paths are
    accepted but get no file_id."""
    from imajin.agent.state import _FILES, get_file

    resolved_paths: list[str] = []
    resolved_ids: list[str] = list(file_ids or [])
    by_path = {rec.path: rec for rec in _FILES.values()}

    for raw in files or []:
        p = str(Path(raw).expanduser().resolve())
        rec = by_path.get(p) or by_path.get(str(Path(raw).expanduser()))
        if rec is not None:
            resolved_paths.append(rec.path)
            if rec.file_id not in resolved_ids:
                resolved_ids.append(rec.file_id)
        else:
            resolved_paths.append(p)

    # If user passed file_ids only, fill in paths from the registry.
    for fid in file_ids or []:
        try:
            rec = get_file(fid)
        except KeyError:
            continue
        if rec.path not in resolved_paths:
            resolved_paths.append(rec.path)

    return resolved_paths, resolved_ids


@tool(
    description="Bulk-annotate samples with user-confirmed group/condition/replicate "
    "metadata. Pass a list of dicts, each with sample_name (required), group, files "
    "(paths) or file_ids (registered ids), layers, notes, and extra (a dict of "
    "user-confirmed fields like genotype/tissue/region/replicate). The agent must "
    "never invent these fields from filenames — only store what the user confirmed.",
    phase="3",
)
def annotate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    from imajin.agent.state import put_sample

    out: list[dict[str, Any]] = []
    for entry in samples:
        if "sample_name" not in entry:
            raise ValueError("each sample must have a sample_name")
        files, file_ids = _resolve_files_for_sample(
            entry.get("files"), entry.get("file_ids")
        )
        name = put_sample(
            sample_name=entry["sample_name"],
            group=entry.get("group"),
            layers=list(entry.get("layers") or []),
            files=files,
            file_ids=file_ids,
            notes=entry.get("notes"),
            sample_id=entry.get("sample_id"),
            extra=dict(entry.get("extra") or {}),
        )
        out.append(
            {
                "sample_name": name,
                "group": entry.get("group"),
                "file_ids": file_ids,
                "files": files,
                "extra": dict(entry.get("extra") or {}),
            }
        )
    return {"n_samples": len(out), "samples": out}


@tool(
    description="Return the current experiment session state: registered files, "
    "sample annotations, analysis recipes, and runs. Use this before batch "
    "analysis or report generation to confirm the experiment shape with the user.",
    phase="3",
)
def list_experiment() -> dict[str, Any]:
    from imajin.agent.state import (
        list_files,
        list_recipes,
        list_runs,
        list_samples,
    )

    return {
        "files": list_files(),
        "samples": list_samples(),
        "recipes": list_recipes(),
        "runs": list_runs(),
    }


@tool(
    description="Create or replace a reusable analysis recipe. Captures target "
    "channel, segmentation/measurement/preprocessing settings, and optional "
    "time-course or colocalization parameters so the same pipeline can be applied "
    "across many samples in a batch.",
    phase="3",
)
def create_analysis_recipe(
    name: str,
    target_channel: str | None = None,
    segmentation: dict[str, Any] | None = None,
    measurement: dict[str, Any] | None = None,
    preprocessing: list[dict[str, Any]] | None = None,
    timecourse: dict[str, Any] | None = None,
    colocalization: list[tuple[str, str]] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    from imajin.agent.state import put_recipe

    recipe_id = put_recipe(
        name=name,
        target_channel=target_channel,
        segmentation=segmentation,
        measurement=measurement,
        preprocessing=preprocessing,
        timecourse=timecourse,
        colocalization=colocalization,
        notes=notes,
    )
    return {
        "recipe_id": recipe_id,
        "name": recipe_id,
        "target_channel": target_channel,
    }


def _scan_measurement_tables(measurement: str) -> "pd.DataFrame":
    """Concatenate every Phase-3 measurement table that has the requested column
    plus the sample/group identifier columns."""
    import pandas as pd

    from imajin.agent.state import _TABLES

    frames: list[pd.DataFrame] = []
    for entry in _TABLES.values():
        df = entry.df
        if df is None or df.empty:
            continue
        needed = {"sample_name", "sample_id", "group", measurement}
        if not needed.issubset(df.columns):
            continue
        frames.append(df)
    if not frames:
        raise ValueError(
            f"No measurement tables found containing column {measurement!r} "
            "alongside sample_name/sample_id/group. Run a recipe first."
        )
    return pd.concat(frames, ignore_index=True)


@tool(
    description="Aggregate per-object measurements into sample-level and "
    "group-level summary tables. Pass the measurement column name (e.g. "
    "'mean_intensity_green_target', 'area_um2'). Sample-level: count, mean, "
    "median, std, sem per sample. Group-level: mean of sample means, "
    "n_samples, and n_objects per group.",
    phase="3",
)
def summarize_experiment(
    measurement: str,
    group_by: str = "group",
    sample_col: str = "sample_name",
) -> dict[str, Any]:
    import pandas as pd

    from imajin.agent.state import put_table

    df = _scan_measurement_tables(measurement)
    sample_grp = df.groupby(sample_col, dropna=False)[measurement]
    sample_summary = sample_grp.agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        sem="sem",
    ).reset_index()

    sample_to_group = (
        df[[sample_col, group_by]]
        .drop_duplicates(subset=[sample_col])
        .set_index(sample_col)
    )
    sample_summary[group_by] = sample_summary[sample_col].map(
        sample_to_group[group_by]
    )

    group_summary = (
        sample_summary.groupby(group_by, dropna=False)
        .agg(
            n_samples=(sample_col, "nunique"),
            mean=("mean", "mean"),
            median=("median", "mean"),
            std=("std", "mean"),
            sem=("sem", "mean"),
        )
        .reset_index()
    )
    object_counts = df.groupby(group_by, dropna=False)[measurement].size()
    group_summary["n_objects"] = (
        group_summary[group_by].map(object_counts).astype(int)
    )

    sample_table_name = put_table(
        f"summary_sample__{measurement}",
        sample_summary,
        spec={"tool": "summarize_experiment", "measurement": measurement, "level": "sample"},
    )
    group_table_name = put_table(
        f"summary_group__{measurement}",
        group_summary,
        spec={"tool": "summarize_experiment", "measurement": measurement, "level": "group"},
    )
    return {
        "measurement": measurement,
        "sample_table": sample_table_name,
        "group_table": group_table_name,
        "n_samples": int(sample_summary[sample_col].nunique()),
        "n_groups": int(group_summary[group_by].nunique(dropna=False)),
    }
