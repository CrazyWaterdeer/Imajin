from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from imajin.session import (
    get_file,
    iter_file_records,
    iter_table_entries,
    list_files,
    list_recipes,
    list_runs,
    list_samples,
    put_file,
    put_recipe,
    put_sample,
    put_table,
)
from imajin.paths import normalize_user_path
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
    normalized_files = [str(normalize_user_path(f)) for f in (files or [])]
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


def _coerce_terms(terms: list[str] | None) -> list[str]:
    return [str(term).strip() for term in (terms or []) if str(term).strip()]


def _match_forms(value: str) -> tuple[str, str, str]:
    normalized = " ".join(
        str(value)
        .lower()
        .replace("\\", "/")
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )
    compact = normalized.replace(" ", "")
    alnum = re.sub(r"[^a-z0-9]+", "", normalized)
    return normalized, compact, alnum


def _contains_user_term(haystack: str, term: str) -> bool:
    hay_norm, hay_compact, hay_alnum = _match_forms(haystack)
    term_norm, term_compact, term_alnum = _match_forms(term)
    return (
        bool(term_norm)
        and (
            term_norm in hay_norm
            or term_compact in hay_compact
            or term_alnum in hay_alnum
        )
    )


def _record_search_text(record: dict[str, Any]) -> str:
    parts = [
        record.get("file_id"),
        record.get("original_name"),
        record.get("path"),
        record.get("notes"),
    ]
    return " ".join(str(part) for part in parts if part)


def _record_matches_terms(
    record: dict[str, Any],
    include: list[str] | None,
    exclude: list[str] | None,
) -> bool:
    include_terms = _coerce_terms(include)
    exclude_terms = _coerce_terms(exclude)
    text = _record_search_text(record)
    return all(_contains_user_term(text, term) for term in include_terms) and not any(
        _contains_user_term(text, term) for term in exclude_terms
    )


def _path_matches_terms(
    path: Path,
    include: list[str] | None,
    exclude: list[str] | None,
) -> bool:
    record = {
        "file_id": path.stem,
        "original_name": path.name,
        "path": str(path),
    }
    return _record_matches_terms(record, include, exclude)


def _scan_image_directory(
    root: Path, *, recursive: bool
) -> tuple[list[Path], int, int, list[str]]:
    files: list[Path] = []
    ignored_non_image = 0
    scanned_dirs = 0
    warnings: list[str] = []
    pending = [root]
    while pending:
        directory = pending.pop(0)
        try:
            children = sorted(directory.iterdir(), key=lambda p: p.name.lower())
        except OSError as exc:
            warnings.append(f"Could not scan directory {directory}: {exc}")
            continue
        scanned_dirs += 1
        for child in children:
            try:
                if child.is_dir():
                    if recursive and not child.is_symlink():
                        pending.append(child)
                    continue
                if not child.is_file():
                    continue
            except OSError as exc:
                warnings.append(f"Could not inspect path {child}: {exc}")
                continue
            supported, _file_type = _classify_extension(child.name)
            if supported:
                files.append(child)
            else:
                ignored_non_image += 1
    return files, ignored_non_image, scanned_dirs, warnings


@tool(
    description="Register one or more imaging files with the experiment without "
    "loading them into napari. Use this when the user names files or folders to "
    "include in a batch analysis. Folder inputs are expanded into supported image "
    "files (.lsm/.czi/.tif/.tiff). Returns one record per file with file_id, "
    "supported/missing flags, and any cheap metadata. Set recursive=True only when "
    "subfolders should be scanned. If the user explicitly names a line, condition, "
    "tissue, region, or other filename text, pass it in include so unrelated files "
    "from the same folder are not registered for the requested batch. Filenames are "
    "NOT parsed into condition/replicate/tissue — call annotate_samples for that.",
    phase="3",
)
def register_files(
    paths: list[str],
    recursive: bool = False,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, Any]:

    out: list[dict[str, Any]] = []
    n_unsupported = 0
    n_missing = 0
    n_discarded_by_filter = 0
    n_input_dirs = 0
    n_scanned_dirs = 0
    n_ignored_non_image = 0
    directories: list[dict[str, Any]] = []
    warnings: list[str] = []

    def register_one(p: Path) -> None:
        nonlocal n_unsupported, n_missing, n_discarded_by_filter
        if not _path_matches_terms(p, include, exclude):
            n_discarded_by_filter += 1
            return
        original_name = p.name
        supported, file_type = _classify_extension(original_name)
        exists = p.exists()
        if not supported:
            n_unsupported += 1
        if not exists:
            n_missing += 1
        resolved = p.resolve() if exists else p
        metadata_summary: dict[str, Any] = {}
        if supported and exists:
            try:
                from imajin.io.metadata import read_metadata_summary

                metadata_summary = read_metadata_summary(resolved)
            except Exception as exc:  # noqa: BLE001
                metadata_summary = {
                    "metadata_error": f"{type(exc).__name__}: {exc}",
                    "metadata_read_mode": "metadata_only",
                }
        file_id = put_file(
            path=str(resolved),
            original_name=original_name,
            file_type=file_type,
            metadata_summary=metadata_summary,
        )
        out.append(
            {
                "file_id": file_id,
                "path": str(resolved),
                "original_name": original_name,
                "file_type": file_type,
                "supported": supported,
                "exists": exists,
                "load_status": "unloaded",
                "metadata_summary": metadata_summary,
            }
        )

    for raw in paths:
        p = normalize_user_path(raw)
        if p.is_dir():
            n_input_dirs += 1
            found, ignored, scanned, scan_warnings = _scan_image_directory(
                p, recursive=recursive
            )
            n_scanned_dirs += scanned
            n_ignored_non_image += ignored
            warnings.extend(scan_warnings)
            directories.append(
                {
                    "input": raw,
                    "path": str(p.resolve()),
                    "recursive": recursive,
                    "n_found": len(found),
                    "n_ignored_non_image": ignored,
                    "n_scanned_dirs": scanned,
                }
            )
            for child in found:
                register_one(child)
            continue
        register_one(p)
    result = {
        "n_registered": len(out),
        "n_supported": len(out) - n_unsupported,
        "n_unsupported": n_unsupported,
        "n_missing": n_missing,
        "n_discarded_by_filter": n_discarded_by_filter,
        "files": out,
        "n_input_dirs": n_input_dirs,
        "n_scanned_dirs": n_scanned_dirs,
        "n_ignored_non_image": n_ignored_non_image,
        "directories": directories,
    }
    include_terms = _coerce_terms(include)
    exclude_terms = _coerce_terms(exclude)
    if include_terms:
        result["include"] = include_terms
    if exclude_terms:
        result["exclude"] = exclude_terms
    if warnings:
        result["warnings"] = warnings
    return result


def _query_registered_file_records(
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    file_ids: list[str] | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:

    records = list_files()
    if file_ids:
        wanted = {str(file_id) for file_id in file_ids}
        records = [rec for rec in records if str(rec.get("file_id")) in wanted]

    records = sorted(
        records,
        key=lambda rec: (
            str(rec.get("original_name", "")).lower(),
            str(rec.get("path", "")).lower(),
        ),
    )
    matched = [
        rec
        for rec in records
        if _record_matches_terms(rec, include=include, exclude=exclude)
    ]

    safe_offset = max(0, int(offset))
    safe_limit = max(1, min(int(limit), 200))
    page = matched[safe_offset : safe_offset + safe_limit]
    next_offset = safe_offset + safe_limit
    representative = matched[0] if matched else None

    result: dict[str, Any] = {
        "n_registered": len(records),
        "n_matched": len(matched),
        "n_unmatched": len(records) - len(matched),
        "offset": safe_offset,
        "limit": safe_limit,
        "has_more": next_offset < len(matched),
        "next_offset": next_offset if next_offset < len(matched) else None,
        "files": page,
        "file_ids": [str(rec.get("file_id")) for rec in page],
        "paths": [str(rec.get("path")) for rec in page],
        "representative_file": representative,
        "representative_path": representative.get("path") if representative else None,
    }
    include_terms = _coerce_terms(include)
    exclude_terms = _coerce_terms(exclude)
    if include_terms:
        result["include"] = include_terms
    if exclude_terms:
        result["exclude"] = exclude_terms
    if file_ids:
        result["file_id_filter"] = [str(file_id) for file_id in file_ids]
    return result


@tool(
    description="List registered imaging files from the experiment registry with "
    "optional include/exclude filename text filters and pagination. Use this instead "
    "of relying on old tool output when you need the full file list. If has_more is "
    "true, call again with next_offset. This does not parse filenames into metadata.",
    phase="3",
)
def list_registered_files(
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    file_ids: list[str] | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    return _query_registered_file_records(
        include=include,
        exclude=exclude,
        file_ids=file_ids,
        offset=offset,
        limit=limit,
    )


@tool(
    description="Filter the current registered file registry by exact user-provided "
    "filename/path text, such as a genotype line ('2966 + 1234'), condition "
    "('venerose'), tissue ('midgut'), or replicate token. Use this immediately after "
    "register_files when a folder contains files outside the user's requested scope. "
    "If n_matched is zero, ask the user instead of falling back to unrelated files.",
    phase="3",
)
def filter_registered_files(
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    file_ids: list[str] | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    return _query_registered_file_records(
        include=include,
        exclude=exclude,
        file_ids=file_ids,
        offset=offset,
        limit=limit,
    )


@tool(
    description="Validate acquisition metadata before batch analysis without loading "
    "pixel arrays. For intensity analyses this compares the target channel across "
    "the selected files for laser intensity, detector gain, color bit depth, and "
    "pinhole size. Counterstain/non-target channels are ignored. Use this after "
    "register_files/filter_registered_files and before run_recipe_on_samples when "
    "comparing fluorescence intensity across files. analysis_kind='area' skips "
    "intensity acquisition settings.",
    phase="3",
)
def validate_analysis_metadata(
    paths: list[str] | None = None,
    file_ids: list[str] | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    target_channel: str | None = None,
    analysis_kind: str = "auto",
    measurement: dict[str, Any] | None = None,
    strict_missing: bool = True,
) -> dict[str, Any]:
    from imajin.analysis.metadata_validation import validate_acquisition_metadata

    records: list[dict[str, Any]] = []
    if paths:
        for raw in paths:
            p = normalize_user_path(raw)
            records.append({"path": str(p.resolve() if p.exists() else p)})

    file_id_filter = {str(fid) for fid in (file_ids or [])}
    for rec in list_files():
        if file_id_filter and str(rec.get("file_id")) not in file_id_filter:
            continue
        if not file_id_filter and paths:
            continue
        if not _record_matches_terms(rec, include=include, exclude=exclude):
            continue
        records.append(
            {
                "path": rec.get("path"),
                "file_id": rec.get("file_id"),
                "metadata_summary": rec.get("metadata_summary"),
            }
        )

    return validate_acquisition_metadata(
        records,
        target_channel=target_channel,
        analysis_kind=analysis_kind,
        measurement=measurement,
        strict_missing=strict_missing,
    )


def _resolve_files_for_sample(
    files: list[str] | None,
    file_ids: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Return (file_paths, file_ids). Either input can be empty.
    Paths are matched against registered FileRecords; unmatched paths are
    accepted but get no file_id."""

    resolved_paths: list[str] = []
    resolved_ids: list[str] = list(file_ids or [])
    by_path = {rec.path: rec for rec in iter_file_records()}

    for raw in files or []:
        raw_path = normalize_user_path(raw)
        p = str(raw_path.resolve())
        rec = by_path.get(p) or by_path.get(str(raw_path))
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

    return {
        "files": list_files(),
        "samples": list_samples(),
        "recipes": list_recipes(),
        "runs": list_runs(),
    }


@tool(
    description="Return batch analysis progress: which files have been analysed "
    "(with their result table), which are still pending, which failed, and the next "
    "pending file. Use this to continue a multi-file batch without re-analysing a "
    "finished file or re-asking for a known path. A file is 'pending' only if it was "
    "registered (register_files); otherwise the universe is unknown.",
    phase="3",
)
def get_batch_progress() -> dict[str, Any]:
    from imajin.agent.context import batch_progress_data

    return batch_progress_data()


@tool(
    description="Create or replace a reusable analysis recipe. Captures target "
    "channel, segmentation/measurement/preprocessing settings, and optional "
    "time-course or colocalization parameters so the same pipeline can be applied "
    "across many samples in a batch. `segmentation` is the Tier-2 step and must "
    "use one of method='target_objects' | 'cellpose_sam' | 'intensity_regions'. "
    "For two-tier expression-domain analysis, put the Tier-1 mask spec into the "
    "separate `domain` slot, e.g. domain={'strategy':'noise_floor','k_mad':6.25,"
    "'dark_percentile':10.0,'min_area_um2':5.0}; do NOT put 'expression_domain' in "
    "the segmentation slot. Optional cell_diameter_um drives Tier-2 size derivation.",
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
    cell_diameter_um: float | None = None,
    domain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from imajin.tools.workflows import (
        _normalize_domain_spec,
        _normalize_segmentation_method,
    )

    if segmentation:
        raw_method = segmentation.get("tool") or segmentation.get("method")
        if raw_method:
            _normalize_segmentation_method(raw_method)
    _normalize_domain_spec(domain)

    recipe_id = put_recipe(
        name=name,
        target_channel=target_channel,
        segmentation=segmentation,
        measurement=measurement,
        preprocessing=preprocessing,
        timecourse=timecourse,
        colocalization=colocalization,
        notes=notes,
        cell_diameter_um=cell_diameter_um,
        domain=domain,
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


    frames: list[pd.DataFrame] = []
    for entry in iter_table_entries():
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
