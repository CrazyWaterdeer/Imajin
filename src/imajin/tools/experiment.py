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
