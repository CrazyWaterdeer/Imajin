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
