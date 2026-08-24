from __future__ import annotations

from typing import Any

from imajin.result_bundles import (
    finalize_analysis as _finalize_analysis,
    start_analysis as _start_analysis,
)
from imajin.tools.registry import tool


@tool(
    description=(
        "Open a new analysis bundle. All subsequent figure, stats, QC, and table "
        "outputs in this task land inside <root>/<timestamp>_<name>/. Call this at "
        "the start of a user task so the bundle has a meaningful name; otherwise "
        "an ad-hoc bundle is opened lazily on the first output."
    ),
    phase="0",
)
def start_analysis(
    name: str,
    *,
    kind: str = "single",
    tier: str | None = None,
) -> dict[str, Any]:
    bundle = _start_analysis(name=name, kind=kind, tier=tier)
    return {
        "bundle_path": str(bundle),
        "metadata_path": str(bundle / "metadata.json"),
        "name": name,
    }


@tool(
    description=(
        "Finalize the currently active analysis bundle. Writes the final "
        "metadata.json status and clears the process slot so the next analysis "
        "starts fresh. Safe to call even if no bundle has been opened — it is "
        "then a no-op rather than creating an empty one."
    ),
    phase="0",
)
def finalize_analysis(
    status: str = "complete",
) -> dict[str, Any]:
    bundle = _finalize_analysis(status=status)
    if bundle is None:
        return {
            "bundle_path": None,
            "status": status,
            "message": "no analysis bundle was open; nothing to finalize",
        }
    return {
        "bundle_path": str(bundle),
        "status": status,
    }
