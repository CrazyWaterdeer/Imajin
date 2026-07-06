"""Read-only tools for resuming a batch analysis from an existing result bundle.

`plan_resume` is the "agent reads the bundle" entry point: point it at a folder and
it finds the prior bundle, recovers its recipe, and diffs analysed-vs-pending files —
without mutating anything. Committing to the resume (promote the bundle, seed the
guard) is a separate step (`open_result_bundle`, P2). See the resume design spec.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from imajin.tools.registry import tool

_IMAGE_EXTS = (".lsm", ".czi", ".ome.tif", ".ome.tiff", ".tif", ".tiff")


def _scan_image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [
        p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.name.lower().endswith(_IMAGE_EXTS)
    ]


def _bundle_dirs_under(root: str | Path | None) -> list[Path]:
    if not root:
        return []
    r = Path(root).expanduser()
    if not r.is_dir():
        return []
    try:
        children = sorted(r.iterdir())
    except OSError:
        return []
    return [c for c in children if c.is_dir() and (c / "metadata.json").exists()]


def _bundle_brief(bundle: Path) -> dict[str, Any]:
    from imajin.result_bundles import read_sample_index
    from imajin.results import read_bundle_metadata

    meta = read_bundle_metadata(bundle)
    idx = read_sample_index(bundle)
    complete = [e for e in idx["entries"] if e.get("status") == "complete"]
    return {
        "path": str(bundle),
        "name": meta.get("name") or bundle.name,
        "kind": meta.get("kind"),
        "tier": meta.get("tier"),
        "input_anchor": idx.get("input_anchor") or meta.get("input_anchor"),
        "n_analysed": len(complete),
        "legacy_inferred": idx["legacy_inferred"],
        "created_at": meta.get("created_at"),
    }


def _find_bundles(directory: str) -> list[dict[str, Any]]:
    from imajin.analysis.resume import bundle_belongs_to_dir
    from imajin.results import user_results_root

    d = Path(directory).expanduser()
    briefs: dict[str, dict[str, Any]] = {}
    for bundle in _bundle_dirs_under(d) + _bundle_dirs_under(user_results_root()):
        key = str(bundle.resolve())
        if key in briefs:
            continue
        brief = _bundle_brief(bundle)
        if bundle_belongs_to_dir(brief["input_anchor"], bundle, d):
            briefs[key] = brief
    return sorted(briefs.values(), key=lambda b: b.get("created_at") or "", reverse=True)


def _read_bundle(bundle_path: str) -> dict[str, Any]:
    from imajin.result_bundles import read_bundle_metadata_normalized, read_sample_index

    bundle = Path(bundle_path).expanduser()
    meta = read_bundle_metadata_normalized(bundle)
    idx = read_sample_index(bundle)
    complete = [e for e in idx["entries"] if e.get("status") == "complete"]
    return {
        "bundle": str(bundle),
        "recipe_params": dict(meta.get("recipe_params") or {}),
        "analysed_keys": [e["key"] for e in complete if e.get("key")],
        "n_analysed": len(complete),
        "legacy_inferred": idx["legacy_inferred"],
        "input_anchor": idx.get("input_anchor"),
    }


@tool(
    description="List prior result bundles that cover an input folder (by stored input "
    "anchor, or bundles written directly under it). Use before resuming a half-finished "
    "batch so you can see what was already analysed. Read-only.",
    phase="7",
)
def find_result_bundles(directory: str) -> dict[str, Any]:
    bundles = _find_bundles(directory)
    return {"directory": str(Path(directory).expanduser()), "n_bundles": len(bundles), "bundles": bundles}


@tool(
    description="Read a result bundle's recipe and the list of source files it already "
    "contains results for (anchor-relative keys). Read-only; recovers the exact analysis "
    "settings so a resume reuses identical parameters.",
    phase="7",
)
def read_result_bundle(bundle_path: str) -> dict[str, Any]:
    return _read_bundle(bundle_path)


@tool(
    description="Plan how to resume analysing a folder from its prior result bundle: find "
    "the bundle, recover its recipe, and diff analysed vs pending files. Read-only — makes "
    "no changes. If no bundle covers the folder, says so; if more than one does, returns "
    "them and asks you to choose. Follow an 'ok' plan with open_result_bundle to commit.",
    phase="7",
)
def plan_resume(directory: str) -> dict[str, Any]:
    from imajin.analysis.resume import diff_keys, rel_key

    d = Path(directory).expanduser()
    bundles = _find_bundles(str(d))
    if not bundles:
        return {
            "status": "no_bundle",
            "directory": str(d),
            "note": "No prior result bundle covers this folder — start a fresh analysis.",
        }
    if len(bundles) > 1:
        return {
            "status": "multiple_bundles",
            "directory": str(d),
            "bundles": bundles,
            "note": "More than one bundle covers this folder — ask the user which to resume.",
        }
    info = _read_bundle(bundles[0]["path"])
    disk_keys = [rel_key(f, d) for f in _scan_image_files(d)]
    diff = diff_keys(info["analysed_keys"], disk_keys)
    pending = diff["pending"]
    return {
        "status": "ok",
        "bundle": info["bundle"],
        "recipe_params": info["recipe_params"],
        "legacy_inferred": info["legacy_inferred"],
        "analysed": diff["analysed"],
        "pending": pending,
        "missing": diff["missing"],
        "n_pending": len(pending),
        "note": (
            "Resume: open_result_bundle(bundle) to import the recipe and skip done files, "
            "then analyse the pending ones into the same bundle."
            if pending
            else "Nothing pending — every file under this folder is already in the bundle."
        ),
    }
