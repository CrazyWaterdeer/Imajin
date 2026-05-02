from __future__ import annotations

from typing import Any

from imajin.tools.registry import tool


@tool(
    description="Create a new Imajin project folder and initialize project JSON files. "
    "Use this before saving a persistent experiment.",
    phase="5",
)
def create_project(path: str, name: str | None = None, notes: str = "") -> dict[str, Any]:
    from imajin.project import create_project as _create_project

    return _create_project(path=path, name=name, notes=notes)


@tool(
    description="Save the current files, sample annotations, channel annotations, "
    "recipes, runs, tables, job history, and provenance into the current Imajin "
    "project folder. Pass path to save-as or to create a project if none is open.",
    phase="5",
    worker=True,
)
def save_project(path: str | None = None) -> dict[str, Any]:
    from imajin.project import save_project as _save_project

    return _save_project(path=path)


@tool(
    description="Load an Imajin project folder. Restores registered files, sample "
    "annotations, channel annotations, recipes, runs, saved tables, and job history "
    "without automatically loading raw microscopy images into napari.",
    phase="5",
)
def load_project(path: str) -> dict[str, Any]:
    from imajin.project import load_project as _load_project

    return _load_project(path=path)


@tool(
    description="Relink a missing or moved raw image file in the current project. "
    "Updates the file record path and saves the project.",
    phase="5",
)
def relink_file(file_id: str, new_path: str) -> dict[str, Any]:
    from imajin.project import relink_file as _relink_file

    return _relink_file(file_id=file_id, new_path=new_path)


@tool(
    description="Return the current project status, including path, counts, missing "
    "raw files, and last autosave result. Does not open a dock or modify data.",
    phase="5",
)
def project_status() -> dict[str, Any]:
    from imajin.project import project_status as _project_status

    return _project_status()


@tool(
    description="Export a human-readable Markdown summary of the current project "
    "without copying raw data or secrets.",
    phase="5",
)
def export_project_summary(path: str) -> dict[str, Any]:
    from imajin.project import export_project_summary as _export_project_summary

    return _export_project_summary(path=path)
