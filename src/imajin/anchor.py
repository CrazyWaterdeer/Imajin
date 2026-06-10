from __future__ import annotations

from pathlib import Path
from typing import Iterable

from imajin.paths import normalize_user_path


def resolve_anchor_folder(file_paths: Iterable[str | Path]) -> Path | None:
    """Pick the anchor folder for a set of input file paths.

    Each path is canonicalized lexically (no filesystem I/O, no symlink
    resolution) via :meth:`pathlib.Path.expanduser` + :meth:`pathlib.Path.absolute`,
    and its parent is collected. The unique set is sorted case-insensitively;
    the first entry wins. Returns ``None`` when the input is empty or every
    entry is falsy.
    """
    parents: set[Path] = set()
    for p in file_paths:
        if not p:
            continue
        parent = normalize_user_path(p).expanduser().absolute().parent
        parents.add(parent)
    if not parents:
        return None
    return sorted(parents, key=lambda p: str(p).lower())[0]


def resolve_session_anchor(extra_paths: Iterable[str | Path] | None = None) -> Path | None:
    """Resolve the anchor folder from session-registered files plus optional extras.

    Pulls file paths from :func:`imajin.session.list_files` (each record's
    ``path`` field) and any ``extra_paths``, then defers to
    :func:`resolve_anchor_folder`. Returns ``None`` if no usable paths exist.
    """
    # Kept function-local on purpose: tests monkeypatch ``imajin.session.list_files``,
    # and a module-level ``from imajin.session import list_files`` would bind the
    # original at import time and defeat the patch. (Not a cycle workaround.)
    from imajin.session import list_files

    paths: list[str | Path] = []
    for rec in list_files():
        path = rec.get("path") if isinstance(rec, dict) else getattr(rec, "path", None)
        if path:
            paths.append(path)
    if extra_paths:
        paths.extend(p for p in extra_paths if p)
    return resolve_anchor_folder(paths)
