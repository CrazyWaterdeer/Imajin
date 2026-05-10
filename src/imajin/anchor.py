from __future__ import annotations

from pathlib import Path
from typing import Iterable


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
        parent = Path(p).expanduser().absolute().parent
        parents.add(parent)
    if not parents:
        return None
    return sorted(parents, key=lambda p: str(p).lower())[0]
