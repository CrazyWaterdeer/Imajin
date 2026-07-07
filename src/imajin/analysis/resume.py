"""Identity + slug helpers for resuming a batch analysis from a result bundle.

Pure and headless (no napari, no session). A source file's identity is its path
**relative to an input anchor** (the common data folder), normalised to forward
slashes, so the same file matches across mounts/platforms — WSL ``/mnt/d/exp/a.lsm``
and Windows ``D:\\exp\\a.lsm`` under a shared anchor both key to ``a.lsm``. This is
what makes cross-session/cross-machine resume work; absolute paths would not match.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from imajin.results import slugify_result_name


def rel_key(source_path: str | os.PathLike[str], anchor: str | os.PathLike[str]) -> str:
    """Source-file identity: its path relative to ``anchor``, forward-slashed.

    Uses ``os.path.relpath`` so a file outside the anchor still gets a stable
    ``../…`` key instead of raising. Case is preserved (POSIX is case-sensitive);
    the :func:`file_signature` size/mtime is the cross-platform conflict backstop.
    """
    rel = os.path.relpath(os.fspath(source_path), os.fspath(anchor))
    return rel.replace("\\", "/")


def file_signature(path: str | os.PathLike[str]) -> dict[str, int] | None:
    """``{size, mtime}`` for staleness/conflict detection, or ``None`` if missing."""
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return {"size": int(st.st_size), "mtime": int(st.st_mtime)}


def sample_slug_for(key: str) -> str:
    """Deterministic, collision-resistant slug from a rel-key.

    ``<slug(stem)>_<8 hex of sha1(key)>`` — re-appending the same file yields the
    same slug (idempotent); two different files never collide even when their stems
    match. Computed from the stable key so the decision is made before analysis.
    """
    stem = Path(key).stem or "sample"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"{slugify_result_name(stem)}_{digest}"


def _canonical_dir(path: str | os.PathLike[str]) -> str:
    return str(Path(path).expanduser().absolute()).replace("\\", "/").rstrip("/")


def bundle_belongs_to_dir(
    input_anchor: str | None, bundle_path: str | os.PathLike[str], directory: str | os.PathLike[str]
) -> bool:
    """Does this bundle cover ``directory``?

    True when the bundle's stored input anchor is that directory, or the bundle
    folder lives directly inside it (the batch case where ``root=anchor``). Anchor
    match is by absolute path, so cross-machine matching falls back to the
    lives-inside test; passing the bundle path explicitly always works.
    """
    d = _canonical_dir(directory)
    if input_anchor and _canonical_dir(input_anchor) == d:
        return True
    return _canonical_dir(Path(bundle_path).parent) == d


def diff_keys(
    analysed_keys: list[str] | set[str], present_keys: list[str] | set[str]
) -> dict[str, list[str]]:
    """Three-way diff of rel-keys for resume.

    ``analysed`` = complete in the bundle; ``pending`` = on disk but not in the
    bundle (still to do); ``missing`` = in the bundle but no longer on disk.
    """
    analysed, present = set(analysed_keys), set(present_keys)
    return {
        "analysed": sorted(analysed),
        "pending": sorted(present - analysed),
        "missing": sorted(analysed - present),
    }
