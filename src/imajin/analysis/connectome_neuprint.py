"""neuPrint connectome backend (Tier-2, optional + credential-gated).

Resolves the neuPrint access token and reports backend readiness. The live
query path (fetch reference neurons → register the query into template space →
NBLAST) is **not** wired here: it requires a token, network access, and a
decision on how to register arbitrary confocal traces into the hemibrain template
space (without that registration, cross-dataset NBLAST is not meaningful). Until
those are provided, ``query_neuprint`` degrades to a typed status rather than
returning unverified results.

``navis`` / ``neuprint`` are imported lazily; this module imports clean without the
``connectome`` extra.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any

from imajin.analysis.morphology_nblast import navis_available

_TOKEN_ENV = "NEUPRINT_APPLICATION_CREDENTIALS"


def neuprint_available() -> bool:
    return importlib.util.find_spec("neuprint") is not None


def neuprint_token() -> str | None:
    return os.environ.get(_TOKEN_ENV) or None


def neuprint_status() -> dict[str, Any]:
    return {
        "backend": "neuprint",
        "package_available": neuprint_available(),
        "navis_available": navis_available(),
        "token_present": neuprint_token() is not None,
        "token_env": _TOKEN_ENV,
    }


def query_neuprint(
    query_points: Any,
    query_units: Any,
    *,
    k: int = 10,
    dataset: str | None = None,
    server: str = "neuprint.janelia.org",
) -> dict[str, Any]:
    """Resolve neuPrint backend readiness; the live fetch is not yet wired.

    Returns a typed ``status``: ``backend_unavailable`` (no navis/neuprint extra),
    ``needs_token`` (extra present, no token), or ``needs_registration`` (token
    present — the remaining blocker is template registration of the query).
    """
    if not (navis_available() and neuprint_available()):
        return {
            "status": "backend_unavailable",
            "matches": [],
            "note": "Install the connectome extra: uv sync --extra connectome",
        }
    if neuprint_token() is None:
        return {
            "status": "needs_token",
            "matches": [],
            "note": f"Set a neuPrint token in ${_TOKEN_ENV} (neuprint.janelia.org → your account).",
        }
    return {
        "status": "needs_registration",
        "matches": [],
        "note": (
            "Token found. Remaining work before live lookup: register the query into "
            "the hemibrain template space (navis-flybrains) so NBLAST against neuPrint "
            "neurons is meaningful, then fetch_neurons/fetch_skeleton + NBLAST. This "
            "path needs network access to verify."
        ),
    }
