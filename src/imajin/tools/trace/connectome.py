from __future__ import annotations

from typing import Any

from imajin.tools.registry import tool


@tool(
    description="Query an external connectome database (neuPrint/FlyWire, Drosophila) for "
    "reference neurons by morphology. Tier-2 backend — currently returns "
    "'not_implemented'. For local, offline morphology search use find_similar_neurons.",
    phase="6B",
    subagent="neural_tracer",
)
def query_connectome(
    skeleton_id: str,
    db: str = "neuprint",
    k: int = 10,
) -> dict[str, Any]:
    db = db.lower().strip()
    if db in {"microns", "allen"}:
        return {
            "skeleton_id": skeleton_id,
            "db": db,
            "matches": [],
            "status": "off_domain",
            "note": f"{db!r} is a mouse connectome; this app targets Drosophila. Not supported.",
        }
    if db not in {"flywire", "neuprint"}:
        raise ValueError(
            f"unknown db {db!r}; expected neuprint|flywire (microns/allen are mouse, off-domain)"
        )
    if db == "neuprint":
        from imajin.analysis.connectome_neuprint import query_neuprint

        # backend/token readiness is resolved without a skeleton lookup, so a bad id
        # returns a graceful status rather than KeyError (the live fetch — pending a
        # token + template registration — is not wired yet).
        result = query_neuprint(None, None, k=k)
        return {"skeleton_id": skeleton_id, "db": db, **result}

    # FlyWire is a later Tier-2 step (heavier CAVE auth); not wired yet.
    return {
        "skeleton_id": skeleton_id,
        "db": db,
        "k": k,
        "matches": [],
        "status": "not_implemented",
        "note": "FlyWire backend not wired yet (Tier 2, after neuPrint).",
    }


