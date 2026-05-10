from __future__ import annotations

from pathlib import Path
from typing import Any

from imajin.agent.state import put_recipe
from imajin.result_bundles import read_bundle_metadata_normalized
from imajin.tools.registry import tool


@tool(
    description=(
        "Read a previous result bundle's metadata.json and register its "
        "recipe_params as a session recipe. Use this when the user points at a "
        "prior bundle and asks to apply the same analysis settings. Sample "
        "annotations, channel roles, and file scope are not imported."
    ),
    phase="7",
)
def import_recipe_from_bundle(
    bundle_path: str,
    name: str | None = None,
) -> dict[str, Any]:
    bundle = Path(bundle_path).expanduser().resolve()
    meta = read_bundle_metadata_normalized(bundle)
    recipe_params = dict(meta.get("recipe_params") or {})
    if not recipe_params:
        raise ValueError(f"No recipe_params found in {bundle / 'metadata.json'}")

    recipe_name = name or recipe_params.get("name") or bundle.name
    put_recipe(
        name=str(recipe_name),
        target_channel=recipe_params.get("target_channel"),
        preprocessing=recipe_params.get("preprocessing") or [],
        segmentation=recipe_params.get("segmentation") or {},
        measurement=recipe_params.get("measurement") or {},
        domain=recipe_params.get("domain"),
        cell_diameter_um=recipe_params.get("cell_diameter_um"),
    )
    return {
        "recipe_name": str(recipe_name),
        "source_bundle": str(bundle),
        "imported": {
            "target_channel": recipe_params.get("target_channel"),
            "preprocessing": recipe_params.get("preprocessing") or [],
            "segmentation": recipe_params.get("segmentation") or {},
            "measurement": recipe_params.get("measurement") or {},
            "domain": recipe_params.get("domain"),
            "cell_diameter_um": recipe_params.get("cell_diameter_um"),
        },
        "note": (
            "Sample annotations, channel roles, and file scope were not imported. "
            "Register and annotate the current data before calling run_recipe_on_samples."
        ),
    }
