from __future__ import annotations

import pytest

from imajin.agent import state
from imajin.result_bundles import finalize_bundle_metadata
from imajin.results import create_result_bundle, write_bundle_metadata
from imajin.tools.recipe_import import import_recipe_from_bundle


def test_import_recipe_from_bundle_registers_recipe_params(tmp_path) -> None:
    bundle = create_result_bundle("old_run", root=tmp_path, kind="batch")
    finalize_bundle_metadata(
        bundle,
        samples=[],
        status="complete",
        extra={
            "recipe_params": {
                "name": "old_recipe",
                "target_channel": "green",
                "preprocessing": [{"step": "rolling_ball", "radius": 25}],
                "segmentation": {"tool": "intensity_regions"},
                "measurement": {"properties": ["area", "mean_intensity"]},
                "domain": {"strategy": "noise_floor", "k_mad": 5.0},
                "cell_diameter_um": 10.0,
            }
        },
    )

    res = import_recipe_from_bundle(str(bundle), name="new_recipe")
    recipe = state.get_recipe("new_recipe")

    assert res["recipe_name"] == "new_recipe"
    assert res["source_bundle"] == str(bundle.resolve())
    assert recipe.target_channel == "green"
    assert recipe.preprocessing == [{"step": "rolling_ball", "radius": 25}]
    assert recipe.segmentation == {"tool": "intensity_regions"}
    assert recipe.measurement == {"properties": ["area", "mean_intensity"]}
    assert recipe.domain == {"strategy": "noise_floor", "k_mad": 5.0}
    assert recipe.cell_diameter_um == 10.0
    assert not state.list_samples()
    assert not state.list_channel_annotations()


def test_import_recipe_from_bundle_rejects_missing_recipe_params(tmp_path) -> None:
    bundle = tmp_path / "empty_bundle"
    write_bundle_metadata(
        bundle,
        {
            "schema_version": 2,
            "recipe_params": {},
            "run_context": {"kind": "batch"},
            "environment": {},
        },
    )

    with pytest.raises(ValueError, match="No recipe_params"):
        import_recipe_from_bundle(str(bundle))
