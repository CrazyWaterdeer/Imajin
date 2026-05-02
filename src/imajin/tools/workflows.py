from __future__ import annotations

from typing import Any

from imajin.agent.qt_dispatch import call_on_main
from imajin.agent.state import (
    AmbiguousChannelError,
    resolve_target_channel,
)
from imajin.tools import measure as _measure
from imajin.tools import preprocess as _preprocess
from imajin.tools import segment as _segment
from imajin.tools.napari_ops import snapshot_layer
from imajin.tools.registry import tool


_VALID_PREPROCESS = {
    "rolling_ball": "rolling_ball",
    "rb": "rolling_ball",
    "background": "rolling_ball",
    "auto_contrast": "auto_contrast",
    "ac": "auto_contrast",
    "contrast": "auto_contrast",
    "gaussian": "gaussian_denoise",
    "gauss": "gaussian_denoise",
    "denoise": "gaussian_denoise",
}


def _normalize_preprocess(name: str | None) -> str | None:
    if name is None:
        return None
    key = name.strip().lower().replace("-", "_")
    if not key or key in {"none", "off"}:
        return None
    if key not in _VALID_PREPROCESS:
        raise ValueError(
            f"unknown preprocess step {name!r}. Use one of: rolling_ball, "
            "auto_contrast, gaussian_denoise, or None."
        )
    return _VALID_PREPROCESS[key]


def _decide_3d(do_3D: bool | None, layer_axes: str | None, ndim: int) -> bool:
    if do_3D is True:
        return True
    if do_3D is False:
        return False
    if layer_axes and "Z" in layer_axes and "T" not in layer_axes:
        return True
    return ndim == 3


def _layer_axes(snapshot: Any) -> str | None:
    md = snapshot.metadata or {}
    axes = md.get("axes") if isinstance(md, dict) else None
    if isinstance(axes, str):
        return axes.replace("C", "")
    return None


@tool(
    description="High-level cell analysis workflow. Resolves the target channel, "
    "optionally preprocesses it, segments cells with Cellpose-SAM, and measures "
    "per-object intensity and size on the same target channel. Pass target as a "
    "layer name, color (green/red/UV/IR), or marker (GFP/DAPI). Pass preprocess="
    "'rolling_ball' / 'auto_contrast' / 'gaussian_denoise' to apply one preprocessing "
    "step before segmentation. Returns labels layer, measurement table, object count, "
    "and QC metrics. Counterstain channels are not auto-selected — annotate them as "
    "'counterstain' first if you only have one target channel.",
    phase="2",
    worker=True,
)
def analyze_target_cells(
    target: str | None = None,
    do_3D: bool | None = None,
    diameter: float | None = None,
    preprocess: str | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    try:
        resolution = resolve_target_channel(target)
    except AmbiguousChannelError as e:
        return {
            "ok": False,
            "error": str(e),
            "candidates": list(e.candidates),
            "stage": "resolve_target",
        }

    target_layer = resolution.layer
    if resolution.source == "inference":
        warnings.append(
            f"single image layer ({target_layer}) was assumed as target — "
            "confirm by annotating the channel."
        )

    seg_input_layer = target_layer
    pre_step = _normalize_preprocess(preprocess)
    pre_record: dict[str, Any] | None = None
    if pre_step == "rolling_ball":
        pre_record = _preprocess.rolling_ball_background(layer=target_layer)
        seg_input_layer = pre_record["new_layer"]
    elif pre_step == "auto_contrast":
        pre_record = _preprocess.auto_contrast(layer=target_layer)
        seg_input_layer = pre_record["new_layer"]
    elif pre_step == "gaussian_denoise":
        pre_record = _preprocess.gaussian_denoise(layer=target_layer)
        seg_input_layer = pre_record["new_layer"]

    snapshot = call_on_main(snapshot_layer, seg_input_layer)
    axes = _layer_axes(snapshot)
    use_3d = _decide_3d(do_3D, axes, getattr(snapshot.data, "ndim", 2))

    seg_result = _segment.cellpose_sam(
        image_layer=seg_input_layer,
        do_3D=use_3d,
        diameter=diameter,
    )
    if seg_result.get("empty_mask", False):
        return {
            "ok": False,
            "stage": "segment",
            "error": "Cellpose-SAM produced zero objects on the target channel; "
            "no measurements were taken. Try a different channel, a preprocess "
            "step (rolling_ball / auto_contrast / gaussian_denoise), or set a "
            "manual diameter.",
            "target_channel": target_layer,
            "target_source": resolution.source,
            "labels_layer": seg_result["labels_layer"],
            "preprocess": pre_step,
            "warnings": warnings,
        }

    measure_result = _measure.measure_intensity(
        labels_layer=seg_result["labels_layer"],
        image_layers=[seg_input_layer],
    )
    if not measure_result.get("has_physical_units"):
        warnings.append(
            "no voxel size on the target layer — physical-unit columns were not "
            "added. Annotate or reload with scale information for area_um2 / "
            "volume_um3."
        )

    voxel = measure_result.get("voxel_scale")
    if voxel and len(voxel) == 3 and voxel[0] != voxel[1]:
        warnings.append(
            f"anisotropic voxel spacing (z={voxel[0]:.3g}, y={voxel[1]:.3g}, "
            f"x={voxel[2]:.3g}); 3D segmentation/measurement may be biased."
        )

    return {
        "ok": True,
        "target_channel": target_layer,
        "target_source": resolution.source,
        "preprocess": pre_step,
        "preprocessed_layer": pre_record["new_layer"] if pre_record else None,
        "labels_layer": seg_result["labels_layer"],
        "n_objects": int(seg_result.get("n_cells", 0)),
        "do_3D": bool(seg_result.get("do_3D", False)),
        "object_area_min": seg_result.get("object_area_min"),
        "object_area_median": seg_result.get("object_area_median"),
        "object_area_max": seg_result.get("object_area_max"),
        "table_name": measure_result["table_name"],
        "table_columns": measure_result["columns"],
        "voxel_scale": voxel,
        "has_physical_units": bool(measure_result.get("has_physical_units")),
        "warnings": warnings,
    }


def _resolve_sample_inputs(sample_name: str) -> dict[str, Any]:
    """Pick the layer name + file path the recipe should operate on for one sample."""
    from imajin.agent.state import _FILES, get_sample

    s = get_sample(sample_name)
    layer_name = s.layers[0] if s.layers else None
    file_path: str | None = None
    file_id: str | None = None
    if s.file_ids:
        file_id = s.file_ids[0]
        rec = _FILES.get(file_id)
        if rec is not None:
            file_path = rec.path
    elif s.files:
        file_path = s.files[0]
    return {
        "sample": s,
        "layer_name": layer_name,
        "file_path": file_path,
        "file_id": file_id,
    }


@tool(
    description="Apply a stored analysis recipe to one or more annotated samples. "
    "Iterates samples one by one: resolves the target channel/layer, runs the "
    "Phase-2 analyze_target_cells pipeline, attaches sample/group/file columns to "
    "the resulting measurement table, and records a per-sample AnalysisRun. A "
    "failure on one sample never aborts the batch.",
    phase="3",
    worker=True,
)
def run_recipe_on_samples(
    recipe_name: str,
    sample_names: list[str] | None = None,
) -> dict[str, Any]:
    from imajin.agent.state import (
        attach_sample_columns_to_table,
        get_recipe,
        list_samples,
        put_run,
    )

    recipe = get_recipe(recipe_name)
    if sample_names is None:
        sample_names = [s["sample_name"] for s in list_samples()]
    if not sample_names:
        return {
            "recipe": recipe_name,
            "n_samples": 0,
            "n_complete": 0,
            "n_failed": 0,
            "runs": [],
        }

    seg = recipe.segmentation or {}
    pre_steps = recipe.preprocessing or []
    pre_choice = pre_steps[0]["step"] if pre_steps else None

    runs: list[dict[str, Any]] = []
    n_complete = 0
    n_failed = 0
    for name in sample_names:
        info = _resolve_sample_inputs(name)
        s = info["sample"]
        try:
            result = analyze_target_cells(
                target=recipe.target_channel,
                do_3D=seg.get("do_3D"),
                diameter=seg.get("diameter"),
                preprocess=pre_choice,
            )
        except Exception as exc:  # noqa: BLE001
            run_id = put_run(
                sample_id=s.sample_id,
                file_id=info["file_id"] or "",
                recipe_id=recipe.recipe_id,
                status="failed",
                error=str(exc),
            )
            runs.append({"run_id": run_id, "status": "failed", "error": str(exc)})
            n_failed += 1
            continue

        if not result.get("ok"):
            run_id = put_run(
                sample_id=s.sample_id,
                file_id=info["file_id"] or "",
                recipe_id=recipe.recipe_id,
                status="failed",
                error=result.get("error", "analysis returned ok=false"),
                summary=result,
            )
            runs.append(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "error": result.get("error"),
                }
            )
            n_failed += 1
            continue

        table_name = result.get("table_name")
        if table_name:
            attach_sample_columns_to_table(
                table_name=table_name,
                sample_id=s.sample_id,
                sample_name=s.sample_name,
                group=s.group,
                file_id=info["file_id"],
                source_file=info["file_path"],
                source_layer=result.get("target_channel"),
            )

        run_id = put_run(
            sample_id=s.sample_id,
            file_id=info["file_id"] or "",
            recipe_id=recipe.recipe_id,
            status="complete",
            table_names=[table_name] if table_name else [],
            layer_names=[
                ln
                for ln in (result.get("labels_layer"), result.get("preprocessed_layer"))
                if ln
            ],
            summary={
                "n_objects": result.get("n_objects"),
                "target_channel": result.get("target_channel"),
                "warnings": result.get("warnings", []),
            },
        )
        runs.append(
            {
                "run_id": run_id,
                "status": "complete",
                "sample_name": s.sample_name,
                "table_names": [table_name] if table_name else [],
            }
        )
        n_complete += 1

    return {
        "recipe": recipe_name,
        "n_samples": len(sample_names),
        "n_complete": n_complete,
        "n_failed": n_failed,
        "runs": runs,
    }
