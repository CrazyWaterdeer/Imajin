from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imajin.agent.execution import raise_if_cancelled, report_progress
from imajin.agent.qt_dispatch import call_on_main
from imajin.session import (
    attach_sample_columns_to_table,
    bulk_state_update,
    canonical_channel_color,
    get_file,
    get_recipe,
    get_sample,
    get_table,
    get_viewer,
    layer_channel_metadata,
    list_channel_annotations,
    list_samples,
    put_run,
    put_table,
)
from imajin.analysis.workflow import (
    build_sample_summary,
    first_analysis_preprocess,
    normalize_domain_spec,
    normalize_match_text,
    projection_request,
    release_worker_memory,
    rss_mb,
)
from imajin.tools.layers import remove_layers_by_name, viewer_layer_names
from imajin.workers.qt_worker import CancelledError


def cleanup_new_layers(base_layer_names: set[str]) -> list[str]:
    return cleanup_sample_layers(base_layer_names, [])


def cleanup_sample_layers(
    base_layer_names: set[str],
    managed_layer_names: list[str],
) -> list[str]:
    current = viewer_layer_names()
    current_set = set(current)
    created = [name for name in current if name not in base_layer_names]
    managed = [name for name in managed_layer_names if name in current_set]
    to_remove = list(dict.fromkeys([*created, *managed]))
    return call_on_main(remove_layers_by_name, to_remove)


def _statistics_partitions(df: Any) -> list[tuple[str, Any]]:
    if "tier" not in df.columns:
        return [("all", df)]
    partitions: list[tuple[str, Any]] = []
    for tier, part in df.groupby("tier", dropna=False, sort=False):
        tier_name = str(tier) if tier is not None else "unknown"
        partitions.append((tier_name, part.reset_index(drop=True)))
    return partitions or [("all", df)]


def _combined_primary_table(primary_tables: list[str]) -> Any | None:
    import pandas as pd


    frames = []
    for name in primary_tables:
        try:
            frame = get_table(name)
        except KeyError:
            continue
        if frame is not None and not frame.empty:
            frames.append(frame.copy())
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


def _batch_stats_input_name(bundle_name: str, tier: str, value_col: str) -> str:
    from imajin.results import slugify_result_name

    return "__".join(
        [
            "stats_input",
            slugify_result_name(bundle_name),
            slugify_result_name(tier),
            slugify_result_name(value_col),
        ]
    )


def _finite_value_rows(part: Any, value_col: str) -> Any:
    import pandas as pd

    valid = part.copy()
    valid[value_col] = pd.to_numeric(valid[value_col], errors="coerce")
    return valid[valid[value_col].notna()].reset_index(drop=True)


def _compare_batch_stats(
    stats_input_name: str,
    value_col: str,
    *,
    valid: Any,
    tier: str,
) -> dict[str, Any] | None:
    from imajin.tools import stats as _stats

    if valid["group"].nunique(dropna=True) < 2:
        return None
    try:
        return _stats.compare_groups(
            stats_input_name,
            value_col,
            level="auto",
            save_csv=True,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "value_col": value_col,
            "tier": tier,
        }


def _batch_statistics_output(
    *,
    tier: str,
    value_col: str,
    stats_input_name: str,
    desc: dict[str, Any],
    compare: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "tier": tier,
        "value_col": value_col,
        "input_table": stats_input_name,
        "object_stats_table": desc.get("object_stats_table"),
        "sample_stats_table": desc.get("sample_stats_table"),
        "object_stats_csv": desc.get("object_stats_csv"),
        "sample_stats_csv": desc.get("sample_stats_csv"),
        "comparison_table": (
            compare.get("result_table")
            if isinstance(compare, dict)
            else None
        ),
        "comparison_csv": (
            compare.get("csv_path")
            if isinstance(compare, dict)
            else None
        ),
        "comparison_error": (
            compare.get("error")
            if isinstance(compare, dict)
            else None
        ),
    }


def loaded_layer_metadata_text(layer: Any) -> str:
    md = getattr(layer, "metadata", {}) or {}
    parts = [getattr(layer, "name", "")]
    if isinstance(md, dict):
        for key in ("name", "channel_name", "marker", "color"):
            if key in md and md[key] is not None:
                parts.append(str(md[key]))
    try:

        channel_info = layer_channel_metadata(layer)
    except Exception:
        channel_info = {}
    if isinstance(channel_info, dict):
        for key in (
            "name",
            "channel_name",
            "marker",
            "color",
            "display_color_name",
            "dye_name",
            "excitation_wavelength_nm",
            "emission_wavelength_nm",
        ):
            if key in channel_info and channel_info[key] is not None:
                parts.append(str(channel_info[key]))
    return " ".join(parts)


def resolve_target_within_loaded_layers(
    target: str | None,
    loaded_layers: list[str],
) -> str | None:
    """Resolve a recipe target only against layers loaded for the current sample."""
    if not loaded_layers:
        return target
    current = list(dict.fromkeys(str(name) for name in loaded_layers))
    if target is None:
        return current[0] if len(current) == 1 else None
    if target in current:
        return target


    query = normalize_match_text(target)
    target_color = canonical_channel_color(target)
    viewer = get_viewer()
    matches: list[str] = []
    for layer_name in current:
        try:
            layer = viewer.layers[layer_name]
        except Exception:
            continue
        text = normalize_match_text(loaded_layer_metadata_text(layer))
        layer_color = canonical_channel_color(text)
        if query and (query == normalize_match_text(layer_name) or query in text):
            matches.append(layer_name)
        elif target_color is not None and layer_color == target_color:
            matches.append(layer_name)

    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    if len(current) == 1:
        return current[0]
    return target


def load_file_for_sample_if_needed(
    info: dict[str, Any],
    *,
    auto_load_files: bool,
) -> dict[str, Any] | None:
    if not auto_load_files or not info.get("file_path"):
        return None
    sample_layers = list(getattr(info["sample"], "layers", []) or [])
    existing = set(call_on_main(viewer_layer_names))
    if sample_layers and any(layer_name in existing for layer_name in sample_layers):
        return None
    from imajin.tools import files as _files

    return call_on_main(_files.load_file, str(info["file_path"]))


def project_layer_for_recipe(
    layer_name: str | None,
    *,
    projection: str | None,
    axis: Any,
) -> dict[str, Any] | None:
    if projection is None:
        return None
    if layer_name is None:
        raise ValueError("projection requires a target layer")
    from imajin.tools import view as _view

    if projection == "mean":
        return _view.average_projection(layer_name, axis=axis)
    if projection == "max":
        return _view.max_projection(layer_name, axis=axis)
    raise ValueError("projection must be mean or max")


def resolve_sample_inputs(sample_name: str) -> dict[str, Any]:
    """Pick the layer name + file path the recipe should operate on for one sample."""
    sample = get_sample(sample_name)
    layer_name = sample.layers[0] if sample.layers else None
    file_path: str | None = None
    file_id: str | None = None
    if sample.file_ids:
        file_id = sample.file_ids[0]
        try:
            file_path = get_file(file_id).path
        except KeyError:
            pass
    elif sample.files:
        file_path = sample.files[0]
    return {
        "sample": sample,
        "layer_name": layer_name,
        "file_path": file_path,
        "file_id": file_id,
    }


@dataclass
class BatchRecipeRunner:
    recipe_name: str
    sample_names: list[str] | None = None
    execution_mode: str = "serial_cleanup"
    auto_load_files: bool = True
    keep_layers: bool = False
    keep_failed_layers: bool = False

    def run(self) -> dict[str, Any]:
        with bulk_state_update("batch_recipe_run"):
            return self._run()

    def _run(self) -> dict[str, Any]:
        self.recipe = get_recipe(self.recipe_name)
        self.names = self._sample_names()
        if not self.names:
            return self._empty_result()

        self.mode = self._normalize_execution_mode()
        self.cleanup_enabled = (
            self.mode in {"serial_cleanup", "cleanup"} and not self.keep_layers
        )
        self.seg = self.recipe.segmentation or {}
        self.measurement = self.recipe.measurement or {}
        self.pre_steps = self.recipe.preprocessing or []
        self.pre_choice = first_analysis_preprocess(self.pre_steps)
        self.projection, self.projection_axis = projection_request(
            self.measurement,
            self.pre_steps,
        )
        self.domain_strategy, self.domain_options = normalize_domain_spec(
            self.recipe.domain
        )
        self.runs: list[dict[str, Any]] = []
        self.sample_summaries: list[dict[str, Any]] = []
        self.statistics_outputs: list[dict[str, Any]] = []
        self.n_complete = 0
        self.n_failed = 0
        self.metadata_validation = self._validate_metadata_preflight()
        if self.metadata_validation.get("status") == "fail":
            return self._metadata_validation_failure()

        self.parent_bundle = self._create_parent_bundle()
        from imajin.result_bundles import with_active_bundle

        with with_active_bundle(self.parent_bundle):
            cancelled = False
            try:
                for index, name in enumerate(self.names):
                    raise_if_cancelled()
                    self._process_sample(index, name)
            except CancelledError:
                cancelled = True
                self._append_cancelled_sample_summaries()
                raise
            finally:
                self._finalize_bundle(cancelled=cancelled)

        return {
            "recipe": self.recipe_name,
            "n_samples": len(self.names),
            "n_complete": self.n_complete,
            "n_failed": self.n_failed,
            "execution_mode": self.mode,
            "cleanup_enabled": self.cleanup_enabled,
            "runs": self.runs,
            "bundle_path": str(self.parent_bundle),
            "metadata_validation": self.metadata_validation,
            "statistics_outputs": list(self.statistics_outputs),
        }

    def _sample_names(self) -> list[str]:
        if self.sample_names is None:
            return [s["sample_name"] for s in list_samples()]
        return list(self.sample_names)

    def _empty_result(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe_name,
            "n_samples": 0,
            "n_complete": 0,
            "n_failed": 0,
            "runs": [],
            "bundle_path": None,
            "execution_mode": self.execution_mode,
            "cleanup_enabled": False,
        }

    def _normalize_execution_mode(self) -> str:
        mode = self.execution_mode.strip().lower().replace("-", "_")
        if mode == "parallel_headless":
            raise ValueError(
                "parallel_headless is planned for headless/CLI workers but is not "
                "implemented in the napari GUI runner yet. Use serial_cleanup."
            )
        if mode not in {"serial_cleanup", "cleanup", "serial"}:
            raise ValueError(
                "execution_mode must be 'serial_cleanup', 'serial', or 'parallel_headless'"
            )
        return mode

    def _validate_metadata_preflight(self) -> dict[str, Any]:
        from imajin.analysis.metadata_validation import validate_acquisition_metadata

        records: list[dict[str, Any]] = []
        for name in self.names:
            info = resolve_sample_inputs(name)
            file_path = info.get("file_path")
            if not file_path:
                continue
            metadata_summary: dict[str, Any] = {}
            file_id = info.get("file_id")
            if file_id:
                try:
                    metadata_summary = dict(get_file(file_id).metadata_summary or {})
                except KeyError:
                    metadata_summary = {}
            records.append(
                {
                    "path": file_path,
                    "file_id": file_id,
                    "sample_name": name,
                    "target_channel": self.recipe.target_channel,
                    "metadata_summary": metadata_summary,
                }
            )
        if not records:
            return {
                "ok": True,
                "status": "warning",
                "analysis_kind": "intensity",
                "settings_checked": [],
                "warnings": [
                    "metadata preflight skipped because no source file paths are "
                    "registered for the selected samples"
                ],
                "mismatches": [],
                "missing_settings": [],
                "metadata_errors": [],
                "channels": [],
                "metadata_only": True,
            }
        return validate_acquisition_metadata(
            records,
            target_channel=self.recipe.target_channel,
            analysis_kind="auto",
            measurement=self.measurement,
            strict_missing=False,
        )

    def _metadata_validation_failure(self) -> dict[str, Any]:
        error = self.metadata_validation.get("error") or "metadata validation failed"
        return {
            "recipe": self.recipe_name,
            "n_samples": len(self.names),
            "n_complete": 0,
            "n_failed": len(self.names),
            "execution_mode": self.mode,
            "cleanup_enabled": False,
            "runs": [],
            "bundle_path": None,
            "metadata_validation": self.metadata_validation,
            "error": (
                f"{error}; analysis was not run. Check that all target-channel "
                "acquisition settings match before comparing intensity."
            ),
        }

    def _create_parent_bundle(self) -> Any:
        from imajin.anchor import resolve_anchor_folder
        from imajin.results import create_result_bundle

        sample_paths: list[str] = []
        for name in self.names:
            info = resolve_sample_inputs(name)
            if info.get("file_path"):
                sample_paths.append(str(info["file_path"]))
        anchor = resolve_anchor_folder(sample_paths)

        bundle = create_result_bundle(
            name=self.recipe.name,
            kind="batch",
            tier="two_tier" if self.domain_strategy is not None else "single_tier",
            metadata={
                "recipe": {
                    "name": self.recipe.name,
                    "target_channel": self.recipe.target_channel,
                    "preprocessing": list(self.recipe.preprocessing or []),
                    "segmentation": dict(self.recipe.segmentation or {}),
                    "measurement": dict(self.recipe.measurement or {}),
                    "domain": dict(self.recipe.domain) if self.recipe.domain else None,
                    "cell_diameter_um": self.recipe.cell_diameter_um,
                },
            },
            root=anchor,
        )
        from imajin.result_bundles import promote_to_process_bundle
        promote_to_process_bundle(bundle)
        return bundle

    def _process_sample(self, index: int, name: str) -> None:
        total = len(self.names)
        info = resolve_sample_inputs(name)
        sample = info["sample"]
        current_file = info["file_path"] or info["layer_name"] or sample.sample_name
        base_layer_names = set(call_on_main(viewer_layer_names))
        mem_before = rss_mb()
        failed_sample = False
        managed_layer_names: list[str] = []
        runs_len_before = len(self.runs)
        report_progress(
            progress=index / total,
            stage="sample",
            current_file=current_file,
            file_index=index + 1,
            total_files=total,
            completed=self.n_complete,
            failed=self.n_failed,
            skipped=0,
            show_in_chat=True,
            message=f"Processing {sample.sample_name} ({index + 1}/{total}).",
            detail={
                "rss_mb": mem_before,
                "layer_count": len(base_layer_names),
                "execution_mode": self.mode,
            },
        )
        try:
            result, target = self._analyze_sample(info, managed_layer_names)
        except CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            failed_sample = True
            self._record_exception(sample, info, exc)
            self._report_sample_failed(index, current_file, sample.sample_name)
        else:
            if not result.get("ok"):
                failed_sample = True
                self._record_not_ok(sample, info, result)
                self._report_sample_failed(index, current_file, sample.sample_name)
            else:
                self._record_success(sample, info, result, target)
                self._report_sample_complete(index, current_file, sample.sample_name)
        finally:
            self._cleanup_after_sample(
                index=index,
                current_file=current_file,
                sample_name=sample.sample_name,
                failed_sample=failed_sample,
                base_layer_names=base_layer_names,
                managed_layer_names=managed_layer_names,
                runs_len_before=runs_len_before,
                mem_before=mem_before,
            )

    def _analyze_sample(
        self,
        info: dict[str, Any],
        managed_layer_names: list[str],
    ) -> tuple[dict[str, Any], str | None]:
        from imajin.results import slugify_result_name
        from imajin.tools import workflows as _workflows
        from imajin.result_bundles import with_active_sample_slug

        load_result = load_file_for_sample_if_needed(
            info,
            auto_load_files=self.auto_load_files,
        )
        raise_if_cancelled()
        target = self.recipe.target_channel or info["layer_name"]
        loaded_layers = list((load_result or {}).get("layer_names") or [])
        managed_layer_names.extend(str(name) for name in loaded_layers)
        target = call_on_main(
            resolve_target_within_loaded_layers,
            target,
            loaded_layers,
        )
        if target is None and len(loaded_layers) == 1:
            target = loaded_layers[0]
        if self.projection is not None and target is None:
            raise ValueError(
                "measurement.projection requires a resolved target layer. "
                "Set recipe.target_channel to a layer name/color that resolves "
                "within each loaded sample."
            )
        projection_record = project_layer_for_recipe(
            target,
            projection=self.projection,
            axis=self.projection_axis,
        )
        raise_if_cancelled()
        analysis_target = projection_record["new_layer"] if projection_record else target
        if projection_record:
            managed_layer_names.append(str(projection_record["new_layer"]))
        sample = info["sample"]
        with with_active_sample_slug(slugify_result_name(sample.sample_name)):
            result = _workflows.analyze_target_cells(
                target=analysis_target,
                do_3D=False if projection_record else self.seg.get("do_3D"),
                diameter=self.seg.get("diameter"),
                preprocess=self.pre_choice,
                segmentation_method=self.seg.get("tool")
                or self.seg.get("method", "target_objects"),
                segmentation_options=self.seg,
                domain_strategy=self.domain_strategy,
                domain_options=self.domain_options,
                review_mode=getattr(self.recipe, "review_mode", "auto"),
            )
        return result, target

    def _record_exception(
        self,
        sample: Any,
        info: dict[str, Any],
        exc: Exception,
    ) -> None:
        run_id = put_run(
            sample_id=sample.sample_id,
            file_id=info["file_id"] or "",
            recipe_id=self.recipe.recipe_id,
            status="failed",
            error=str(exc),
        )
        self.runs.append({"run_id": run_id, "status": "failed", "error": str(exc)})
        self.sample_summaries.append(
            build_sample_summary(
                sample_name=sample.sample_name,
                status="failed",
                error=str(exc),
                group=sample.group,
                file_id=info["file_id"],
                source_file=info["file_path"],
                source_layer=info["layer_name"],
            )
        )
        self.n_failed += 1

    def _record_not_ok(
        self,
        sample: Any,
        info: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        error = result.get("error", "analysis returned ok=false")
        run_id = put_run(
            sample_id=sample.sample_id,
            file_id=info["file_id"] or "",
            recipe_id=self.recipe.recipe_id,
            status="failed",
            error=error,
            summary=result,
        )
        self.runs.append(
            {
                "run_id": run_id,
                "status": "failed",
                "error": result.get("error"),
            }
        )
        self.sample_summaries.append(
            build_sample_summary(
                sample_name=sample.sample_name,
                status="failed",
                error=error,
                group=sample.group,
                file_id=info["file_id"],
                source_file=info["file_path"],
                source_layer=info["layer_name"],
            )
        )
        self.n_failed += 1

    def _record_success(
        self,
        sample: Any,
        info: dict[str, Any],
        result: dict[str, Any],
        target: str | None,
    ) -> None:
        attached_tables: list[str] = []
        for tname_key in ("table_name", "tier_table_name"):
            tname = result.get(tname_key)
            if not tname or tname in attached_tables:
                continue
            attach_sample_columns_to_table(
                table_name=tname,
                sample_id=sample.sample_id,
                sample_name=sample.sample_name,
                group=sample.group,
                file_id=info["file_id"],
                source_file=info["file_path"],
                source_layer=result.get("target_channel"),
            )
            attached_tables.append(tname)

        run_id = put_run(
            sample_id=sample.sample_id,
            file_id=info["file_id"] or "",
            recipe_id=self.recipe.recipe_id,
            status="complete",
            table_names=list(attached_tables),
            layer_names=[
                layer_name
                for layer_name in (
                    result.get("labels_layer"),
                    result.get("preprocessed_layer"),
                )
                if layer_name
            ],
            summary={
                "n_objects": result.get("n_objects"),
                "object_unit": result.get("object_unit"),
                "segmentation_method": result.get("segmentation_method"),
                "analysis_dim": result.get("analysis_dim"),
                "target_channel": result.get("target_channel"),
                "source_target_channel": target,
                "projection": self.projection,
                "projection_axis": self.projection_axis,
                "warnings": result.get("warnings", []),
                "qc_png_skipped_reason": result.get("qc_png_skipped_reason"),
            },
        )
        self.runs.append(
            {
                "run_id": run_id,
                "status": "complete",
                "sample_name": sample.sample_name,
                "table_names": list(attached_tables),
            }
        )
        self.sample_summaries.append(
            build_sample_summary(
                sample_name=sample.sample_name,
                status="complete",
                n_cells=int(result.get("n_cells", result.get("n_objects", 0)) or 0),
                n_domain_components=result.get("n_domain_components"),
                domain_label_count=result.get("domain_label_count"),
                domain_area_um2=result.get("domain_area_um2"),
                domain_volume_um3=result.get("domain_volume_um3"),
                domain_voxels=result.get("domain_voxels"),
                qc_warnings=list(result.get("warnings") or []),
                outputs=dict(result.get("result_files") or {}),
                group=sample.group,
                file_id=info["file_id"],
                source_file=info["file_path"],
                source_layer=result.get("target_channel"),
            )
        )
        self.n_complete += 1

    def _report_sample_failed(
        self,
        index: int,
        current_file: str,
        sample_name: str,
    ) -> None:
        total = len(self.names)
        report_progress(
            progress=(index + 1) / total,
            stage="failed",
            current_file=current_file,
            file_index=index + 1,
            total_files=total,
            completed=self.n_complete,
            failed=self.n_failed,
            skipped=0,
            show_in_chat=True,
            message=f"Failed {sample_name}; continuing.",
        )

    def _report_sample_complete(
        self,
        index: int,
        current_file: str,
        sample_name: str,
    ) -> None:
        total = len(self.names)
        report_progress(
            progress=(index + 1) / total,
            stage="complete_sample",
            current_file=current_file,
            file_index=index + 1,
            total_files=total,
            completed=self.n_complete,
            failed=self.n_failed,
            skipped=0,
            show_in_chat=True,
            message=f"Completed {sample_name} ({index + 1}/{total}).",
        )

    def _cleanup_after_sample(
        self,
        *,
        index: int,
        current_file: str,
        sample_name: str,
        failed_sample: bool,
        base_layer_names: set[str],
        managed_layer_names: list[str],
        runs_len_before: int,
        mem_before: float,
    ) -> None:
        removed_layers: list[str] = []
        if self.cleanup_enabled and not (failed_sample and self.keep_failed_layers):
            try:
                removed_layers = cleanup_sample_layers(
                    base_layer_names,
                    managed_layer_names,
                )
            except Exception:
                removed_layers = []
        release_worker_memory()
        mem_after = rss_mb()
        if len(self.runs) > runs_len_before:
            self.runs[-1]["cleanup_removed_layers"] = removed_layers
            self.runs[-1]["rss_mb_before"] = mem_before
            self.runs[-1]["rss_mb_after"] = mem_after
        total = len(self.names)
        report_progress(
            progress=(index + 1) / total,
            stage="cleanup",
            current_file=current_file,
            file_index=index + 1,
            total_files=total,
            completed=self.n_complete,
            failed=self.n_failed,
            skipped=0,
            show_in_chat=True,
            message=f"Cleaned up {len(removed_layers)} layers for {sample_name}.",
            detail={
                "rss_mb": mem_after,
                "cleanup_removed_layers": len(removed_layers),
                "layer_count": len(call_on_main(viewer_layer_names)),
            },
        )

    def _append_cancelled_sample_summaries(self) -> None:
        processed_names = {
            summary["sample_name"] for summary in self.sample_summaries
        }
        for skip_name in self.names:
            if skip_name in processed_names:
                continue
            try:
                sample = get_sample(skip_name)
                group = sample.group
                file_id = sample.file_ids[0] if sample.file_ids else None
            except Exception:  # noqa: BLE001
                group = None
                file_id = None
            self.sample_summaries.append(
                build_sample_summary(
                    sample_name=skip_name,
                    status="skipped",
                    group=group,
                    file_id=file_id,
                )
            )

    def _write_batch_statistics(self, primary_tables: list[str]) -> None:
        if not primary_tables:
            return

        from imajin.tools import stats as _stats

        combined = _combined_primary_table(primary_tables)
        if combined is None:
            return
        if not {"sample_name", "group"}.issubset(combined.columns):
            return

        outputs: list[dict[str, Any]] = []
        for tier, part in _statistics_partitions(combined):
            value_cols = _stats.default_statistics_value_columns(part)
            if not value_cols:
                continue
            for value_col in value_cols:
                valid = _finite_value_rows(part, value_col)
                if valid.empty:
                    continue
                stats_input_name = _batch_stats_input_name(
                    self.parent_bundle.name,
                    tier,
                    value_col,
                )
                stats_input_name = put_table(
                    stats_input_name,
                    valid,
                    spec={
                        "tool": "batch_auto_statistics_input",
                        "source_tables": list(primary_tables),
                        "tier": tier,
                        "value_col": value_col,
                    },
                )
                desc = _stats.describe_table(
                    stats_input_name,
                    value_col,
                    save_csv=True,
                )
                compare = _compare_batch_stats(
                    stats_input_name,
                    value_col,
                    valid=valid,
                    tier=tier,
                )
                outputs.append(
                    _batch_statistics_output(
                        tier=tier,
                        value_col=value_col,
                        stats_input_name=stats_input_name,
                        desc=desc,
                        compare=compare,
                    )
                )
        self.statistics_outputs = outputs

    def _finalize_bundle(self, *, cancelled: bool) -> None:
        from imajin.result_bundles import finalize_bundle_metadata, write_combined_csv

        primary_tables: list[str] = []
        for run in self.runs:
            names = run.get("table_names") or []
            if names:
                primary_tables.append(names[-1])
        if cancelled:
            try:
                write_combined_csv(self.parent_bundle, primary_tables)
                self._write_batch_statistics(primary_tables)
                finalize_bundle_metadata(
                    self.parent_bundle,
                    samples=self.sample_summaries,
                    status="cancelled",
                    extra={"run_context_extras": self._run_context_extras()},
                )
            except Exception:  # noqa: BLE001
                pass
            return

        write_combined_csv(self.parent_bundle, primary_tables)
        self._write_batch_statistics(primary_tables)
        finalize_bundle_metadata(
            self.parent_bundle,
            samples=self.sample_summaries,
            status="complete",
            extra={"run_context_extras": self._run_context_extras()},
        )

    def _run_context_extras(self) -> dict[str, Any]:
        from pathlib import Path

        folder_set: set[str] = set()
        for name in self.names:
            try:
                info = resolve_sample_inputs(name)
            except Exception:  # noqa: BLE001
                continue
            file_path = info.get("file_path")
            if file_path:
                folder_set.add(str(Path(file_path).expanduser().resolve().parent))

        channel_roles: dict[str, str] = {}
        for entry in list_channel_annotations():
            layer_name = entry.get("layer_name")
            role = entry.get("role")
            if layer_name and role:
                channel_roles[str(layer_name)] = str(role)

        return {
            "folder_set": sorted(folder_set, key=str.lower),
            "channel_roles": channel_roles,
            "scope_filters": [],
            "metadata_validation": getattr(self, "metadata_validation", {}),
            "statistics_outputs": list(getattr(self, "statistics_outputs", [])),
        }
