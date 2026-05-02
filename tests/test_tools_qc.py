from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imajin.agent import state
from imajin.tools import call_tool, qc


@pytest.fixture(autouse=True)
def _clean_tables_and_qc():
    state.reset_tables()
    state.reset_qc_records()
    yield
    state.reset_tables()
    state.reset_qc_records()


def test_compute_segmentation_qc_passes_clean_labels(viewer) -> None:
    labels = np.zeros((24, 24), dtype=np.int32)
    labels[4:10, 4:10] = 1
    labels[14:20, 14:20] = 2
    image = np.zeros_like(labels, dtype=np.uint16)
    image[labels > 0] = 1000
    viewer.add_labels(labels, name="masks", scale=(0.5, 0.5))
    viewer.add_image(image, name="green", scale=(0.5, 0.5))

    result = qc.compute_segmentation_qc("masks", image_layer="green")

    assert result["status"] == "pass"
    assert result["metrics"]["n_objects"] == 2
    assert result["metrics"]["saturation_fraction"] == 0.0
    assert state.get_qc_record("masks").status == "pass"


def test_compute_segmentation_qc_flags_empty_labels(viewer) -> None:
    viewer.add_labels(np.zeros((8, 8), dtype=np.int32), name="empty")

    result = qc.compute_segmentation_qc("empty")

    assert result["status"] == "fail"
    assert result["metrics"]["n_objects"] == 0
    assert result["warnings"]


def test_compute_segmentation_qc_flags_border_touching_labels(viewer) -> None:
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[0:5, 0:5] = 1
    labels[8:13, 8:13] = 2
    viewer.add_labels(labels, name="bordered", scale=(0.5, 0.5))

    result = qc.compute_segmentation_qc("bordered")

    assert result["status"] == "warning"
    assert result["metrics"]["border_touching_labels"] == [1]
    assert any("border" in warning for warning in result["warnings"])


def test_create_label_outline_adds_image_layer(viewer) -> None:
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[3:9, 4:8] = 1
    viewer.add_labels(labels, name="masks")

    result = qc.create_label_outline("masks")

    assert result["new_layer"] == "masks_outline"
    assert result["n_outline_pixels"] > 0
    assert "masks_outline" in viewer.layers
    assert viewer.layers["masks_outline"].colormap.name == "red"


def test_compute_measurement_qc_flags_all_zero_intensity() -> None:
    state.set_table(
        "measurements",
        pd.DataFrame(
            {
                "label": [1, 2],
                "area": [25, 30],
                "mean_intensity_green": [0.0, 0.0],
            }
        ),
        spec={"tool": "measure_intensity", "labels_layer": "masks"},
    )

    result = qc.compute_measurement_qc("measurements")

    assert result["status"] == "warning"
    assert result["metrics"]["all_zero_intensity_columns"] == ["mean_intensity_green"]
    assert state.get_qc_record("measurements").status == "warning"


def test_compute_timecourse_qc_flags_flat_traces() -> None:
    state.set_table(
        "timecourse",
        pd.DataFrame(
            {
                "time_index": [0, 1, 2, 0, 1, 2],
                "label": [1, 1, 1, 2, 2, 2],
                "mean_intensity": [5.0, 5.0, 5.0, 10.0, 10.0, 10.0],
            }
        ),
        spec={"tool": "measure_intensity_over_time"},
    )

    result = qc.compute_timecourse_qc("timecourse")

    assert result["status"] == "warning"
    assert result["metrics"]["n_rois"] == 2
    assert result["metrics"]["flat_trace_fraction"] == pytest.approx(1.0)


def test_jump_to_object_selects_label_and_centers_camera(viewer) -> None:
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[4:8, 6:10] = 3
    viewer.add_labels(labels, name="masks", scale=(0.5, 0.25))
    state.set_table(
        "measurements",
        pd.DataFrame(
            {
                "label": [3],
                "centroid-0": [5.5],
                "centroid-1": [7.5],
                "area": [16],
                "mean_intensity_green": [42.0],
            }
        ),
        spec={"tool": "measure_intensity", "labels_layer": "masks"},
    )

    result = qc.jump_to_object("measurements", 3)

    assert result["selected"] is True
    assert viewer.layers["masks"].selected_label == 3
    assert viewer.layers["masks"].show_selected_label is True
    assert viewer.camera.center == pytest.approx((2.75, 1.875))


def test_mark_qc_status_preserves_metrics_and_notes() -> None:
    state.put_qc_record(
        "measurements",
        status="warning",
        warnings=["check segmentation"],
        metrics={"n_rows": 2},
    )

    result = qc.mark_qc_status("measurements", "pass", notes="reviewed manually")
    record = state.get_qc_record("measurements")

    assert result["reviewed_by_user"] is True
    assert record.status == "pass"
    assert record.metrics == {"n_rows": 2}
    assert record.warnings == ["check segmentation"]
    assert record.notes == "reviewed manually"


def test_qc_tools_are_registered() -> None:
    result = call_tool(
        "mark_qc_status",
        source="manual_review",
        status="not_checked",
        notes="pending",
    )

    assert result["source"] == "manual_review"
    assert state.get_qc_record("manual_review").notes == "pending"
