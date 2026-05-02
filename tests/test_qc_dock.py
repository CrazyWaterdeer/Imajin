from __future__ import annotations

import numpy as np
import pandas as pd

from imajin.agent import state


def test_qc_dock_computes_label_qc(qtbot, viewer) -> None:
    from imajin.ui.qc_dock import QCDock

    labels = np.zeros((16, 16), dtype=np.int32)
    labels[4:8, 5:9] = 1
    viewer.add_labels(labels, name="masks", scale=(0.5, 0.5))

    dock = QCDock(viewer=viewer)
    qtbot.addWidget(dock)
    idx = dock._find_source_index("labels", "masks")
    dock.source_picker.setCurrentIndex(idx)

    dock._compute_qc()

    assert state.get_qc_record("masks").status == "pass"
    assert dock.status_label.text() == "pass"
    assert dock.metrics_table.rowCount() > 0


def test_qc_dock_marks_status_with_notes(qtbot, viewer) -> None:
    from imajin.ui.qc_dock import QCDock

    state.set_table(
        "measurements",
        pd.DataFrame({"label": [1], "area": [10], "mean_intensity": [5.0]}),
        spec={"tool": "measure_intensity"},
    )
    state.put_qc_record("measurements", status="warning", warnings=["check"])

    dock = QCDock(viewer=viewer)
    qtbot.addWidget(dock)
    idx = dock._find_source_index("table", "measurements")
    dock.source_picker.setCurrentIndex(idx)
    dock.notes_edit.setPlainText("looks acceptable")

    dock._mark("pass")

    record = state.get_qc_record("measurements")
    assert record.status == "pass"
    assert record.reviewed_by_user is True
    assert record.notes == "looks acceptable"
