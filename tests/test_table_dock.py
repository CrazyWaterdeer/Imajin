from __future__ import annotations

import pandas as pd

from imajin.agent import state


def test_table_dock_picks_up_new_tables(qtbot, viewer) -> None:
    from imajin.ui.table_dock import TableDock

    state.reset_tables()
    dock = TableDock(viewer=viewer)
    qtbot.addWidget(dock)
    assert dock.table_picker.count() == 0

    state.put_table("demo", pd.DataFrame({"label": [1, 2], "area": [10, 20]}))
    qtbot.waitUntil(lambda: dock.table_picker.count() == 1, timeout=1000)
    assert dock.table_picker.itemText(0) == "demo"
    assert dock._model.rowCount() == 2
    assert dock._model.columnCount() == 2

    state.reset_tables()


def test_dataframe_model_renders_floats(qtbot, viewer) -> None:
    from imajin.ui.table_dock import TableDock, _DataFrameModel

    model = _DataFrameModel(pd.DataFrame({"x": [1.23456789, 2.0]}))
    from qtpy.QtCore import Qt

    idx = model.index(0, 0)
    text = model.data(idx, Qt.ItemDataRole.DisplayRole)
    assert text == "1.235"
