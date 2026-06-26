from __future__ import annotations


def test_manual_dock_populates_tools(qtbot, viewer) -> None:
    from imajin.ui.manual_dock import ManualDock

    dock = ManualDock(viewer=viewer)
    qtbot.addWidget(dock)

    assert dock.tool_picker.count() >= 6, "expected at least 6 registered tools"
    assert dock._current_widget is not None, "form should be built for first tool"


def test_manual_dock_form_rebuilds_on_change(qtbot, viewer) -> None:
    from imajin.ui.manual_dock import ManualDock

    dock = ManualDock(viewer=viewer)
    qtbot.addWidget(dock)

    first = dock._current_widget
    if dock.tool_picker.count() >= 2:
        dock.tool_picker.setCurrentIndex(1)
        assert dock._current_widget is not first


def test_layer_param_names_offers_boundary_mask_dropdown() -> None:
    # The dock builds a layer dropdown for any param _layer_param_names returns.
    # boundary_mask must be offered (it is a layer), but scalar params must not be.
    import imajin.tools  # noqa: F401 - ensure @tool registration
    from imajin.tools.registry import get_tool
    from imajin.ui.manual_dock import _layer_param_names

    entry = get_tool("segment_target_objects")
    names = _layer_param_names(entry.func)
    assert "image_layer" in names
    assert "boundary_mask" in names
    assert "min_snr" not in names
