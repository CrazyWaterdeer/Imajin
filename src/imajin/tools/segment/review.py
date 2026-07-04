from __future__ import annotations

from typing import Any

from imajin.session import get_viewer
from imajin.agent.qt_dispatch import call_on_main
from imajin.tools.registry import tool


@tool(
    description="Open the interactive ROI review dock against an existing "
    "(image, labels) pair so the user can mark points/regions to add or "
    "remove on a MIP overlay and rebuild the ROI on the original 3D stack. "
    "Single-sample manual mode entry point for Phase 2 of the SNR/ROI work.",
    phase="2",
    llm=True,
    worker=False,
)
def review_target_roi(
    image_layer: str,
    labels_layer: str,
) -> dict[str, Any]:
    from imajin.ui.main import _show_review_panel

    viewer = get_viewer()
    if viewer is None:
        return {
            "ok": False,
            "error": "No napari viewer is available; review can only run "
            "inside the imajin GUI.",
        }

    if image_layer not in viewer.layers:
        return {"ok": False, "error": f"image_layer '{image_layer}' not found"}
    if labels_layer not in viewer.layers:
        return {"ok": False, "error": f"labels_layer '{labels_layer}' not found"}

    dock_widget = call_on_main(_show_review_panel, viewer)
    if dock_widget is None:
        return {"ok": False, "error": "review dock could not be opened"}

    call_on_main(dock_widget.request_layers, image_layer, labels_layer)

    return {
        "ok": True,
        "image_layer": image_layer,
        "labels_layer": labels_layer,
        "message": (
            "Review dock opened. Mark add/remove points and regions, then "
            "click Rebuild ROI; click Commit to write changes back to the "
            "labels layer."
        ),
    }
