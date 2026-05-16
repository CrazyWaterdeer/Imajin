"""Napari dock for reviewing and correcting an automatic target ROI.

Phase 2 of the SNR/ROI initiative. The user opens this dock against a
(target image, current labels) pair; the dock shows their MIPs and a
pair of Points/Shapes layers for marking pixels to add or remove. The
"Rebuild" button re-segments on the original 3D stack using
:func:`imajin.analysis.interactive_roi.correct_roi_from_markings`, and
"Commit" promotes the corrected labels back to the source labels layer.

This module only handles UI. The algorithm itself lives in
``analysis/interactive_roi.py`` and is tested headlessly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from imajin.ui.theme import NoScrollComboBox, NoScrollDoubleSpinBox, apply_dock_theme


_MARKING_MODES = (
    ("add_point", "Add point (recover signal)"),
    ("remove_point", "Remove point (drop component)"),
    ("add_region", "Add region (recover dim area)"),
    ("remove_region", "Remove region (mask out)"),
)

_ADD_COLOR = "#3DD68C"
_REMOVE_COLOR = "#DA4E42"


def _mip(array: np.ndarray) -> np.ndarray:
    """Max projection along the leading axis for 3D, identity for 2D."""
    arr = np.asarray(array)
    if arr.ndim == 3:
        return np.nanmax(arr, axis=0)
    return arr


def _layer_suffix_names(stem: str) -> dict[str, str]:
    return {
        "mip_image": f"{stem} · review MIP",
        "mip_labels": f"{stem} · review labels",
        "add_points": f"{stem} · add points",
        "remove_points": f"{stem} · remove points",
        "add_shapes": f"{stem} · add regions",
        "remove_shapes": f"{stem} · remove regions",
    }


class ReviewDock(QWidget):
    """Single-sample interactive ROI review widget."""

    review_committed = Signal(dict)
    review_skipped = Signal(dict)

    def __init__(self, viewer: Any) -> None:
        super().__init__()
        apply_dock_theme(self)
        self.viewer = viewer

        self._target_layer_name: str | None = None
        self._labels_layer_name: str | None = None
        self._scratch_names: dict[str, str] = {}
        self._original_corrected: np.ndarray | None = None
        self._original_labels: np.ndarray | None = None
        self._current_labels: np.ndarray | None = None
        self._noise_sigma: float = 0.0
        self._base_threshold: float = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        target_box = QGroupBox("Target")
        target_layout = QFormLayout(target_box)
        self.image_picker = NoScrollComboBox()
        self.labels_picker = NoScrollComboBox()
        target_layout.addRow(QLabel("Image layer"), self.image_picker)
        target_layout.addRow(QLabel("Labels layer"), self.labels_picker)
        self.load_btn = QPushButton("Load selection")
        self.load_btn.clicked.connect(self._on_load_clicked)
        target_layout.addRow(self.load_btn)
        layout.addWidget(target_box)

        mode_box = QGroupBox("Marking mode")
        mode_layout = QVBoxLayout(mode_box)
        self._mode_buttons = QButtonGroup(self)
        self._mode_buttons.setExclusive(True)
        for idx, (key, label) in enumerate(_MARKING_MODES):
            rb = QRadioButton(label)
            rb.setProperty("mode_key", key)
            if idx == 0:
                rb.setChecked(True)
            self._mode_buttons.addButton(rb, idx)
            mode_layout.addWidget(rb)
        self._mode_buttons.buttonClicked.connect(self._on_mode_change)
        layout.addWidget(mode_box)

        params_box = QGroupBox("Parameters")
        params_layout = QFormLayout(params_box)
        self.growth_k_spin = NoScrollDoubleSpinBox()
        self.growth_k_spin.setRange(0.0, 20.0)
        self.growth_k_spin.setSingleStep(0.25)
        self.growth_k_spin.setValue(1.5)
        self.region_scale_spin = NoScrollDoubleSpinBox()
        self.region_scale_spin.setRange(0.0, 1.0)
        self.region_scale_spin.setSingleStep(0.05)
        self.region_scale_spin.setValue(0.5)
        self.min_size_spin = NoScrollDoubleSpinBox()
        self.min_size_spin.setDecimals(0)
        self.min_size_spin.setRange(0, 1_000_000)
        self.min_size_spin.setValue(16)
        params_layout.addRow(QLabel("Add-point growth k·σ"), self.growth_k_spin)
        params_layout.addRow(QLabel("Add-region SNR scale"), self.region_scale_spin)
        params_layout.addRow(QLabel("Min size (voxels)"), self.min_size_spin)
        layout.addWidget(params_box)

        actions_box = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_box)
        row1 = QHBoxLayout()
        self.rebuild_btn = QPushButton("Rebuild ROI")
        self.rebuild_btn.clicked.connect(self._on_rebuild)
        self.reset_btn = QPushButton("Reset markings")
        self.reset_btn.clicked.connect(self._on_reset_markings)
        row1.addWidget(self.rebuild_btn)
        row1.addWidget(self.reset_btn)
        actions_layout.addLayout(row1)
        row2 = QHBoxLayout()
        self.commit_btn = QPushButton("Commit to labels layer")
        self.commit_btn.clicked.connect(self._on_commit)
        self.skip_btn = QPushButton("Skip sample")
        self.skip_btn.clicked.connect(self._on_skip)
        row2.addWidget(self.commit_btn)
        row2.addWidget(self.skip_btn)
        actions_layout.addLayout(row2)
        row3 = QHBoxLayout()
        self.close_btn = QPushButton("Close review")
        self.close_btn.clicked.connect(self._on_close_review)
        row3.addStretch(1)
        row3.addWidget(self.close_btn)
        actions_layout.addLayout(row3)
        layout.addWidget(actions_box)

        self.status_label = QLabel("No layer loaded.")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status_label)

        self._refresh_pickers()
        try:
            self.viewer.layers.events.inserted.connect(lambda _e: self._refresh_pickers())
            self.viewer.layers.events.removed.connect(lambda _e: self._refresh_pickers())
        except Exception:
            pass

        self._set_active(False)

    # ------------------------------------------------------------------ picker

    def _refresh_pickers(self) -> None:
        from napari.layers import Image as ImageLayer, Labels as LabelsLayer

        image_names = []
        labels_names = []
        for L in self.viewer.layers:
            if isinstance(L, LabelsLayer):
                labels_names.append(L.name)
            elif isinstance(L, ImageLayer):
                image_names.append(L.name)

        for picker, names in (
            (self.image_picker, image_names),
            (self.labels_picker, labels_names),
        ):
            current = picker.currentText()
            picker.blockSignals(True)
            picker.clear()
            picker.addItems(names)
            if current in names:
                picker.setCurrentText(current)
            picker.blockSignals(False)

    def request_layers(self, image_layer: str, labels_layer: str) -> None:
        """Programmatic entry: select the given layers and load."""
        self._refresh_pickers()
        if image_layer in [
            self.image_picker.itemText(i) for i in range(self.image_picker.count())
        ]:
            self.image_picker.setCurrentText(image_layer)
        if labels_layer in [
            self.labels_picker.itemText(i) for i in range(self.labels_picker.count())
        ]:
            self.labels_picker.setCurrentText(labels_layer)
        self._on_load_clicked()

    # ------------------------------------------------------------------ load

    def _on_load_clicked(self) -> None:
        image_name = self.image_picker.currentText()
        labels_name = self.labels_picker.currentText()
        if not image_name or not labels_name:
            self.status_label.setText("Pick an Image layer and a Labels layer.")
            return
        try:
            img_layer = self.viewer.layers[image_name]
            labels_layer = self.viewer.layers[labels_name]
        except KeyError:
            self.status_label.setText("Selected layer is no longer present.")
            return

        from imajin.analysis.segmentation import robust_background_sigma

        target = np.asarray(img_layer.data, dtype=np.float32)
        labels = np.asarray(labels_layer.data, dtype=np.int32)
        if target.shape != labels.shape:
            self.status_label.setText(
                f"Shape mismatch: image {target.shape} vs labels {labels.shape}."
            )
            return
        if target.ndim not in (2, 3):
            self.status_label.setText(
                f"Review supports 2D or 3D layers; got {target.ndim}D."
            )
            return

        self._tear_down_scratch_layers()

        self._target_layer_name = image_name
        self._labels_layer_name = labels_name
        self._original_corrected = target
        self._original_labels = labels
        self._current_labels = labels.copy()

        meta = dict(getattr(labels_layer, "metadata", {}) or {})
        self._noise_sigma = float(
            meta.get("noise_sigma")
            or robust_background_sigma(target)
        )
        self._base_threshold = float(meta.get("threshold", 0.0))

        self._scratch_names = _layer_suffix_names(image_name)
        self._build_scratch_layers()
        self._set_active(True)
        self._update_status(loaded=True)

    # ------------------------------------------------------ scratch layers

    def _build_scratch_layers(self) -> None:
        from napari.layers import Points, Shapes

        assert self._original_corrected is not None
        assert self._current_labels is not None

        mip_image = _mip(self._original_corrected)
        mip_labels = _mip(self._current_labels).astype(np.int32)

        names = self._scratch_names
        viewer = self.viewer
        viewer.add_image(
            mip_image,
            name=names["mip_image"],
            colormap="gray",
            blending="additive",
        )
        viewer.add_labels(mip_labels, name=names["mip_labels"])

        add_points = Points(
            np.empty((0, 2), dtype=float),
            name=names["add_points"],
            face_color=_ADD_COLOR,
            border_color="white",
            size=6,
            symbol="o",
        )
        remove_points = Points(
            np.empty((0, 2), dtype=float),
            name=names["remove_points"],
            face_color=_REMOVE_COLOR,
            border_color="white",
            size=6,
            symbol="x",
        )
        viewer.add_layer(add_points)
        viewer.add_layer(remove_points)

        add_shapes = Shapes(
            name=names["add_shapes"],
            edge_color=_ADD_COLOR,
            face_color=[0.24, 0.84, 0.55, 0.25],
            edge_width=2,
        )
        remove_shapes = Shapes(
            name=names["remove_shapes"],
            edge_color=_REMOVE_COLOR,
            face_color=[0.86, 0.31, 0.26, 0.25],
            edge_width=2,
        )
        viewer.add_layer(add_shapes)
        viewer.add_layer(remove_shapes)

        self._on_mode_change(self._mode_buttons.checkedButton())

    def _tear_down_scratch_layers(self) -> None:
        for layer_name in self._scratch_names.values():
            if layer_name and layer_name in self.viewer.layers:
                try:
                    self.viewer.layers.remove(layer_name)
                except Exception:
                    pass
        self._scratch_names = {}

    # ------------------------------------------------------------ active state

    def _set_active(self, active: bool) -> None:
        for w in (
            self.rebuild_btn,
            self.reset_btn,
            self.commit_btn,
            self.skip_btn,
            self.close_btn,
        ):
            w.setEnabled(active)

    # ------------------------------------------------------------- marking mode

    def _on_mode_change(self, _button: Any) -> None:
        btn = self._mode_buttons.checkedButton()
        if btn is None:
            return
        mode = btn.property("mode_key")
        target_name: str | None = None
        mode_str: str | None = None
        if mode == "add_point":
            target_name = self._scratch_names.get("add_points")
            mode_str = "add"
        elif mode == "remove_point":
            target_name = self._scratch_names.get("remove_points")
            mode_str = "add"
        elif mode == "add_region":
            target_name = self._scratch_names.get("add_shapes")
            mode_str = "add_polygon"
        elif mode == "remove_region":
            target_name = self._scratch_names.get("remove_shapes")
            mode_str = "add_polygon"
        if target_name and target_name in self.viewer.layers:
            layer = self.viewer.layers[target_name]
            try:
                self.viewer.layers.selection.active = layer
                layer.mode = mode_str
            except Exception:
                pass

    # ------------------------------------------------------------- gather + rebuild

    def _collect_points(self, layer_key: str) -> list[tuple[int, int]]:
        name = self._scratch_names.get(layer_key)
        if not name or name not in self.viewer.layers:
            return []
        data = np.asarray(self.viewer.layers[name].data)
        if data.size == 0:
            return []
        # Points layer on a 2D image stores rows as (y, x).
        return [(int(round(p[0])), int(round(p[1]))) for p in data]

    def _collect_regions(self, layer_key: str) -> list[np.ndarray]:
        """Rasterize Shapes layer polygons into YX boolean masks."""
        name = self._scratch_names.get(layer_key)
        if not name or name not in self.viewer.layers:
            return []
        layer = self.viewer.layers[name]
        try:
            shape_data = list(layer.data)
        except Exception:
            shape_data = []
        if not shape_data:
            return []
        assert self._original_corrected is not None
        Y, X = self._original_corrected.shape[-2:]
        regions: list[np.ndarray] = []
        for verts in shape_data:
            arr = np.asarray(verts, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            yx = arr[:, -2:]
            mask = _rasterize_polygon(yx, (Y, X))
            if mask.any():
                regions.append(mask)
        return regions

    def _on_rebuild(self) -> None:
        from imajin.analysis.interactive_roi import correct_roi_from_markings

        if self._original_corrected is None or self._original_labels is None:
            self.status_label.setText("No layer loaded.")
            return

        add_points = self._collect_points("add_points")
        remove_points = self._collect_points("remove_points")
        add_regions = self._collect_regions("add_shapes")
        remove_regions = self._collect_regions("remove_shapes")

        try:
            new_labels, info = correct_roi_from_markings(
                self._original_labels,
                self._original_corrected,
                add_points=add_points,
                remove_points=remove_points,
                add_regions=add_regions,
                remove_regions=remove_regions,
                noise_sigma=self._noise_sigma,
                base_threshold=self._base_threshold,
                add_seed_growth_k_snr=float(self.growth_k_spin.value()),
                region_min_snr_scale=float(self.region_scale_spin.value()),
                min_size=int(self.min_size_spin.value()),
            )
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Rebuild failed: {exc}")
            return

        self._current_labels = new_labels
        mip_name = self._scratch_names.get("mip_labels")
        if mip_name and mip_name in self.viewer.layers:
            self.viewer.layers[mip_name].data = _mip(new_labels).astype(np.int32)

        self._update_status(rebuild_info=info)

    def _on_reset_markings(self) -> None:
        from napari.layers import Points, Shapes

        for key in ("add_points", "remove_points"):
            name = self._scratch_names.get(key)
            if name and name in self.viewer.layers:
                layer = self.viewer.layers[name]
                if isinstance(layer, Points):
                    layer.data = np.empty((0, 2), dtype=float)
        for key in ("add_shapes", "remove_shapes"):
            name = self._scratch_names.get(key)
            if name and name in self.viewer.layers:
                layer = self.viewer.layers[name]
                if isinstance(layer, Shapes):
                    layer.data = []
        # Restore current labels MIP to last rebuilt (or original).
        labels = self._current_labels
        if labels is None:
            labels = self._original_labels
        mip_name = self._scratch_names.get("mip_labels")
        if labels is not None and mip_name and mip_name in self.viewer.layers:
            self.viewer.layers[mip_name].data = _mip(labels).astype(np.int32)
        self.status_label.setText("Markings cleared.")

    def _on_commit(self) -> None:
        if (
            self._labels_layer_name is None
            or self._current_labels is None
            or self._labels_layer_name not in self.viewer.layers
        ):
            self.status_label.setText("No labels layer to commit to.")
            return
        layer = self.viewer.layers[self._labels_layer_name]
        layer.data = self._current_labels.astype(np.int32)
        meta = dict(getattr(layer, "metadata", {}) or {})
        meta["reviewed"] = True
        meta["review_noise_sigma"] = self._noise_sigma
        layer.metadata = meta
        info = {
            "image_layer": self._target_layer_name,
            "labels_layer": self._labels_layer_name,
            "final_voxels": int((self._current_labels > 0).sum()),
            "final_objects": int(self._current_labels.max()) if self._current_labels.size else 0,
            "noise_sigma": self._noise_sigma,
        }
        self._update_status(committed=True)
        self.review_committed.emit(info)
        from imajin.agent.review_checkpoint import is_review_active, notify_review_committed
        if is_review_active():
            notify_review_committed(**info)

    def _on_skip(self) -> None:
        info = {
            "image_layer": self._target_layer_name,
            "labels_layer": self._labels_layer_name,
            "reason": "user_skipped",
        }
        self.status_label.setText("Sample skipped.")
        self.review_skipped.emit(info)
        from imajin.agent.review_checkpoint import is_review_active, notify_review_skipped
        if is_review_active():
            notify_review_skipped(**info)

    def _on_close_review(self) -> None:
        self._tear_down_scratch_layers()
        self._target_layer_name = None
        self._labels_layer_name = None
        self._original_corrected = None
        self._original_labels = None
        self._current_labels = None
        self._set_active(False)
        self.status_label.setText("Review closed.")
        self._refresh_pickers()

    # ------------------------------------------------------------------ status

    def _update_status(
        self,
        *,
        loaded: bool = False,
        rebuild_info: dict[str, Any] | None = None,
        committed: bool = False,
    ) -> None:
        lines: list[str] = []
        if self._target_layer_name and self._labels_layer_name:
            lines.append(
                f"Image: {self._target_layer_name}   "
                f"Labels: {self._labels_layer_name}"
            )
        lines.append(
            f"σ (background) = {self._noise_sigma:.3f}   "
            f"base threshold = {self._base_threshold:.3f}"
        )
        if loaded:
            voxels = int((self._original_labels > 0).sum()) if self._original_labels is not None else 0
            lines.append(f"Initial voxels in label > 0: {voxels:,}")
        if rebuild_info is not None:
            lines.append(
                f"Rebuilt: {rebuild_info['final_objects']} objects, "
                f"{rebuild_info['final_voxels']:,} voxels "
                f"(add+{rebuild_info['add_points_voxels'] + rebuild_info['add_regions_voxels']:,}, "
                f"remove-{rebuild_info['remove_points_voxels'] + rebuild_info['remove_regions_voxels']:,}, "
                f"skipped points: {rebuild_info['skipped_points']})"
            )
        if committed:
            lines.append("✓ Committed to labels layer.")
        self.status_label.setText("\n".join(lines))


def _rasterize_polygon(
    vertices_yx: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Rasterize a polygon given as (N, 2) YX vertices into an HxW bool mask."""
    from skimage.draw import polygon as sk_polygon

    Y, X = shape
    if vertices_yx.shape[0] < 3:
        # Treat a 2-vertex shape (rectangle from napari) as its bbox.
        if vertices_yx.shape[0] == 2:
            y0, x0 = vertices_yx[0]
            y1, x1 = vertices_yx[1]
            ymin = max(0, int(np.floor(min(y0, y1))))
            ymax = min(Y, int(np.ceil(max(y0, y1))))
            xmin = max(0, int(np.floor(min(x0, x1))))
            xmax = min(X, int(np.ceil(max(x0, x1))))
            mask = np.zeros(shape, dtype=bool)
            mask[ymin:ymax, xmin:xmax] = True
            return mask
        return np.zeros(shape, dtype=bool)
    rr, cc = sk_polygon(vertices_yx[:, 0], vertices_yx[:, 1], shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask
