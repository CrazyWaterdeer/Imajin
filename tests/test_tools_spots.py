from __future__ import annotations

import numpy as np

from imajin import session as state
from imajin.tools import spots


def _puncta_2d(shape, centers, sigma=2.0, amp=800.0, bg=40.0):
    img = np.full(shape, bg, dtype=np.float32)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for cy, cx in centers:
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    rng = np.random.default_rng(0)
    return img + rng.normal(0, 2.0, size=shape).astype(np.float32)


def _puncta_3d(shape, centers, sigma=2.0, amp=800.0, bg=40.0):
    img = np.full(shape, bg, dtype=np.float32)
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    for cz, cy, cx in centers:
        img += amp * np.exp(
            -((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)
        )
    return img


def test_detect_spots_2d_counts_and_microns(viewer) -> None:
    state.reset_tables()
    centers = [(20, 25), (60, 40), (30, 90), (95, 70)]
    viewer.add_image(_puncta_2d((128, 128), centers), name="ch1", scale=(0.2, 0.2))

    res = spots.detect_spots("ch1", min_diameter_um=0.6, max_diameter_um=1.8)

    assert res["n_spots"] == len(centers)
    assert res["points_layer"] in viewer.layers
    df = state.get_table(res["table_name"])
    assert {"spot_id", "y", "x", "y_um", "x_um", "diameter_um", "snr", "quality"} <= set(
        df.columns
    )
    assert "intensity_ch1" in df.columns
    # µm columns are index * scale (data-coord layer geometry, µm in the table).
    assert np.allclose(df["y_um"], df["y"] * 0.2)
    # Every planted centre is matched within ~1 px.
    found = df[["y", "x"]].to_numpy()
    for cy, cx in centers:
        assert np.min(np.hypot(found[:, 0] - cy, found[:, 1] - cx)) < 2.0


def test_detect_spots_3d_anisotropic(viewer) -> None:
    state.reset_tables()
    centers = [(4, 16, 16), (5, 40, 44), (3, 24, 50)]
    viewer.add_image(_puncta_3d((9, 64, 64), centers), name="vol", scale=(1.0, 0.3, 0.3))

    res = spots.detect_spots(
        "vol", min_diameter_um=0.9, max_diameter_um=2.4, mode="3d"
    )

    assert res["detection_mode"] == "3d"
    assert res["n_spots"] == len(centers)
    df = state.get_table(res["table_name"])
    assert {"z", "y", "x", "z_um", "axial_diameter_um"} <= set(df.columns)
    assert np.allclose(df["z_um"], df["z"] * 1.0)


def test_detect_spots_2d_projection_localises_z(viewer) -> None:
    state.reset_tables()
    # Two puncta on different z planes; projection should find both, z by argmax.
    centers = [(2, 20, 20), (6, 45, 45)]
    viewer.add_image(_puncta_3d((9, 64, 64), centers), name="stk", scale=(1.0, 0.3, 0.3))

    res = spots.detect_spots(
        "stk", min_diameter_um=0.9, max_diameter_um=2.4, mode="2d_projection"
    )

    assert res["detection_mode"] == "2d_projection"
    assert res["n_spots"] == 2
    assert any("argmax" in w for w in res["warnings"])
    df = state.get_table(res["table_name"])
    zmap = {(round(r.y), round(r.x)): r.z for r in df.itertuples()}
    # z recovered near each planted plane.
    assert abs(zmap[(20, 20)] - 2) <= 1
    assert abs(zmap[(45, 45)] - 6) <= 1


def test_detect_spots_boundary_mask_filters(viewer) -> None:
    state.reset_tables()
    centers = [(20, 20), (100, 100)]
    viewer.add_image(_puncta_2d((128, 128), centers), name="chb", scale=(0.2, 0.2))
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[:64, :64] = 1  # only the first punctum is inside
    viewer.add_labels(mask, name="roi")

    res = spots.detect_spots(
        "chb", min_diameter_um=0.6, max_diameter_um=1.8, boundary_mask="roi"
    )

    assert res["n_spots"] == 1
    df = state.get_table(res["table_name"])
    assert df["y"].iloc[0] < 64 and df["x"].iloc[0] < 64


def test_detect_spots_empty_image_is_graceful(viewer) -> None:
    state.reset_tables()
    viewer.add_image(np.full((64, 64), 30.0, dtype=np.float32), name="flat", scale=(0.2, 0.2))

    res = spots.detect_spots("flat", min_diameter_um=0.6, max_diameter_um=1.8)

    assert res["n_spots"] == 0
    assert any("no spots" in w for w in res["warnings"])
    df = state.get_table(res["table_name"])
    assert len(df) == 0


def test_compute_spots_qc_flags_low_count(viewer) -> None:
    state.reset_tables()
    centers = [(20, 25), (60, 40)]
    viewer.add_image(_puncta_2d((128, 128), centers), name="chq", scale=(0.2, 0.2))
    res = spots.detect_spots("chq", min_diameter_um=0.6, max_diameter_um=1.8)

    qc = spots.compute_spots_qc(res["table_name"], min_count=5)
    assert qc["status"] == "fail"
    assert qc["n_spots"] == 2
