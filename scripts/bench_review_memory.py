"""Memory benchmark for the SNR/ROI review path.

Loads a real LSM z-stack, runs a simple auto-segmentation, then measures
peak memory through:
  (a) the dock load step (emulated — no Qt, just the same np.asarray /
      copy semantics the dock used to perform), and
  (b) correct_roi_from_markings with a realistic set of user markings.

Run on `master` (or a baseline commit) vs. the patched HEAD and compare
RSS deltas. The script also prints tracemalloc-based "tracked allocated
bytes" for the algorithm call so the comparison is independent of
napari/Qt caches.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import tracemalloc
from pathlib import Path

import numpy as np

# Ensure the in-tree imajin package is importable regardless of cwd.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import psutil  # noqa: E402

PROC = psutil.Process(os.getpid())


def rss_mb() -> float:
    return PROC.memory_info().rss / (1024 * 1024)


def banner(label: str) -> None:
    print(f"\n=== {label} ===  rss={rss_mb():.1f} MB")


def load_lsm_one_channel(path: Path, channel: int) -> np.ndarray:
    import tifffile

    banner(f"open {path.name}")
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        axes = series.axes
        shape = series.shape
        dtype = series.dtype
        print(f"  axes={axes} shape={shape} dtype={dtype}")
        data = series.asarray()
    banner("loaded full series")
    print(f"  data.shape={data.shape} data.dtype={data.dtype} "
          f"nbytes={data.nbytes / 1024**2:.1f} MB")

    # Reduce to (Z, Y, X) for a single channel; LSM is typically TZCYX or ZCYX.
    a = axes.upper()
    if "C" in a:
        c_axis = a.index("C")
        data = np.take(data, channel, axis=c_axis)
        a = a.replace("C", "")
    if "T" in a:
        t_axis = a.index("T")
        data = np.take(data, 0, axis=t_axis)
        a = a.replace("T", "")
    # Now expect ZYX or YX.
    assert a in ("ZYX", "YX"), f"unexpected axes after reduce: {a}"
    banner(f"single-channel slice ({a})")
    print(f"  shape={data.shape} dtype={data.dtype} "
          f"nbytes={data.nbytes / 1024**2:.1f} MB")
    return data


def quick_threshold_labels(image: np.ndarray) -> np.ndarray:
    """Cheap proxy for an auto segmentation result.

    We just take a high percentile of the finite pixels as a threshold so
    we get an auto_labels array with the right shape/dtype. Quality is
    irrelevant — the goal is to feed correct_roi_from_markings with a
    realistic mix of label-1/label-0 voxels.
    """
    from scipy import ndimage as ndi

    img = np.asarray(image, dtype=np.float32)
    finite = img[np.isfinite(img)]
    thr = float(np.percentile(finite, 99.0))
    binary = (img > thr) & np.isfinite(img)
    labels, _ = ndi.label(binary)
    return labels.astype(np.int32)


def emulate_dock_load_before(image: np.ndarray, labels: np.ndarray) -> tuple:
    """Match the pre-patch _on_load_clicked allocations exactly."""
    target = np.asarray(image, dtype=np.float32)
    lab = np.asarray(labels, dtype=np.int32)
    original_corrected = target
    original_labels = lab
    current_labels = lab.copy()
    return original_corrected, original_labels, current_labels


def emulate_dock_load_after(image: np.ndarray, labels: np.ndarray) -> tuple:
    """Match the patched _on_load_clicked allocations."""
    target = np.asarray(image, dtype=np.float32)
    lab = np.asarray(labels, dtype=np.int32)
    original_corrected = target
    original_labels = lab
    current_labels = lab  # shared until rebuild
    return original_corrected, original_labels, current_labels


def make_markings(
    shape: tuple[int, ...],
    *,
    n_add_points: int,
    n_remove_points: int,
    n_add_regions: int,
    n_remove_regions: int,
):
    """Synthesize a realistic set of markings on the YX plane."""
    Y, X = shape[-2:]
    rng = np.random.default_rng(0)
    add_points = [
        (int(rng.integers(Y // 4, 3 * Y // 4)), int(rng.integers(X // 4, 3 * X // 4)))
        for _ in range(n_add_points)
    ]
    remove_points = [
        (int(rng.integers(Y // 4, 3 * Y // 4)), int(rng.integers(X // 4, 3 * X // 4)))
        for _ in range(n_remove_points)
    ]
    regions = []
    for _ in range(n_add_regions + n_remove_regions):
        m = np.zeros((Y, X), dtype=bool)
        y0 = int(rng.integers(0, Y - 20))
        x0 = int(rng.integers(0, X - 20))
        m[y0:y0 + 20, x0:x0 + 20] = True
        regions.append(m)
    return (
        add_points,
        remove_points,
        regions[:n_add_regions],
        regions[n_add_regions:],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lsm", type=Path)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--add-points", type=int, default=5)
    ap.add_argument("--remove-points", type=int, default=3)
    ap.add_argument("--add-regions", type=int, default=2)
    ap.add_argument("--remove-regions", type=int, default=2)
    ap.add_argument(
        "--mode",
        choices=("after", "before"),
        default="after",
        help="Whether to emulate the pre-patch dock load (forces an extra "
        "labels.copy) or the patched share-until-rebuild load.",
    )
    args = ap.parse_args()

    if not args.lsm.exists():
        print(f"file not found: {args.lsm}", file=sys.stderr)
        return 2

    banner("startup")
    image = load_lsm_one_channel(args.lsm, args.channel)
    gc.collect()
    banner("after load_lsm_one_channel + gc")

    labels = quick_threshold_labels(image)
    gc.collect()
    banner("after quick_threshold_labels")
    print(f"  labels>0 voxels: {int((labels > 0).sum()):,}")

    if args.mode == "before":
        oc, ol, cl = emulate_dock_load_before(image, labels)
    else:
        oc, ol, cl = emulate_dock_load_after(image, labels)
    gc.collect()
    banner(f"after dock load (mode={args.mode})")
    print(f"  oc.shape={oc.shape} ol.shape={ol.shape} "
          f"cl is ol? {cl is ol}")

    # Synthetic markings.
    add_p, rem_p, add_r, rem_r = make_markings(
        image.shape,
        n_add_points=args.add_points,
        n_remove_points=args.remove_points,
        n_add_regions=args.add_regions,
        n_remove_regions=args.remove_regions,
    )

    # Track Python-level allocations inside correct_roi_from_markings.
    from imajin.analysis.interactive_roi import correct_roi_from_markings

    rss_before = rss_mb()
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    new_labels, info = correct_roi_from_markings(
        ol,
        oc,
        add_points=add_p,
        remove_points=rem_p,
        add_regions=add_r,
        remove_regions=rem_r,
        noise_sigma=1.0,
        base_threshold=float(np.percentile(oc[np.isfinite(oc)], 99.0)),
        min_size=16,
    )

    snap_after = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = rss_mb()

    banner("after correct_roi_from_markings")
    print(f"  rss delta: {rss_after - rss_before:+.1f} MB")
    print(f"  tracemalloc peak inside call: {peak / 1024**2:.1f} MB")
    print(f"  tracemalloc current after:   {current / 1024**2:.1f} MB")
    print(f"  result voxels: {int((new_labels > 0).sum()):,}")
    print(f"  info: {info}")

    # Top allocation diffs to highlight where the budget went.
    stats = snap_after.compare_to(snap_before, "filename")
    print("\n  top diffs (by filename):")
    for stat in stats[:8]:
        print(f"    {stat}")

    banner("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
