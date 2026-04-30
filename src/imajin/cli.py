from __future__ import annotations

import argparse
import importlib
import os
import sys


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _setup_wsl_env() -> None:
    if not _is_wsl():
        return
    # Mesa 25.x on WSLg defaults to ZINK and fails pdev selection, falling back
    # to llvmpipe (software). GALLIUM_DRIVER=d3d12 routes through /dev/dxg for
    # GPU-accelerated rendering on WSLg ≥1.0.71.
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
    if not os.environ.get("XDG_RUNTIME_DIR") and os.path.exists("/mnt/wslg/runtime-dir"):
        os.environ["XDG_RUNTIME_DIR"] = "/mnt/wslg/runtime-dir"


def _check_import(module: str) -> tuple[bool, str]:
    try:
        m = importlib.import_module(module)
        return True, str(getattr(m, "__version__", "?"))
    except Exception as e:
        return False, str(e)[:80]


def _check_gui_renderer() -> tuple[bool, str]:
    try:
        from OpenGL import GL
        from vispy import app as vapp

        canvas = vapp.Canvas(show=False, size=(8, 8), keys=None)
        renderer = GL.glGetString(GL.GL_RENDERER).decode()
        canvas.close()
        return ("llvmpipe" not in renderer.lower()), renderer
    except Exception as e:
        return False, f"probe failed: {type(e).__name__}: {str(e)[:80]}"


def _doctor() -> int:
    print("imajin doctor")
    print("=" * 48)

    ok = True

    print(f"\n[Python] {sys.version.split()[0]}")

    print("\n[Core imports]")
    for mod in [
        "numpy",
        "scipy",
        "pandas",
        "skimage",
        "napari",
        "magicgui",
        "qtpy",
        "tifffile",
        "bioio",
        "bioio_czi",
        "xarray",
        "dask",
        "anthropic",
        "openai",
        "pydantic",
        "cellpose",
        "skan",
        "btrack",
    ]:
        good, info = _check_import(mod)
        marker = "ok  " if good else "MISS"
        print(f"  [{marker}] {mod:<14} {info}")
        if not good:
            ok = False

    print("\n[CUDA]")
    try:
        import torch

        print(f"  torch: {torch.__version__}")
        cuda = torch.cuda.is_available()
        print(f"  cuda available: {cuda}")
        if cuda:
            print(f"  device: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  vram: {vram:.1f} GB")
        else:
            print("  WARNING: CUDA not available — Cellpose-SAM will run on CPU (slow).")
    except ImportError as e:
        print(f"  torch not installed ({e}). Run: uv sync")
        ok = False

    print("\n[Display]")
    ok_gui, info = _check_gui_renderer()
    marker = "ok  " if ok_gui else "WARN"
    print(f"  [{marker}] GL_RENDERER: {info}")
    if "llvmpipe" in info.lower():
        print("  WARNING: software rendering — napari will be sluggish.")
        print("  On WSL, imajin auto-sets GALLIUM_DRIVER=d3d12; check that it")
        print("  isn't being overridden in your shell.")

    print("\n[Providers]")
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        present = bool(os.environ.get(var))
        print(f"  {var}: {'set' if present else 'not set'}")

    print()
    return 0 if ok else 1


def _launch_gui() -> int:
    from imajin.ui.main import launch

    return launch()


def main() -> int:
    _setup_wsl_env()
    parser = argparse.ArgumentParser(
        prog="imajin",
        description="GUI-based AI agent for confocal microscopy analysis.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Verify environment (CUDA, deps, API keys) and exit.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run bundled demo on sample LSM (Phase 4+).",
    )
    args = parser.parse_args()

    if args.doctor:
        return _doctor()
    if args.demo:
        print("demo not yet implemented — Phase 4")
        return 1

    return _launch_gui()


if __name__ == "__main__":
    sys.exit(main())
