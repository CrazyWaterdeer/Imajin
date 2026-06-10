from __future__ import annotations

from imajin.session import get_viewer


def viewer_layer_names() -> list[str]:
    viewer = get_viewer()
    return [str(layer.name) for layer in viewer.layers]


def remove_layers_by_name(layer_names: list[str]) -> list[str]:
    viewer = get_viewer()
    removed: list[str] = []
    for name in reversed(list(dict.fromkeys(str(n) for n in layer_names))):
        try:
            layer = viewer.layers[name]
        except Exception:
            continue
        try:
            viewer.layers.remove(layer)
            removed.append(name)
        except Exception:
            try:
                viewer.layers.remove(name)
                removed.append(name)
            except Exception:
                continue
    return removed
