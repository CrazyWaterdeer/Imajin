"""Topological persistence features (registration-free shape descriptors).

navis persistence vectors summarize a neuron's branching topology via the
distribution of geodesic distances from the root. They are **translation- and
rotation-invariant** (unlike NBLAST), so they enrich the morphometric feature
vector for unregistered confocal traces without needing a template. They are still
scale-sensitive (path lengths scale with size), so the matcher treats them like the
other physical-length features (used only when query and library share microns).

``navis`` is the optional ``connectome`` extra and is imported lazily; this module
imports clean without it. Any failure (no navis, degenerate skeleton) returns
``None`` so callers fall back to the pure-Python morphometric features.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from imajin.analysis.morphology_nblast import navis_available


def persistence_available() -> bool:
    return navis_available()


def persistence_features_from_swc(
    swc_path: str | Path, *, samples: int = 64
) -> dict[str, float] | None:
    """Persistence feature vector for a neuron given as an SWC file.

    Returns ``{"pers_00": .., ..., "pers_NN": ..}`` (length ``samples``) or ``None``
    if navis is unavailable or the skeleton is too degenerate for persistence.
    """
    if not navis_available():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import navis

            neuron = navis.read_swc(str(swc_path))
            if isinstance(neuron, navis.NeuronList):
                if len(neuron) == 0:
                    return None
                neuron = neuron[0]
            if getattr(neuron, "nodes", None) is None or neuron.nodes.shape[0] < 3:
                return None
            vec = np.asarray(navis.persistence_vectors(neuron, samples=samples)[0]).ravel()
        if vec.size != samples or not np.all(np.isfinite(vec)):
            return None
        return {f"pers_{i:02d}": float(v) for i, v in enumerate(vec)}
    except Exception:
        return None
