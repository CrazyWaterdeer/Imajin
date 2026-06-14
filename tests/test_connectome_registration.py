from __future__ import annotations

import numpy as np
import pytest

from imajin.analysis.connectome_registration import (
    connectome_backend_available,
    nblast_candidates,
    points_to_dotprops,
    warp_to_connectome_space,
)

requires_backend = pytest.mark.skipif(
    not connectome_backend_available(),
    reason="connectome extra (navis + flybrains) not installed",
)


def _line(axis, n=60, length=40.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, length, n)
    pts = np.zeros((n, 3))
    pts[:, axis] = t
    return pts + rng.normal(0, 0.3, (n, 3))


def test_points_to_dotprops_rejects_tiny_clouds() -> None:
    # Independent of navis: too few points -> None.
    assert points_to_dotprops(np.zeros((3, 3)), k=5) is None
    assert points_to_dotprops(np.zeros((10, 2)), k=5) is None  # wrong shape


@requires_backend
def test_nblast_candidates_ranks_morphologically_similar_higher() -> None:
    ref_x = points_to_dotprops(_line(0, seed=1), neuron_id=1)
    ref_y = points_to_dotprops(_line(1, seed=2), neuron_id=2)
    query = points_to_dotprops(_line(0, seed=3), neuron_id=99)  # an x-line, like ref_x

    cands = nblast_candidates(query, [(1, "x-type", ref_x), (2, "y-type", ref_y)], top_k=2)

    assert len(cands) == 2
    assert cands[0]["type"] == "x-type"  # query (x-line) is closest to the x-line
    assert cands[0]["nblast_score"] >= cands[1]["nblast_score"]
    assert {"id", "type", "nblast_score"} <= set(cands[0])


@requires_backend
def test_nblast_candidates_empty_inputs() -> None:
    assert nblast_candidates(None, []) == []
    assert nblast_candidates(points_to_dotprops(_line(0), neuron_id=1), []) == []


@requires_backend
def test_warp_to_connectome_space_returns_typed_status() -> None:
    # Robust to bridge-asset state: the brain->hemibrain bridge needs large JRC H5
    # transforms that may not be downloaded, so a typed status is the contract.
    out, status = warp_to_connectome_space(
        np.array([[250.0, 180.0, 90.0]]), source="JRC2018F", target="JRCFIB2018F"
    )
    assert status in {"ok", "needs_bridge_assets"}
    if status == "ok":
        assert out.shape == (1, 3)
    else:
        assert out is None
