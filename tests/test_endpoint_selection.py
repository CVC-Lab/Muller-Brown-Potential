"""Endpoint selection preserves full histories and matching observables."""

import numpy as np
import pytest

from muller_brown import select_paths_by_endpoint


def test_selection_uses_final_frame_and_preserves_path_histories():
    positions = np.array([
        [[0., 0.], [3., 0.], [0., 0.], [0., 0.]],
        [[3., 0.], [0., 0.], [2., 0.], [1.9, 0.]],
    ])
    data = {
        "positions": positions,
        "velocities": positions + 1,
        "forces": positions + 2,
        "potential_energy": np.arange(8).reshape(2, 4),
        "n_particles": 4,
        "dt": 0.1,
        "config": {"simulation": {"n_particles": 4}},
    }
    selected, indices = select_paths_by_endpoint(
        data, target=(3., 0.), reference=(0., 0.), max_distance_ratio=0.5
    )
    # Path 1 visits the target but finishes at the reference; path 2 is on the boundary.
    np.testing.assert_array_equal(indices, [0, 2])
    for name in ("positions", "velocities", "forces", "potential_energy"):
        np.testing.assert_array_equal(selected[name], data[name][:, [0, 2]])
        assert not np.shares_memory(selected[name], data[name])
    assert selected["n_particles"] == 2
    assert selected["config"]["simulation"]["n_particles"] == 2
    assert data["config"]["simulation"]["n_particles"] == 4
    assert selected["dt"] == 0.1


def test_empty_selection_and_nonfinite_endpoints():
    data = {"positions": np.array([[[0., 0.], [np.inf, 0.], [np.nan, 0.]]])}
    selected, indices = select_paths_by_endpoint(data, target=(3., 0.), reference=(0., 0.))
    assert indices.size == 0
    assert selected["positions"].shape == (1, 0, 2)
    assert selected["n_particles"] == 0


@pytest.mark.parametrize("ratio", [0, -0.5, 1.1, np.nan, np.inf])
def test_invalid_distance_ratio(ratio):
    with pytest.raises(ValueError, match="max_distance_ratio"):
        select_paths_by_endpoint(
            {"positions": np.zeros((1, 1, 2))}, (3., 0.), (0., 0.), ratio
        )


def test_empty_extraction_does_not_leave_stale_selected_files(tmp_path):
    from extract_paths import extract_mb_paths

    filenames = ("selected_paths.h5", "selected_paths.png", "selected_endpoints.png")
    for name in filenames:
        (tmp_path / name).write_text("previous selection")
    data = {"positions": np.tile([-0.558, 1.442], (1, 2, 1))}
    assert extract_mb_paths(data, tmp_path) is None
    assert (tmp_path / "endpoint_selection.csv").exists()
    assert all(not (tmp_path / name).exists() for name in filenames)
