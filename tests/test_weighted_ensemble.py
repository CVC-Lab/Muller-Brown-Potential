"""Probability conservation, path genealogy, and first-passage estimators."""

import numpy as np
import pytest
import torch

from muller_brown.constants import MULLER_BROWN_MINIMA
from muller_brown.weighted_ensemble import (
    TransitionPath,
    WeightedPathReservoir,
    path_progress,
    resample_bins,
    sample_weighted_ensemble,
)


def test_resampling_conserves_each_bins_mass():
    weights = np.array([0.6, 0.1, 0.2, 0.1])
    bins = np.array([0, 0, 2, 2])
    parents, new_weights = resample_bins(weights, bins, 7, np.random.default_rng(0))
    assert len(parents) == 14
    for index in np.unique(bins):
        assert new_weights[bins[parents] == index].sum() == pytest.approx(
            weights[bins == index].sum()
        )


def test_resampling_preserves_weighted_history_in_expectation():
    rng = np.random.default_rng(12)
    estimates = []
    for _ in range(5000):
        parents, weights = resample_bins(np.array([0.9, 0.1]), np.zeros(2), 4, rng)
        estimates.append(weights[parents == 1].sum())
    assert np.mean(estimates) == pytest.approx(0.1, abs=0.005)


def test_balanced_bins_keep_distinct_histories():
    weights = np.full(8, 1 / 8)
    parents, new_weights = resample_bins(
        weights, np.zeros(8), 8, np.random.default_rng(0)
    )
    np.testing.assert_array_equal(parents, np.arange(8))
    np.testing.assert_array_equal(new_weights, weights)


def test_output_reservoir_keeps_original_weights_below_capacity():
    reservoir = WeightedPathReservoir(3, 42)
    for weight in [0.1, 0.9]:
        reservoir.add(TransitionPath(np.zeros((2, 2)), np.zeros((2, 2)), weight, 0, 1))
    np.testing.assert_array_equal(reservoir.weights, [0.1, 0.9])
    assert reservoir.event_ids == [0, 1]
    assert not reservoir.resampled


def test_output_reservoir_preserves_weighted_statistics_in_expectation():
    estimates = []
    for seed in range(3000):
        reservoir = WeightedPathReservoir(1, seed)
        for weight in [0.9, 0.1]:
            reservoir.add(
                TransitionPath(np.zeros((2, 2)), np.zeros((2, 2)), weight, 0, 1)
            )
        assert reservoir.weights.sum() == pytest.approx(1.0)
        estimates.append(reservoir.event_ids[0] == 1)
    assert np.mean(estimates) == pytest.approx(0.1, abs=0.02)


def test_progress_is_arclength_and_allows_backtracking():
    guide = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    x = np.array([[1.0, 2.0], [1.0, 1.0], [0.5, -0.1], [0.0, 0.0]])
    np.testing.assert_allclose(path_progress(x, guide), [1.0, 2 / 3, 1 / 6, 0.0])
    assert path_progress(np.empty((0, 2)), guide).shape == (0,)


class _Potential:
    def __call__(self, positions):
        source = torch.tensor(MULLER_BROWN_MINIMA[0], dtype=positions.dtype)
        return torch.sum((positions - source) ** 2, dim=1)

    def force(self, positions):
        return torch.zeros_like(positions)

    def get_saddle_points(self):
        return [(-0.822, 0.624), (0.212, 0.293)]


class _ScriptedSimulator:
    potential = _Potential()
    temperature = 3.0
    friction = 0.5
    mass = 1.0
    dt = 0.002
    dtype = torch.float64
    device = "cpu"

    def __init__(self):
        self.step = 0

    def _baoab_step(self, positions, velocities, forces):
        # Visits A again before B, then returns to A after hitting B. Checks that
        # the retained phase-space state survives resampling and cycle boundaries.
        assert torch.all(velocities[:, 0] == self.step)
        self.step += 1
        sequence = [
            (-0.45, 1.3),
            (0.0, 1.0),
            MULLER_BROWN_MINIMA[0],
            (-0.822, 0.624),
            MULLER_BROWN_MINIMA[2],
            MULLER_BROWN_MINIMA[1],
            MULLER_BROWN_MINIMA[0],
            MULLER_BROWN_MINIMA[0],
        ]
        x = torch.tensor(sequence[self.step - 1], dtype=self.dtype).repeat(
            len(positions), 1
        )
        v = torch.full_like(velocities, float(self.step))
        return x, v, forces


def test_extracts_last_A_to_first_B_across_resampling_boundaries():
    result = sample_weighted_ensemble(
        _ScriptedSimulator(),
        n_iterations=3,
        steps_per_iteration=4,
        walkers_per_bin=2,
        n_bins=4,
        progress=False,
    )
    assert len(result.paths) == 2
    assert sum(path.weight for path in result.paths) == pytest.approx(1.0)
    np.testing.assert_allclose(result.history[:, 1] + result.history[:, 2], 1.0)
    assert result.history[-1, 1] == 0.0
    for path in result.paths:
        assert (path.start_step, path.end_step) == (3, 6)
        assert len(path.positions) == path.end_step - path.start_step + 1
        np.testing.assert_array_equal(path.positions[0], MULLER_BROWN_MINIMA[0])
        np.testing.assert_array_equal(path.positions[-1], MULLER_BROWN_MINIMA[1])
        np.testing.assert_array_equal(path.velocities[:, 0], [3.0, 4.0, 5.0, 6.0])


def test_streaming_paths_preserves_the_ensemble_and_history():
    options = {
        "n_iterations": 3,
        "steps_per_iteration": 4,
        "walkers_per_bin": 2,
        "n_bins": 4,
        "progress": False,
    }
    retained = sample_weighted_ensemble(_ScriptedSimulator(), **options)
    collected = []
    streamed = sample_weighted_ensemble(
        _ScriptedSimulator(), on_transition=collected.append, **options
    )
    assert streamed.paths == []
    np.testing.assert_array_equal(streamed.history, retained.history)
    assert len(collected) == len(retained.paths)
    for first, second in zip(collected, retained.paths):
        assert first.weight == second.weight
        np.testing.assert_array_equal(first.positions, second.positions)
        np.testing.assert_array_equal(first.velocities, second.velocities)


class _JumpSimulator(_ScriptedSimulator):
    def _baoab_step(self, positions, velocities, forces):
        # A known first-passage law: at each step, independently jump to B with
        # probability 0.1; otherwise remain at A.
        jumps = torch.rand(len(positions)) < 0.1
        x = torch.tensor(MULLER_BROWN_MINIMA[0], dtype=self.dtype).repeat(
            len(positions), 1
        )
        x[jumps] = torch.tensor(MULLER_BROWN_MINIMA[1], dtype=self.dtype)
        return x, velocities, forces


@pytest.mark.statistical
def test_absorbed_weights_match_known_first_passage_probability():
    estimates = []
    for seed in range(200):
        result = sample_weighted_ensemble(
            _JumpSimulator(),
            n_iterations=5,
            steps_per_iteration=1,
            walkers_per_bin=16,
            n_bins=4,
            seed=seed,
            progress=False,
        )
        estimates.append(sum(path.weight for path in result.paths))
    assert np.mean(estimates) == pytest.approx(1 - 0.9**5, abs=0.025)


@pytest.mark.parametrize(
    "options",
    [
        {"n_iterations": 0},
        {"steps_per_iteration": 0},
        {"walkers_per_bin": 0},
        {"n_bins": 0},
        {"source_radius": float("nan")},
        {"target_radius": -1},
        {"source_radius": 2.0},
    ],
)
def test_invalid_parameters_fail_before_dynamics(options):
    with pytest.raises(ValueError):
        sample_weighted_ensemble(_ScriptedSimulator(), progress=False, **options)
