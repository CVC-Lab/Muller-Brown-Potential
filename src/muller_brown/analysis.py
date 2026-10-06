"""Trajectory analysis and statistical computations."""

import numpy as np


def select_paths_by_endpoint(
    data: dict,
    target: tuple[float, float],
    reference: tuple[float, float],
    max_distance_ratio: float = 0.5,
) -> tuple[dict, np.ndarray]:
    """Select full paths with d(endpoint, target) <= ratio * d(endpoint, reference).

    Distances are Euclidean and use only the last saved frame. Nonfinite
    endpoints are excluded. Return a copy of the data with all available
    observables sliced along the particle axis, plus the original path indices.
    """
    if not np.isfinite(max_distance_ratio) or not 0 < max_distance_ratio <= 1:
        raise ValueError("max_distance_ratio must be finite and in (0, 1]")

    positions = np.asarray(data["positions"])
    if positions.ndim != 3 or positions.shape[0] == 0 or positions.shape[2] != 2:
        raise ValueError("positions must have shape (n_saved_times, n_paths, 2) with at least one frame")

    target = np.asarray(target, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if (
        target.shape != (2,) or reference.shape != (2,)
        or not np.isfinite(target).all() or not np.isfinite(reference).all()
        or np.array_equal(target, reference)
    ):
        raise ValueError("target and reference must be distinct finite 2D points")

    endpoints = positions[-1]
    target_distances = np.linalg.norm(endpoints - target, axis=1)
    reference_distances = np.linalg.norm(endpoints - reference, axis=1)
    mask = np.isfinite(endpoints).all(axis=1) & (
        target_distances <= max_distance_ratio * reference_distances
    )
    indices = np.flatnonzero(mask)

    selected = data.copy()
    for observable in ("positions", "velocities", "forces", "potential_energy"):
        if observable in data:
            selected[observable] = data[observable][:, indices].copy()
    selected["n_particles"] = len(indices)
    if "config" in data:
        selected["config"] = data["config"].copy()
        selected["config"]["simulation"] = data["config"]["simulation"].copy()
        selected["config"]["simulation"]["n_particles"] = len(indices)
    return selected, indices


def calculate_trajectory_statistics(data: dict) -> dict:
    """Calculate summary statistics (means, stds, extrema) from simulation data."""
    if "positions" not in data:
        raise ValueError("Positions data is required for trajectory statistics")
    
    positions = data["positions"]
    n_steps, n_particles, n_dims = positions.shape

    if n_dims != 2:
        raise ValueError(f"Expected 2D trajectories, got {n_dims} dimensions")
    if n_steps < 1:
        raise ValueError(f"Trajectory must have at least 1 time step, got {n_steps}")
    if n_particles < 1:
        raise ValueError(f"Trajectory must have at least 1 particle, got {n_particles}")

    stats = {
        "n_steps": n_steps,
        "n_particles": n_particles,
        "n_dimensions": n_dims,
    }

    positions_flat = positions.reshape(-1, n_dims)
    stats.update({
        "position_mean": positions_flat.mean(axis=0),
        "position_std": positions_flat.std(axis=0),
        "position_min": positions_flat.min(axis=0),
        "position_max": positions_flat.max(axis=0),
    })

    if "velocities" in data:
        velocities = data["velocities"]
        velocities_flat = velocities.reshape(-1, n_dims)
        stats.update({
            "velocity_mean": velocities_flat.mean(axis=0),
            "velocity_std": velocities_flat.std(axis=0),
            "velocity_magnitude_mean": np.linalg.norm(velocities_flat, axis=1).mean(),
        })

    if "forces" in data:
        forces = data["forces"]
        forces_flat = forces.reshape(-1, n_dims)
        stats.update({
            "force_mean": forces_flat.mean(axis=0),
            "force_std": forces_flat.std(axis=0),
            "force_magnitude_mean": np.linalg.norm(forces_flat, axis=1).mean(),
        })

    if "potential_energy" in data:
        energies = data["potential_energy"]
        energies_flat = energies.reshape(-1)
        stats.update({
            "energy_mean": energies_flat.mean(),
            "energy_std": energies_flat.std(),
            "energy_min": energies_flat.min(),
            "energy_max": energies_flat.max(),
        })

    return stats
