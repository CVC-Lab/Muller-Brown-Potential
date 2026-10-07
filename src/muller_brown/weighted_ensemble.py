"""Weighted first-passage paths under the original BAOAB dynamics.

Walkers are resampled within progress bins with probabilities proportional to
their weights. Clones retain positions AND velocities and receive independent
thermostat noise. B is absorbing: its incoming weights estimate the finite-time
first-passage probability, not a stationary rate. No force bias or heating is used.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import trange

from muller_brown.constants import MULLER_BROWN_MINIMA
from muller_brown.simulation import LangevinSimulator


@dataclass
class TransitionPath:
    positions: np.ndarray
    velocities: np.ndarray
    weight: float
    start_step: int
    end_step: int


class WeightedPathReservoir:
    """Bound memory by sampling the observed weighted path pool with replacement.

    Below capacity, keep all original paths and weights. Above capacity, each
    slot is a draw proportional to the accumulated event weights, with output
    weight total_weight/capacity. This adds sampling variance, never changes
    dynamics, and does not turn correlated WE paths into independent data.
    """

    def __init__(self, capacity: int, seed: int):
        if capacity < 1:
            raise ValueError("reservoir capacity must be positive")
        self.capacity = capacity
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.paths: list[TransitionPath] = []
        self.event_ids: list[int] = []
        self.total_weight = 0.0
        self.n_events = 0
        self.resampled = False

    def add(self, path: TransitionPath) -> None:
        if not math.isfinite(path.weight) or path.weight <= 0:
            raise ValueError("path weight must be finite and positive")
        event_id = self.n_events
        self.n_events += 1
        if len(self.paths) < self.capacity:
            self.paths.append(path)
            self.event_ids.append(event_id)
        else:
            if not self.resampled:
                selected = self.rng.choice(
                    self.capacity,
                    self.capacity,
                    replace=True,
                    p=np.array([p.weight for p in self.paths]) / self.total_weight,
                )
                self.paths = [self.paths[index] for index in selected]
                self.event_ids = [self.event_ids[index] for index in selected]
                self.resampled = True
            # Binomial count + uniform subset is equivalent to independently
            # replacing each slot, without drawing capacity randoms per event.
            probability = path.weight / (self.total_weight + path.weight)
            count = self.rng.binomial(self.capacity, probability)
            for index in self.rng.choice(self.capacity, count, replace=False):
                self.paths[index] = path
                self.event_ids[index] = event_id
        self.total_weight += path.weight

    @property
    def weights(self) -> np.ndarray:
        if self.resampled:
            return np.full(self.capacity, self.total_weight / self.capacity)
        return np.array([path.weight for path in self.paths])


@dataclass
class WeightedEnsembleResult:
    paths: list[TransitionPath]
    # Columns: step, active_weight, absorbed_weight, walkers, occupied_bins, hits.
    history: np.ndarray


@dataclass
class _History:
    parent: "_History | None"
    positions: np.ndarray
    velocities: np.ndarray
    start_step: int

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        nodes = []
        node = self
        while node is not None:
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return (
            np.concatenate([node.positions for node in nodes]),
            np.concatenate([node.velocities for node in nodes]),
        )


def resample_bins(
    weights: np.ndarray,
    bins: np.ndarray,
    walkers_per_bin: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected parent indices and weights; conserve each bin's mass.

    Systematic resampling is unbiased in path space and preserves an already
    balanced bin instead of needlessly duplicating and discarding walkers.
    Repeated seeds, rather than treating correlated clones as independent, are
    necessary for uncertainty estimates.
    """
    if walkers_per_bin < 1:
        raise ValueError("walkers_per_bin must be positive")
    if weights.ndim != 1 or bins.shape != weights.shape:
        raise ValueError("weights and bins must be matching one-dimensional arrays")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("weights must be finite and positive")
    parents, new_weights = [], []
    for bin_index in np.unique(bins):
        members = np.flatnonzero(bins == bin_index)
        total = weights[members].sum()
        if len(members) == walkers_per_bin and np.all(
            weights[members] == weights[members[0]]
        ):
            selected = members
        else:
            probabilities = np.cumsum(weights[members] / total)
            probabilities[-1] = 1.0
            points = (rng.random() + np.arange(walkers_per_bin)) / walkers_per_bin
            selected = members[np.searchsorted(probabilities, points, side="right")]
        parents.extend(selected)
        new_weights.extend([total / walkers_per_bin] * walkers_per_bin)
    return np.asarray(parents, dtype=int), np.asarray(new_weights, dtype=float)


def path_progress(positions: np.ndarray, guide: np.ndarray) -> np.ndarray:
    """Normalized arclength of the nearest projection onto a guide polyline.

    The guide directs allocation of computation, never the dynamics. Walkers
    can explore off-guide channels and return to earlier bins.
    """
    vectors = np.diff(guide, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    offsets = positions[:, None, :] - guide[:-1]
    fraction = np.clip(
        np.sum(offsets * vectors, axis=2) / lengths**2,
        0,
        1,
    )
    projections = guide[:-1] + fraction[:, :, None] * vectors
    closest = np.argmin(
        np.sum((positions[:, None, :] - projections) ** 2, axis=2), axis=1
    )
    distance = np.r_[0.0, np.cumsum(lengths)][closest]
    distance += fraction[np.arange(len(positions)), closest] * lengths[closest]
    return distance / lengths.sum()


def sample_weighted_ensemble(
    simulator: LangevinSimulator,
    *,
    n_iterations: int = 2000,
    steps_per_iteration: int = 50,
    walkers_per_bin: int = 32,
    n_bins: int = 40,
    binning: str = "energy",
    source_radius: float = 0.15,
    target_radius: float = 0.15,
    seed: int = 42,
    progress: bool = True,
    on_transition: Callable[[TransitionPath], None] | None = None,
) -> WeightedEnsembleResult:
    """Sample A→B first passages from MA with zero initial velocity.

    This matches the fixed initial-state ensemble in example.py (up to rounded
    minimum coordinates). Every integration step is checked for basin visits
    and retained on reactive segments, from the last A frame to the first B
    frame. Results are weighted, correlated samples of a finite-horizon process;
    they are not automatically an equilibrated transition-path ensemble.
    Supply on_transition to consume paths as they arrive instead of retaining
    them all in result.paths (useful for convergence studies or streaming I/O).
    """
    for name, value in (
        ("n_iterations", n_iterations),
        ("steps_per_iteration", steps_per_iteration),
        ("walkers_per_bin", walkers_per_bin),
        ("n_bins", n_bins),
    ):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("temperature", "friction", "mass", "dt"):
        value = getattr(simulator, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if str(simulator.device) != "cpu" or simulator.dtype != torch.float64:
        raise ValueError("weighted ensemble currently requires CPU float64 dynamics")
    if binning not in ("energy", "arclength"):
        raise ValueError("binning must be energy or arclength")
    source, target, intermediate = np.asarray(MULLER_BROWN_MINIMA)
    for name, value in (
        ("source_radius", source_radius),
        ("target_radius", target_radius),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if source_radius + target_radius >= np.linalg.norm(target - source):
        raise ValueError("source and target basins must not overlap")
    saddles = np.asarray(simulator.potential.get_saddle_points())
    guide = np.array([source, saddles[0], intermediate, saddles[1], target])
    reference_energies = (
        simulator.potential(torch.tensor(guide, dtype=simulator.dtype)).detach().numpy()
    )
    minimum_energy = reference_energies[0]
    energy_range = reference_energies[1] - minimum_energy
    if not np.isfinite(energy_range) or energy_range <= 0:
        raise ValueError("the source-to-saddle energy barrier must be positive")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    positions = torch.tensor(source, dtype=simulator.dtype).repeat(walkers_per_bin, 1)
    velocities = torch.zeros_like(positions)
    forces = simulator.potential.force(positions)
    weights = np.full(walkers_per_bin, 1 / walkers_per_bin)
    tails: list[_History | None] = [None] * walkers_per_bin
    paths, history = [], []
    absorbed_weight = 0.0
    source_tensor = torch.tensor(source, dtype=simulator.dtype)
    target_tensor = torch.tensor(target, dtype=simulator.dtype)

    for iteration in trange(
        n_iterations, disable=not progress, desc="Weighted ensemble"
    ):
        count = len(weights)
        if count == 0:
            break
        frames = np.empty((steps_per_iteration + 1, count, 2))
        velocity_frames = np.empty_like(frames)
        frames[0], velocity_frames[0] = positions.numpy(), velocities.numpy()
        first_hit = np.full(count, -1, dtype=int)
        last_source = np.where(
            np.linalg.norm(frames[0] - source, axis=1) <= source_radius, 0, -1
        )

        for step in range(1, steps_per_iteration + 1):
            positions, velocities, forces = simulator._baoab_step(
                positions, velocities, forces
            )
            frames[step], velocity_frames[step] = positions.numpy(), velocities.numpy()
            unfinished = first_hit < 0
            if (
                not np.isfinite(frames[step, unfinished]).all()
                or not np.isfinite(velocity_frames[step, unfinished]).all()
            ):
                raise ValueError("nonfinite dynamics; reduce the integration timestep")
            in_source = (
                torch.linalg.vector_norm(positions - source_tensor, dim=1).numpy()
                <= source_radius
            )
            in_target = (
                torch.linalg.vector_norm(positions - target_tensor, dim=1).numpy()
                <= target_radius
            )
            last_source[unfinished & in_source] = step
            first_hit[unfinished & in_target] = step

        next_tails = []
        for index in range(count):
            stop = first_hit[index] if first_hit[index] >= 0 else steps_per_iteration
            last = last_source[index]
            parent = None if last >= 0 else tails[index]
            if last < 0 and parent is None:
                raise RuntimeError("lost source ancestry")
            start = last if last >= 0 else 1
            start_step = (
                iteration * steps_per_iteration + last
                if last >= 0
                else parent.start_step
            )
            tail = _History(
                parent,
                frames[start : stop + 1, index].copy(),
                velocity_frames[start : stop + 1, index].copy(),
                start_step,
            )
            if first_hit[index] >= 0:
                x, v = tail.arrays()
                path = TransitionPath(
                    x,
                    v,
                    float(weights[index]),
                    start_step,
                    iteration * steps_per_iteration + stop,
                )
                if on_transition is None:
                    paths.append(path)
                else:
                    on_transition(path)
            else:
                next_tails.append(tail)

        survivors = np.flatnonzero(first_hit < 0)
        absorbed_weight += float(weights[first_hit >= 0].sum())
        positions = positions[survivors]
        velocities = velocities[survivors]
        forces = forces[survivors]
        weights = weights[survivors]
        if binning == "energy":
            # Low-friction escape requires accumulating energy, not merely moving
            # uphill. Separate nearest-minimum regions protect walkers that have
            # crossed into the intermediate well as they cool down.
            energy = simulator.potential(positions).detach().numpy()
            energy += 0.5 * simulator.mass * np.sum(velocities.numpy() ** 2, axis=1)
            energy_bin = np.clip(
                ((energy - minimum_energy) / energy_range * n_bins).astype(int),
                0,
                n_bins - 1,
            )
            basin = np.argmin(
                np.sum(
                    (positions.numpy()[:, None, :] - np.asarray(MULLER_BROWN_MINIMA))
                    ** 2,
                    axis=2,
                ),
                axis=1,
            )
            bins = basin * n_bins + energy_bin
        else:
            bins = np.minimum(
                (path_progress(positions.numpy(), guide) * n_bins).astype(int),
                n_bins - 1,
            )
        parents, weights = resample_bins(weights, bins, walkers_per_bin, rng)
        positions, velocities, forces = (
            positions[parents],
            velocities[parents],
            forces[parents],
        )
        tails = [next_tails[index] for index in parents]
        active_weight = float(weights.sum())
        if not math.isclose(
            active_weight + absorbed_weight, 1.0, rel_tol=0.0, abs_tol=1e-11
        ):
            raise RuntimeError("weighted ensemble probability mass was not conserved")
        history.append(
            [
                (iteration + 1) * steps_per_iteration,
                active_weight,
                absorbed_weight,
                len(weights),
                len(np.unique(bins)),
                int(np.sum(first_hit >= 0)),
            ]
        )
    return WeightedEnsembleResult(paths, np.asarray(history, dtype=float))
