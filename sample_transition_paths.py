#!/usr/bin/env python3
"""Sample weighted MA→MB transition segments with temperature-3 BAOAB dynamics."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from muller_brown import (
    LangevinSimulator,
    MuellerBrownPotential,
    MuellerBrownVisualizer,
)
from muller_brown.constants import MULLER_BROWN_MINIMA
from muller_brown.weighted_ensemble import (
    WeightedPathReservoir,
    sample_weighted_ensemble,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--steps-per-iteration", type=int, default=50)
    parser.add_argument("--walkers-per-bin", type=int, default=32)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--binning", choices=["energy", "arclength"], default="energy")
    parser.add_argument("--source-radius", type=float, default=0.15)
    parser.add_argument("--target-radius", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-paths",
        type=int,
        default=10000,
        help="Maximum saved weighted paths; 0 keeps every event (uses more memory)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Refuse overwrites so independent runs cannot accidentally replace each other.
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new run directory")
    if args.max_paths < 0:
        parser.error("max-paths must be nonnegative")
    potential = MuellerBrownPotential(dtype=torch.float64)
    reservoir = (
        WeightedPathReservoir(args.max_paths, args.seed) if args.max_paths else None
    )
    event_weights = []
    weighted_durations = []

    def collect(path):
        event_weights.append(path.weight)
        weighted_durations.append(
            path.weight * (path.end_step - path.start_step) * args.dt
        )
        reservoir.add(path)

    try:
        simulator = LangevinSimulator(
            potential,
            temperature=args.temperature,
            friction=args.friction,
            dt=args.dt,
        )
        result = sample_weighted_ensemble(
            simulator,
            n_iterations=args.iterations,
            steps_per_iteration=args.steps_per_iteration,
            walkers_per_bin=args.walkers_per_bin,
            n_bins=args.bins,
            binning=args.binning,
            source_radius=args.source_radius,
            target_radius=args.target_radius,
            seed=args.seed,
            on_transition=collect if reservoir is not None else None,
        )
    except (ValueError, ZeroDivisionError) as error:
        parser.error(str(error))

    args.output_dir.mkdir(parents=True)
    paths = reservoir.paths if reservoir is not None else result.paths
    path_weights = (
        reservoir.weights
        if reservoir is not None
        else np.array([path.weight for path in paths])
    )
    original_weights = (
        np.array(event_weights) if reservoir is not None else path_weights
    )
    event_ids = (
        reservoir.event_ids if reservoir is not None else list(range(len(paths)))
    )
    metadata = vars(args).copy()
    metadata["output_dir"] = str(args.output_dir)
    metadata.update(
        {
            "method": "weighted_ensemble_systematic_absorbing_B",
            "initial_position": MULLER_BROWN_MINIMA[0],
            "initial_velocity": [0.0, 0.0],
            "source_center": MULLER_BROWN_MINIMA[0],
            "target_center": MULLER_BROWN_MINIMA[1],
            "mass": 1.0,
            "kB": 1.0,
            "integrator": "BAOAB",
            "dtype": "float64",
            "device": "cpu",
            "saved_every": 1,
            "completed_iterations": len(result.history),
            "horizon": float(result.history[-1, 0] * args.dt),
            "n_transitions": len(original_weights),
            "n_saved_paths": len(paths),
            "output_resampling": "weighted reservoir with replacement"
            if reservoir is not None and reservoir.resampled
            else "none",
            "absorbed_weight": float(result.history[-1, 2]),
            "ensemble": "finite-horizon first passages from fixed MA, zero velocity",
            "path_definition": "last frame inside A through first frame inside B",
            "weight_definition": "absolute first-passage probability contribution after optional output resampling",
            "samples_are_correlated": True,
        }
    )
    # This weight-concentration diagnostic ignores genealogy; it is NOT an ESS
    # of independent paths and must not be used for uncertainty estimates.
    concentration = (
        (original_weights.sum() / np.linalg.norm(original_weights)) ** 2
        if len(original_weights)
        else 0.0
    )
    metadata["weight_concentration_count_ignoring_correlations"] = float(concentration)
    metadata["weighted_mean_duration_before_output_resampling"] = (
        float(np.sum(weighted_durations) / original_weights.sum())
        if reservoir is not None and len(original_weights)
        else float(
            sum(
                path.weight * (path.end_step - path.start_step) * args.dt
                for path in paths
            )
            / path_weights.sum()
        )
        if paths
        else None
    )
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.savetxt(
        args.output_dir / "sampling_history.csv",
        result.history,
        delimiter=",",
        header="step,active_weight,absorbed_weight,walkers,occupied_bins,hits",
        comments="",
    )
    with h5py.File(args.output_dir / "transition_paths.h5", "w") as file:
        file.attrs["metadata_json"] = json.dumps(metadata)
        file.create_dataset("weights", data=path_weights)
        paths_group = file.create_group("paths")
        for index, path in enumerate(paths):
            group = paths_group.create_group(f"{index:06d}")
            group.create_dataset(
                "positions", data=path.positions, compression="gzip", shuffle=True
            )
            group.create_dataset(
                "velocities", data=path.velocities, compression="gzip", shuffle=True
            )
            group.attrs.update(
                {
                    "weight": path_weights[index],
                    "original_event_weight": path.weight,
                    "event_id": event_ids[index],
                    "start_step": path.start_step,
                    "end_step": path.end_step,
                    "dt": args.dt,
                }
            )

    if paths:
        visualizer = MuellerBrownVisualizer(potential)
        fig, ax = visualizer.plot_potential_surface()
        # Display a reproducible weighted subset; leave all paths in the HDF5.
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(
            len(paths),
            size=min(100, len(paths)),
            replace=True,
            p=path_weights / path_weights.sum(),
        )
        for index in selected:
            x = paths[index].positions
            ax.plot(x[:, 0], x[:, 1], color="white", alpha=0.2, linewidth=0.7)
        for center, radius in zip(
            MULLER_BROWN_MINIMA[:2], [args.source_radius, args.target_radius]
        ):
            ax.add_patch(
                plt.Circle(center, radius, fill=False, color="red", linewidth=1.5)
            )
        ax.set_title(f"Weighted MA→MB transitions, T={args.temperature:g}")
        fig.savefig(
            args.output_dir / "transition_paths.png", dpi=180, bbox_inches="tight"
        )
        plt.close(fig)
    print(
        f"Observed {len(original_weights)} transition events at T={args.temperature:g}; saved {len(paths)} weighted segments"
    )
    print(
        f"Estimated finite-horizon first-passage probability: {metadata['absorbed_weight']:.6g}"
    )
    print(f"Saved to {args.output_dir}")
    print(
        "Use path weights; clones are correlated. Validate across seeds and timesteps."
    )


if __name__ == "__main__":
    main()
