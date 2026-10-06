#!/usr/bin/env python3
"""
Simple example script demonstrating the Müller-Brown simulation package.

This script samples 100 independent paths from a common initial state
and visualizes their trajectories and spatial distributions.
"""

import matplotlib.pyplot as plt
import torch

from extract_paths import extract_mb_paths

from muller_brown import (
    MuellerBrownPotential,
    LangevinSimulator,
    MuellerBrownVisualizer,
    save_simulation_data,
)


def main():
    """Sample and visualize an ensemble of 100 Langevin paths."""
    print("=== Müller-Brown Simulation Example ===")
    torch.manual_seed(42)
    
    # 1. Create the potential
    print("Setting up Müller-Brown potential...")
    potential = MuellerBrownPotential(dtype=torch.float64)
    
    # Show potential features
    print(f"Potential minima: {potential.get_minima()}")
    print(f"Saddle points: {potential.get_saddle_points()}")
    
    # 2. Set up the simulator
    print("Initializing Langevin simulator...")
    simulator = LangevinSimulator(
        potential=potential,
        temperature=9.0,  # Temperature for thermostat
        friction=0.5,      # Friction coefficient
        dt=0.01           # Time step
    )
    
    # 3. Sample independent paths in one vectorized simulation
    n_paths = 100
    print(f"Sampling {n_paths} paths...")
    initial_positions = torch.tensor(
        [[-0.55822363, 1.44172584]], dtype=torch.float64
    ).repeat(n_paths, 1)
    initial_velocities = torch.zeros_like(initial_positions)
    
    results = simulator.simulate(
        initial_positions=initial_positions,
        initial_velocities=initial_velocities,
        n_steps=1_000_000,
        save_every=100
    )
    
    paths = results["positions"]  # (n_saved_times, n_paths, 2)
    print(
        f"Simulation completed! Generated {paths.shape[1]} paths "
        f"with {paths.shape[0]} saved frames each"
    )
    
    # 4. Save the data
    print("Saving simulation data...")
    save_path = save_simulation_data(results, create_artifact_dir=True)
    print(f"Data saved to: {save_path}")

    # Extract full paths whose final positions are much closer to MB than MA
    extract_mb_paths(
        results, save_path.parent / "mb_endpoint_selection", max_distance_ratio=0.5
    )
    
    # 5. Create visualizations
    print("Creating visualizations...")
    visualizer = MuellerBrownVisualizer(potential)
    
    # Plot potential surface
    fig, ax = visualizer.plot_potential_surface()
    fig.savefig(save_path.parent / "potential_surface.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Overlay all paths on the potential to show the ensemble's shape
    fig, ax = visualizer.plot_potential_surface()
    ax.plot(
        paths[:, :, 0], paths[:, :, 1],
        color="white", alpha=0.15, linewidth=0.6,
    )
    ax.set_title(f"Ensemble of {n_paths} Paths on the Potential Surface")
    fig.savefig(save_path.parent / "ensemble_paths.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Pool all paths and saved times to show time-averaged occupancy
    fig, axes = visualizer.plot_position_distributions(results)
    fig.savefig(save_path.parent / "ensemble_occupancy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Use only the final frame to show the spatial ensemble at the final time
    fig, axes = visualizer.plot_position_distributions({"positions": paths[-1:]})
    fig.savefig(save_path.parent / "ensemble_endpoints.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Plot trajectory on potential surface
    fig, ax = visualizer.plot_trajectory_on_potential(results, sample_idx=0)
    fig.savefig(save_path.parent / "trajectory_on_potential.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Plot position time series
    fig, axes = visualizer.plot_position_time_series(results, sample_idx=0)
    fig.savefig(save_path.parent / "position_time_series.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Plots saved to: {save_path.parent}")
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
