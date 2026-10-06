#!/usr/bin/env python3
"""Extract full ensemble paths whose final positions are much closer to MB than MA."""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from muller_brown import (
    MuellerBrownPotential,
    MuellerBrownVisualizer,
    load_simulation_data,
    save_simulation_data,
    select_paths_by_endpoint,
)
from muller_brown.constants import MULLER_BROWN_MINIMA


def extract_mb_paths(
    data: dict, output_dir: str | Path, max_distance_ratio: float = 0.5
) -> Path | None:
    """Save qualifying paths, their original indices, an endpoint CSV, and plots.

    MA and MB are the first two minima, matching the potential plot labels.
    Return the selected HDF5 file path, or None when no endpoints qualify.
    """
    ma, mb = MULLER_BROWN_MINIMA[:2]
    selected, indices = select_paths_by_endpoint(
        data, target=mb, reference=ma, max_distance_ratio=max_distance_ratio
    )
    endpoints = data["positions"][-1]
    distance_ma = np.linalg.norm(endpoints - ma, axis=1)
    distance_mb = np.linalg.norm(endpoints - mb, axis=1)
    selected_mask = np.zeros(len(endpoints), dtype=bool)
    selected_mask[indices] = True

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_dir / "endpoint_selection.csv",
        np.column_stack((
            np.arange(len(endpoints)), endpoints,
            distance_ma, distance_mb, selected_mask,
        )),
        delimiter=",",
        header="path_index,endpoint_x,endpoint_y,distance_to_MA,distance_to_MB,selected",
        comments="",
        fmt=["%d", "%.17g", "%.17g", "%.17g", "%.17g", "%d"],
    )
    print(
        f"Selected {len(indices)} of {len(endpoints)} paths with "
        f"distance to MB <= {max_distance_ratio:g} * distance to MA"
    )
    print(f"Original path indices (zero-based): {indices.tolist()}")
    if len(indices) == 0:
        # Do not leave an earlier selection's data or plots beside the new CSV.
        for name in ("selected_paths.h5", "selected_paths.png", "selected_endpoints.png"):
            (output_dir / name).unlink(missing_ok=True)
        print(f"No paths qualify; endpoint distances saved to {output_dir}")
        return None

    save_path = save_simulation_data(
        selected, output_dir=output_dir, filename="selected_paths.h5"
    )
    # Preserve the mapping from extracted columns back to the original ensemble.
    with h5py.File(save_path, "a") as f:
        selection = f.create_group("selection")
        selection.create_dataset("original_path_indices", data=indices)
        selection.create_dataset("distance_to_MA", data=distance_ma[indices])
        selection.create_dataset("distance_to_MB", data=distance_mb[indices])
        selection.attrs["MA"] = ma
        selection.attrs["MB"] = mb
        selection.attrs["max_distance_ratio"] = max_distance_ratio
        selection.attrs["criterion"] = "distance_to_MB <= max_distance_ratio * distance_to_MA"

    visualizer = MuellerBrownVisualizer(MuellerBrownPotential())
    paths = selected["positions"]
    fig, ax = visualizer.plot_potential_surface()
    ax.plot(paths[:, :, 0], paths[:, :, 1], alpha=0.5, linewidth=0.7)
    ax.scatter(
        paths[-1, :, 0], paths[-1, :, 1],
        color="red", edgecolors="black", label="Selected endpoints", zorder=12,
    )
    ax.set_title(f"{len(indices)} Paths Ending Much Closer to MB Than MA")
    ax.legend()
    fig.savefig(output_dir / "selected_paths.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = visualizer.plot_position_distributions({"positions": paths[-1:]})
    fig.savefig(output_dir / "selected_endpoints.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Extracted data and plots saved to: {output_dir}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", type=Path, help="Saved ensemble HDF5 file")
    parser.add_argument(
        "--distance-ratio", type=float, default=0.5,
        help="Require distance to MB <= this fraction of distance to MA (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory (default: mb_endpoint_selection beside the input file)",
    )
    args = parser.parse_args()
    try:
        extract_mb_paths(
            load_simulation_data(args.data_file),
            args.output_dir or args.data_file.parent / "mb_endpoint_selection",
            max_distance_ratio=args.distance_ratio,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
