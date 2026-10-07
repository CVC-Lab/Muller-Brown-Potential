# Müller-Brown Potential Simulation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A PyTorch implementation of the Müller-Brown potential energy surface with BAOAB Langevin dynamics.

## Overview

This package implements the Müller-Brown potential, a 2D model potential energy surface widely used for testing molecular dynamics algorithms and studying rare events in chemical systems.

See [Implementing the Müller-Brown Potential in PyTorch](https://hunterheidenreich.com/posts/muller-brown-in-pytorch/) for a detailed blog post explaining the implementation and usage.

## The Müller-Brown Potential

The Müller-Brown potential is defined as a sum of four Gaussian-like terms:

$$V(x,y) = \sum_{i=1}^{4} A_i \exp\left(a_i(x-x_{0i})^2 + b_i(x-x_{0i})(y-y_{0i}) + c_i(y-y_{0i})^2\right)$$

This creates a complex energy landscape with:

- **3 local minima** at different energy levels
- **2 saddle points** connecting the minima
- **Energy barriers** that control transition rates between states

This makes it ideal for studying rare event transitions, temperature effects on barrier crossing, and molecular dynamics algorithm performance.

## Installation

Install dependencies using uv:

```bash
uv sync
```

## Quick Start

The easiest way to get started is with the demo mode:

```bash
uv run python main.py --mode demo
```

This shows the potential's key features: 3 minima, 2 saddle points, and demonstrates energy/force calculations.

### Command Line Interface

```bash
# Quick demo - see potential features
uv run python main.py --mode demo

# Single simulation with visualization
uv run python main.py --mode single --n-particles 1 --n-steps 50000 --save-plots

# Single simulation with animated trajectory
uv run python main.py --mode single --n-particles 1 --n-steps 50000 --save-plots --save-animation

# Multiple simulations for statistics
uv run python main.py --mode batch --n-simulations 10 --n-particles 2 --save-plots

# List available artifact directories
uv run python main.py --mode list

# Regenerate plots from existing data
uv run python main.py --mode plot --artifact-dir artifacts/20250824_184610
```

### Programmatic Usage

```python
import torch
from muller_brown import MuellerBrownPotential, LangevinSimulator, MuellerBrownVisualizer

# Create the potential
potential = MuellerBrownPotential(dtype=torch.float64)

# Set up the simulator
simulator = LangevinSimulator(
    potential=potential,
    temperature=15.0,
    friction=1.0,
    dt=0.01
)

# Run simulation
results = simulator.simulate(
    initial_positions=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
    n_steps=10000,
    save_every=10
)

# Visualize results
visualizer = MuellerBrownVisualizer(potential)
fig, ax = visualizer.plot_potential_surface()
```

Run the example with:

```bash
uv run python example.py
```

The example samples 100 paths and extracts full paths whose final saved
positions satisfy `distance_to_MB <= 0.5 * distance_to_MA`, using Euclidean
distance. Extracted observables, original path indices, an endpoint-distance
CSV, and plots are saved in `mb_endpoint_selection` inside the artifact folder.

To extract paths from an existing ensemble without rerunning the simulation:

```bash
uv run python extract_paths.py artifacts/<run>/trajectory_<timestamp>.h5
```

Use `--distance-ratio 0.25` for a stricter cutoff or `--output-dir <path>` to
choose a separate output folder. Selection uses the final saved position;
visiting MB earlier in a path does not qualify it.

### Rare transitions at temperature 3

The barrier out of MA is approximately 106 in this potential's energy units.
With `kB=1` and `T=3`, this is about 35 thermal energy units, so ordinary
MA-started rollouts are unlikely to yield transitions. The Boltzmann factor
`exp(-106/3)` is an activation factor, not a crossing probability or rate.

Use weighted-ensemble sampling to allocate more walkers along the transition
region while retaining the original temperature-3 BAOAB dynamics:

```bash
uv run python sample_transition_paths.py --temperature 3 --seed 42 \
    --output-dir artifacts/we_temp3_seed42
```

The sampler checks both basins at **every integration step** and saves each
reactive segment from its last frame inside A to its first frame inside B.
A and B are radius-0.15 disks around MA and MB; configure their radii with
`--source-radius` and `--target-radius`. By default, resampling uses total-energy
bins, separately within each nearest-minimum region. This protects the rare
energetic walkers needed for low-friction escape and the walkers entering the
intermediate well. Alternatively, `--binning arclength` uses bins along the
minima/saddle polyline. Bins affect allocation, never the forces.
Systematic resampling preserves balanced bins to reduce loss of diversity.
Positions and velocities are copied together, then clones receive independent
thermostat noise. B is absorbing, without recycling.

The initial state is the rounded MA minimum with zero velocity, matching the
example's initial-state convention. This samples **finite-horizon first
passages from that initial state**, not automatically the stationary
equilibrium transition-path ensemble. The default run uses 2,000 resampling
iterations, 50 integration steps per iteration, `dt=0.002`,
and 32 walkers per occupied bin (40 energy bins per nearest-minimum region).
Its horizon is 200 simulation time units.
By default, at most 10,000 paths are retained in a weighted reservoir. If more
events occur, the output samples their weighted empirical distribution with
replacement and redistributes the total event weight equally among retained
records. This bounds storage and adds sampling variance. `--max-paths 0` keeps
all original events and weights, with larger memory and storage requirements.

Outputs include:

- `transition_paths.h5`: variable-length position/velocity arrays under
  `paths/<index>`, per-path timestep and start/end step, and absolute probability
  contributions in `weights` (also attached to each path).
- `sampling_history.csv`: active/absorbed probability mass and walker counts.
- `summary.json`: dynamics, basin definitions, initial state, and sampling metadata.
- `transition_paths.png`: a weighted sample of the recorded crossings.

**Use the path weights.** For a path statistic `f`, estimate its conditional
mean as `sum(w * f) / sum(w)`. Do not treat the raw path count as a transition
probability or the cloned paths as independent samples. The absorbed weight
estimates first-passage probability within the specified horizon, not a
stationary transition rate. Establish convergence across independent seeds,
bin/iteration settings, basin definitions, and smaller timesteps before using
the output as a ground-truth reference. The saved weight-concentration count
ignores genealogical correlations and is not an independent-sample count.
After output resampling, use `weights`/the per-path `weight`, rather than
`original_event_weight`. Repeated `event_id` values identify repeated selections
of one event; different event IDs can still share ancestral trajectory segments.
For memory-bounded analysis, the library sampler also accepts an `on_transition`
callback, which consumes paths as they arrive instead of keeping them all in RAM.

Method reference: [Zhang, Jasnow, and Zuckerman, weighted-ensemble path sampling](https://arxiv.org/abs/0810.1963).

### Managing Artifacts

The repository includes a utility to manage simulation artifacts:

```bash
# List all artifact directories
uv run python manage_artifacts.py list

# Clean artifacts older than 7 days (dry run)
uv run python manage_artifacts.py clean

# Actually delete old artifacts
uv run python manage_artifacts.py clean --delete
```

## Testing and Validation

The test suite uses `pytest`:

```bash
uv run --extra dev pytest                       # full suite
uv run --extra dev pytest -m "not statistical"  # fast deterministic tests only
```

It has two tiers:

- **Deterministic** tests check exact properties: the potential's critical points (Hessian signatures), forces against autograd, energy conservation in the frictionless (velocity-Verlet) limit, seed reproducibility, HDF5 round-trips, and array shape/dtype invariants.
- **Statistical** tests (marked `statistical`) check that the thermostat samples the canonical distribution: equipartition, harmonic-oscillator distributions, the Müller-Brown mean energy against a grid-integrated Boltzmann average, the free-particle velocity autocorrelation, and the timestep-convergence order. They are seeded, with tolerances set above the sampling error.

Two standalone scripts produce validation plots:

```bash
uv run python verify_langevin.py     # sampled vs analytic harmonic-oscillator distributions
uv run python convergence_study.py   # kinetic-temperature bias vs timestep
```

> **Note on accuracy.** BAOAB samples configurational averages accurately (exactly for a harmonic oscillator), but the kinetic temperature carries an O(dt²) bias that grows with curvature — about 4% on the Müller-Brown surface at dt=0.01. It shrinks as the timestep is reduced; `convergence_study.py` quantifies this.

## Data Format

Simulations save all observables in HDF5 format:

- **Positions**: (n_steps, n_particles, 2) - x,y coordinates over time
- **Velocities**: (n_steps, n_particles, 2) - velocity components over time
- **Forces**: (n_steps, n_particles, 2) - force components over time
- **Potential Energy**: (n_steps, n_particles) - potential energy over time
- **Metadata**: Temperature, friction, timestep, and simulation parameters

## Project Structure

```
muller_brown/
├── src/muller_brown/        # Core package
│   ├── potential.py         # Müller-Brown potential (torch.compile force)
│   ├── simulation.py        # Langevin dynamics simulator
│   ├── visualization.py     # Plotting and animation suite
│   ├── io.py               # HDF5 data I/O operations
│   ├── data.py             # Data processing utilities
│   ├── analysis.py         # Statistical analysis functions
│   ├── config.py           # Experiment configuration
│   └── constants.py        # Physical and computational constants
├── tests/                  # pytest suite (deterministic + statistical)
├── main.py                 # Command-line interface
├── benchmark.py            # Performance benchmarking
├── example.py              # Simple usage example
├── verify_langevin.py      # BAOAB sampling vs analytic harmonic oscillator
├── convergence_study.py    # Timestep-convergence study
├── manage_artifacts.py     # Artifact management utility
├── pyproject.toml         # Project dependencies and metadata
├── LICENSE                # MIT License
└── README.md              # This file
```

## Key Classes

### `MuellerBrownPotential`

- `torch.compile`-accelerated force evaluation
- Automatic gradient computation for forces
- Built-in critical point locations (3 minima, 2 saddle points)

### `LangevinSimulator`

- BAOAB integration (Leimkuhler & Matthews, 2013) for accurate canonical sampling
- Tracks all observables (positions, velocities, forces, energies)
- Configurable temperature, friction, and time step

### `MuellerBrownVisualizer`

- Distribution plots for every observable (position, velocity, force, energy)
- Time-series plots for all observables
- Log-scale visualization for rare events
- Automatic critical point annotation

## Performance

Design choices that affect performance:

- **Compiled forces**: `torch.compile` (shape-dynamic) on the force path
- **Device-agnostic**: runs on CPU or CUDA; the included benchmark measures CPU
- **Storage**: HDF5 with gzip compression for large trajectories
- **Numerical precision**: double precision (`float64`) by default
- **Memory**: configurable save frequency for long simulations

Each step is vectorized over particles, so for small particle counts the wall
time is dominated by fixed per-step overhead (the force call and noise draw): a
1-particle step and a 100-particle step take roughly the same time (~40 vs
~50 µs). For sampling throughput, prefer an ensemble (`--n-particles 100+`) over
many separate single-particle runs — that is ~100× the sampling for ~25% more
wall time.

## Tips for Research Use

- **Temperature effects**: Lower temperatures (T ~ 5-10) emphasize rare events; higher temperatures (T ~ 20-30) increase barrier crossing
- **Equilibration**: Use `--n-transient` to remove initial transient behavior
- **Statistics**: Run batch simulations for ensemble averages and error estimates
- **Long trajectories**: For rare events, use 10⁶ steps or more with appropriate save frequency
- **Multiple particles**: Study collective behavior with `--n-particles > 1`

## References

**Original Müller-Brown Potential:**

- Müller, K., & Brown, L. D. (1979). Location of saddle points and minimum energy paths by a constrained simplex optimization procedure. _Theoretica Chimica Acta_, 53, 75-93.

**Implementation Based On:**

- [LED-Molecular Repository](https://github.com/cselab/LED-Molecular) - Original implementation of the Müller-Brown potential

**Related Work:**

- Vlachas, P. R., Zavadlav, J., Praprotnik, M., & Koumoutsakos, P. (2022). Accelerated simulations of molecular systems through learning of effective dynamics. _Journal of Chemical Theory and Computation_, 18(1), 538-549.
- Vlachas, P. R., Arampatzis, G., Uhler, C., & Koumoutsakos, P. (2022). Multiscale simulations of complex systems by learning their effective dynamics. _Nature Machine Intelligence_.

## License

This project is available under the MIT License.
