# Experiment: Synthetic Trajectory Data

## Purpose

TEMPORA uses synthetic trajectories as controlled ground truth for early
stability, invariance, and topology experiments.

## Dataset

The Phase 1 data API exposes:

- `generate_circle`: 2D circular trajectory.
- `generate_torus`: 3D trajectory on a torus surface.
- `generate_lorenz`: 3D Lorenz ODE trajectory.
- `generate_rossler`: 3D Rossler ODE trajectory.

All generators return `TemporalDataset` with:

- `times`: shape `(n_steps,)`
- `observations`: shape `(n_steps, observation_dim)`
- `clean_states`: shape `(n_steps, state_dim)`
- `metadata`: generator parameters and shape metadata

Example generator configs are available in `configs/synth_circle.yaml`,
`configs/synth_torus.yaml`, `configs/synth_lorenz.yaml`, and
`configs/synth_rossler.yaml`.

## Perturbations

- `time_warp` changes the time grid while preserving sampled trajectory values.
- `add_gaussian_noise` perturbs observations while preserving clean states.
- `mask_missing_segment` replaces one contiguous observation segment with a
  finite mask value while preserving clean states.

## Procedure

1. Choose a generator and fixed seed.
2. Generate a finite trajectory.
3. Apply at most one perturbation at a time for isolated tests.
4. Validate shapes, finite values, metadata, and deterministic seeds.

## Expected Failure Modes

- Invalid parameters such as non-positive duration or degenerate torus radii.
- Non-finite observations or states.
- Non-monotonic time grids.
- ODE integration failures for unstable parameter choices.

## Reproducibility Notes

The same generator parameters and seed must produce the same arrays. The
synthetic data tests are deliberately small enough for CI and do not constitute
benchmark evidence.
