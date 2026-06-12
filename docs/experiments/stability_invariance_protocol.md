# Experiment: Stability and Invariance Diagnostics

## Purpose

Phase 5 makes TEMPORA falsifiable with small numerical diagnostics for finite
synthetic trajectories. The metrics are smoke diagnostics, not benchmark
claims.

## Dataset

The evaluator can run on circle, torus, Lorenz, and Rossler trajectories with a
fixed seed and sample count.

## Models

Phase 5 evaluates finite trajectory arrays directly. Later phases can pass
latent trajectories from trained models into the same metrics.

## Metrics

- nearest-neighbor largest Lyapunov estimate,
- time-warp invariance score,
- Gaussian-noise robustness score,
- missing-segment robustness score,
- reconstruction MSE between clean states and observations.

## Procedure

Run:

```bash
python scripts/eval_synth.py --run-id phase5_smoke
```

The command writes:

```text
outputs/<run_id>/
  metrics.json
  figures/
    circle_trajectory.png
    torus_trajectory.png
    lorenz_trajectory.png
    rossler_trajectory.png
```

## Expected Failure Modes

- non-finite trajectory values,
- non-monotonic time grids,
- too-short trajectories for Lyapunov horizons,
- perturbation scores outside finite numeric ranges,
- missing or unwritable output directories.

## Reproducibility Notes

The same seed, dataset list, and sample count should reproduce the same metrics.
Generated figures are artifacts and should not be committed.

