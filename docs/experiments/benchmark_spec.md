# Experiment: Synthetic Smoke Benchmark

## Purpose

The Phase 7 benchmark stitches together the MVP components into one
reproducible smoke run. It trains TEMPORA on synthetic datasets, computes finite
diagnostics, compares simple baselines, saves figures, and generates a Markdown
report.

## Dataset

The default config runs on:

- circle
- torus
- Lorenz
- Rossler

The smoke defaults are intentionally small and CI-friendly.

## Models

- TEMPORA Contractive CTRNN with projected plasticity.
- GRU baseline.
- Unconstrained Neural-ODE-style baseline.
- Reservoir baseline.

## Metrics

- prediction MSE,
- reconstruction MSE,
- contraction margin,
- largest Lyapunov estimate,
- H0/H1 persistence bottleneck distances,
- time-warp invariance score,
- noise robustness score,
- missing-segment robustness score.

## Procedure

Run:

```bash
python scripts/train_synth.py --config configs/benchmark_smoke.yaml
```

The command writes:

```text
outputs/<run_id>/
  config.yaml
  metrics.json
  report.md
  figures/
  checkpoints/
```

`checkpoints/` contains one `*_model.pt` file per dataset in the run. Each
checkpoint stores the `ContractiveCTRNN` state dict, reconstructable model
kwargs, the resolved config, seed, dataset name, and training metrics.

## Expected Failure Modes

- non-finite metrics,
- missing optional TDA dependencies,
- output directory not writable,
- training smoke loop does not produce finite states.

## Reproducibility Notes

`metrics.json` stores the seed, resolved config, artifact paths, git commit hash
when available, dependency versions, and runtime metadata. Generated outputs are
ignored by git.
