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
The default smoke config is `configs/benchmark_smoke.yaml`; standalone dataset
and model examples live in the other `configs/*.yaml` files.

## Models

- TEMPORA Contractive CTRNN with projected plasticity.
- GRU baseline.
- Unconstrained Neural-ODE-style baseline.
- Reservoir baseline.

## Metrics

- prediction MSE,
- reconstruction MSE,
- contraction margin,
- sufficient-contraction, projected-learning stability, and empirical topology
  comparison certificate payloads,
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

For each dataset, `figures/` contains input and latent trajectory plots plus
H1 persistence diagrams for the input and latent point clouds.

`checkpoints/` contains one `*_model.pt` file per dataset in the run. Each
checkpoint stores the `ContractiveCTRNN` state dict, reconstructable model
kwargs, the resolved config, seed, dataset name, and training metrics.
Use `tempora.training.load_contractive_ctrnn_checkpoint` to load a checkpoint
and reconstruct the model with its metadata.

`metrics.json` also stores per-dataset proof-adjacent certificates under
`datasets.<name>.certificates`. The `contraction` certificate records the
theorem identifier, assumptions, limitation, computed contraction margin,
required margin, and certification result. When projected plasticity is enabled,
`learning_stability` records the last projected Oja update, including the
pre-update margin, post-projection margin, update norm, required margin, and
certification result. `topology_comparison` records the empirical H1 persistence
diagram distance against the configured `topology_max_distance` threshold. This
threshold is a review policy for finite smoke artifacts, not a proof of semantic
equivalence. The Markdown report repeats compact certificate summaries for
review.

At the run level, `metrics.json` stores `certificate_summary` with per
certificate-type counts and an explicit list of failed certificates. The resolved
config can also list `required_certificates`; these selected certificate types
are evaluated into `certificate_gate`. This gate separates mandatory smoke-run
checks from optional review evidence such as topology thresholds. The summary and
gate are repeated in the Markdown report before the dataset sections so a
reviewer can see immediately which proof-adjacent checks passed and which need
attention.

## Expected Failure Modes

- non-finite metrics,
- missing optional TDA dependencies,
- output directory not writable,
- training smoke loop does not produce finite states.

## Reproducibility Notes

`metrics.json` stores the seed, resolved config, artifact paths, git commit hash
when available, dependency versions, runtime metadata, proof certificate
payloads, certificate-summary counts, certificate-gate status, and the
topology certificate threshold from the resolved config. The generated Markdown
report repeats these reproducibility fields for review. Generated outputs are
ignored by git.
