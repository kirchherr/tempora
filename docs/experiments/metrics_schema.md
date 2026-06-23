# Experiment: Benchmark Metrics Schema

## Purpose

This document describes the machine-readable `metrics.json` artifact emitted by
`scripts/train_synth.py`. The schema is intentionally conservative: it records
finite smoke-benchmark evidence, reproducibility metadata, and proof-adjacent
certificate payloads. It does not encode a proof of general temporal semantic
preservation.

## Top-Level Object

The benchmark metrics file is a JSON object with these required top-level
fields:

- `run_id`: output run identifier.
- `seed`: deterministic benchmark seed.
- `config`: resolved benchmark configuration.
- `artifacts`: relative paths to generated report, figures, checkpoints, and
  config snapshots.
- `git_commit`: repository commit hash when available.
- `dependency_versions`: package versions used for the run.
- `runtime`: host and timing metadata.
- `datasets`: per-dataset benchmark records.
- `certificate_summary`: run-level certificate counts and failure listings.
- `certificate_gate`: policy evaluation for configured mandatory
  certificates.

Generated files referenced by `artifacts` are written under `outputs/<run_id>/`
and are not committed to the repository.

## Config Fields

The `config` object stores the resolved values used by the smoke benchmark,
including dataset selection, training hyperparameters, model dimensions,
baseline settings, and certificate policy.

Certificate-related config fields include:

- `required_certificates`: certificate types that must pass for the run-level
  `certificate_gate` to pass.
- `topology_max_distance`: maximum accepted empirical H1 bottleneck distance
  for the `topology_comparison` certificate.

The topology threshold is a review policy over finite generated artifacts. It is
not a mathematical proof of homeomorphism preservation or semantic equivalence.

## Dataset Records

Each entry under `datasets.<name>` records one synthetic dataset result. Dataset
records contain finite training and evaluation metrics, baseline comparisons,
diagnostic values, artifact paths, and machine-readable certificates.

Expected dataset-level groups include:

- `metrics`: TEMPORA prediction and reconstruction metrics.
- `baselines`: GRU, unconstrained Neural-ODE-style, and reservoir baseline
  results when enabled.
- `diagnostics`: stability, Lyapunov, invariance, robustness, and topology
  metrics.
- `artifacts`: generated figure and checkpoint paths for the dataset.
- `certificates`: proof-adjacent certificate payloads.

All numerical values intended for JSON output must be finite. NaN and Inf are
treated as invalid benchmark artifacts.

## Certificates

Dataset certificates live under `datasets.<name>.certificates`.

The current smoke benchmark emits these certificate types:

- `contraction`: sufficient-contraction certificate with theorem identifier,
  explicit assumptions, limitation text, contraction margin, required margin,
  and certification result.
- `learning_stability`: projected-learning stability certificate for the last
  projected Oja update, including pre-update margin, post-projection margin,
  update norm, required margin, assumptions, limitation text, and certification
  result.
- `topology_comparison`: empirical persistent-homology comparison certificate
  with H1 bottleneck distance, configured `topology_max_distance`, assumptions,
  limitation text, and certification result.

Certificate payloads are evidence records for the generated smoke run. They are
valid only under their stated assumptions and limitations.

## Certificate Summary And Gate

`certificate_summary` aggregates certificate outcomes across all datasets. It
contains per-certificate counts, total failures, and an explicit list of failed
certificate records.

`certificate_gate` evaluates only the configured `required_certificates`.
Mandatory certificates that are missing or uncertified cause the gate to fail.
Optional certificates can still be present as review evidence without being
release-blocking.

The default smoke config currently requires:

- `contraction`
- `learning_stability`

`topology_comparison` remains review evidence unless it is explicitly added to
`required_certificates` in the resolved config.

## Reproducibility Requirements

A benchmark result should not be reported unless the corresponding
`metrics.json`, `config.yaml`, generated figures, checkpoint artifacts, and
`report.md` were produced by an actual run under `outputs/<run_id>/`.

For release review, regenerate the smoke benchmark and then run:

```bash
python scripts/check_certificates.py outputs/benchmark_smoke/metrics.json
```

This command checks the stored `certificate_gate` when present and can recompute
the gate from `datasets`, `certificate_summary`, and `config.required_certificates`
for older compatible metrics files.
