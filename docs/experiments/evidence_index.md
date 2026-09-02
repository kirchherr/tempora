# Experiment: Evidence Index

## Purpose

The evidence index is a compact review artifact for one or more generated
benchmark runs. It records which `metrics.json` files were indexed, which git
commits they reference, which synthetic datasets are present, whether stored
certificate gates passed, and whether a release-smoke artifact manifest was
available.

It is a reproducibility and review aid. It is not a proof of general temporal
semantic preservation and does not replace inspecting the underlying
`metrics.json`, `report.md`, figures, checkpoints, and artifact manifests.

## Dataset

The index reads already-generated benchmark artifacts. It does not generate new
synthetic datasets.

## Models

The index is model-agnostic with respect to the benchmark payload. For the
current alpha benchmark, indexed runs contain TEMPORA Contractive CTRNN results
and the configured baselines.

## Metrics

The index records:

- run id,
- seed,
- metrics path,
- git commit,
- dataset names and count,
- optional artifact manifest path and artifact count,
- certificate types,
- certificate-summary failure count,
- certificate-gate status and required certificate names.

## Procedure

First generate a benchmark run:

```bash
python scripts/release_smoke.py --config configs/benchmark_smoke.yaml
```

Then build an evidence index:

```bash
python scripts/build_evidence_index.py outputs/benchmark_smoke/metrics.json --require-manifest --check-files
```

The command writes:

```text
outputs/evidence_index.json
```

Multiple metrics files can be passed to the same command when reviewing a set
of generated runs.

## Expected Failure Modes

- a metrics path is missing,
- a metrics payload fails schema validation,
- `--check-files` is set and a referenced artifact is missing,
- `--require-manifest` is set and `artifact_manifest.json` is missing,
- the artifact manifest run id does not match the metrics run id.

## Reproducibility Notes

The evidence index summarizes generated artifacts only. It should be rebuilt
after rerunning benchmarks, changing configs, or changing release-smoke
manifests.
