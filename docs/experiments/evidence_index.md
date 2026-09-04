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
- compact certificate-summary failure records,
- certificate-gate status and required certificate names,
- compact certificate-gate failure records.

## Procedure

Run the complete release evidence path:

```bash
python scripts/release_evidence.py --config configs/benchmark_smoke.yaml
```

This command executes the CI-small smoke benchmark, writes the benchmark
artifact manifest, builds and validates the evidence index, renders and
validates the Markdown report, and builds and validates the evidence bundle.

For manual audits, first generate a benchmark run:

```bash
python scripts/release_smoke.py --config configs/benchmark_smoke.yaml
```

Then build an evidence index:

```bash
python scripts/build_evidence_index.py outputs/benchmark_smoke/metrics.json --require-manifest --check-files
```

Validate the generated evidence index for release review:

```bash
python scripts/validate_evidence_index.py outputs/evidence_index.json --require-gates-passed --require-manifests --require-git-commits --check-files
```

Render a Markdown evidence report for review:

```bash
python scripts/render_evidence_report.py outputs/evidence_index.json
```

Validate that the report still matches the index:

```bash
python scripts/validate_evidence_report.py outputs/evidence_index.json outputs/evidence_report.md
```

Build a checksum manifest for the index and report:

```bash
python scripts/build_evidence_bundle.py --index outputs/evidence_index.json --report outputs/evidence_report.md
```

Validate that the bundle still matches local evidence artifacts:

```bash
python scripts/validate_evidence_bundle.py outputs/evidence_bundle.json
```

The index and report commands write:

```text
outputs/evidence_index.json
outputs/evidence_report.md
outputs/evidence_bundle.json
```

Multiple metrics files can be passed to the same command when reviewing a set
of generated runs.

## Expected Failure Modes

- a metrics path is missing,
- a metrics payload fails schema validation,
- `--check-files` is set and a referenced artifact is missing,
- `--require-manifest` is set and `artifact_manifest.json` is missing,
- the artifact manifest run id does not match the metrics run id,
- certificate failure records are malformed,
- report rendering fails when the evidence index is invalid,
- report validation fails when the saved report is stale,
- bundle manifest generation fails when the report and index do not match,
- bundle validation fails when indexed artifact hashes are stale,
- the release evidence runner fails if any upstream benchmark, index, report, or
  bundle gate fails.

## Reproducibility Notes

The evidence index summarizes generated artifacts only. It should be rebuilt
after rerunning benchmarks, changing configs, or changing release-smoke
manifests.
