# TEMPORA v0.1 Release Checklist

## Scope

TEMPORA v0.1 is a conservative research MVP. It supports synthetic trajectories,
contractive CTRNN experiments, projected plasticity, stability diagnostics,
persistent homology metrics, simple baselines, and a CI-small smoke benchmark.

It does not claim general temporal semantic preservation, arbitrary
homeomorphism preservation, real-world video understanding, or AGI-like
capability.

## Required Checks

- [x] Confirm Phase 1 through Phase 32 release-readiness work is merged into the
  final source state.
- [x] Merge reviewed branches in order.
- [x] Run `docker compose build tempora`.
- [x] Run `docker compose run --rm tempora`.
- [x] Run `docker compose run --rm tempora python scripts/release_smoke.py --config configs/benchmark_smoke.yaml`.
- [x] Confirm `outputs/benchmark_smoke/metrics.json` exists after the smoke run.
- [x] Confirm `outputs/benchmark_smoke/config.yaml` exists after the smoke run.
- [x] Confirm `outputs/benchmark_smoke/report.md` exists after the smoke run.
- [x] Confirm `outputs/benchmark_smoke/artifact_manifest.json` exists after the
  smoke run.
- [x] Confirm generated figures exist under `outputs/benchmark_smoke/figures/`.
- [x] Confirm trajectory and persistence figures exist for each benchmark dataset.
- [x] Confirm model checkpoints exist under `outputs/benchmark_smoke/checkpoints/`.
- [x] Confirm release provenance can link the tag, metrics, artifact manifest,
  and certificate gate.
- [x] Confirm generated outputs are not staged for commit.
- [x] Confirm docs and README links resolve.
- [x] Confirm GitHub Actions CI includes tests, linting, typing, the release
  smoke command, metrics schema validation, artifact path validation, and
  certificate gate check.
- [x] Confirm theory documents include assumptions, limitations, and related
  tests.
- [x] Confirm no invented benchmark results are added to docs or changelog.

See [release_status_v0_1_alpha.md](release_status_v0_1_alpha.md) for the
current release-readiness summary and remaining owner decisions.

## License Decision

- [x] Choose the project license.
- [x] Replace `LICENSE.md` placeholder with the selected license text.
- [x] Update `pyproject.toml` license metadata.

TEMPORA v0.1-alpha uses the MIT License. The license text is stored in
`LICENSE.md`, and package metadata uses the `MIT` SPDX identifier.

## Citation

- [x] Review `CITATION.cff` structure.
- [x] Keep project-level `TEMPORA Contributors` author metadata without
  inferring person-specific author details.
- [x] Update version/date fields for the alpha release.

## Release Notes

- [x] Update `CHANGELOG.md` from `0.1.0-alpha - Unreleased` to the release tag.
- [x] Include only results that were generated and reviewed.
- [x] Include known limitations and open questions.

The release tag should point to the final source state after post-decision
release gates pass.
