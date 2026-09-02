# TEMPORA v0.1 Release Checklist

## Scope

TEMPORA v0.1 is a conservative research MVP. It supports synthetic trajectories,
contractive CTRNN experiments, projected plasticity, stability diagnostics,
persistent homology metrics, simple baselines, and a CI-small smoke benchmark.

It does not claim general temporal semantic preservation, arbitrary
homeomorphism preservation, real-world video understanding, or AGI-like
capability.

## Required Checks

- [x] Confirm Phase 1 through Phase 31 are merged into `main`.
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

- [ ] Choose a project license.
- [ ] Replace `LICENSE.md` placeholder with the selected license text.
- [ ] Update `pyproject.toml` license metadata.

This remains the main release blocker. Until the project owner chooses a
license, the repository keeps the explicit `License Pending` placeholder and
`LicenseRef-Proprietary` metadata.

## Citation

- [x] Review `CITATION.cff` structure.
- [ ] Replace placeholder author metadata if needed before public release.
- [ ] Update version/date fields when tagging an actual release.

## Release Notes

- [ ] Update `CHANGELOG.md` from `0.1.0-alpha - Unreleased` to the release tag.
- [ ] Include only results that were generated and reviewed.
- [ ] Include known limitations and open questions.

Do not tag the release until the license, citation metadata, changelog header,
and final post-decision release gates are complete.
