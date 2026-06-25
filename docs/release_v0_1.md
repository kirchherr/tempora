# TEMPORA v0.1 Release Checklist

## Scope

TEMPORA v0.1 is a conservative research MVP. It supports synthetic trajectories,
contractive CTRNN experiments, projected plasticity, stability diagnostics,
persistent homology metrics, simple baselines, and a CI-small smoke benchmark.

It does not claim general temporal semantic preservation, arbitrary
homeomorphism preservation, real-world video understanding, or AGI-like
capability.

## Required Checks

- [ ] Review stacked branches from Phase 1 through Phase 9.
- [ ] Merge reviewed branches in order.
- [ ] Run `docker compose build tempora`.
- [ ] Run `docker compose run --rm tempora`.
- [ ] Run `docker compose run --rm tempora python scripts/train_synth.py --config configs/benchmark_smoke.yaml`.
- [ ] Run `docker compose run --rm tempora python scripts/validate_metrics.py outputs/benchmark_smoke/metrics.json --check-files`.
- [ ] Run `docker compose run --rm tempora python scripts/check_certificates.py outputs/benchmark_smoke/metrics.json`.
- [ ] Confirm `outputs/benchmark_smoke/metrics.json` exists after the smoke run.
- [ ] Confirm `outputs/benchmark_smoke/config.yaml` exists after the smoke run.
- [ ] Confirm `outputs/benchmark_smoke/report.md` exists after the smoke run.
- [ ] Confirm generated figures exist under `outputs/benchmark_smoke/figures/`.
- [ ] Confirm trajectory and persistence figures exist for each benchmark dataset.
- [ ] Confirm model checkpoints exist under `outputs/benchmark_smoke/checkpoints/`.
- [ ] Confirm generated outputs are not staged for commit.
- [ ] Confirm docs and README links resolve.
- [ ] Confirm GitHub Actions CI includes tests, linting, typing, the smoke
  benchmark command, metrics schema validation, and certificate gate check.
- [ ] Confirm theory documents include assumptions, limitations, and related
  tests.
- [ ] Confirm no invented benchmark results are added to docs or changelog.

## License Decision

- [ ] Choose a project license.
- [ ] Replace `LICENSE.md` placeholder with the selected license text.
- [ ] Update `pyproject.toml` license metadata.

## Citation

- [ ] Review `CITATION.cff`.
- [ ] Replace placeholder author metadata if needed before public release.
- [ ] Update version/date fields when tagging an actual release.

## Release Notes

- [ ] Update `CHANGELOG.md` from `0.1.0-alpha - Unreleased` to the release tag.
- [ ] Include only results that were generated and reviewed.
- [ ] Include known limitations and open questions.
