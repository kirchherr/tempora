# TEMPORA v0.1-alpha Release Readiness Status

## Scope

This status note records the release-readiness state for the current
`0.1.0-alpha` preparation line. It is limited to the conservative TEMPORA v0.1
MVP: synthetic dynamical systems, a contractive CTRNN, projected local
plasticity, stability diagnostics, persistent-homology comparison, simple
baselines, and a CI-small smoke benchmark.

TEMPORA still does not claim general temporal semantic preservation, arbitrary
homeomorphism preservation, real-world video understanding, or AGI-like
capability.

## Verified Technical Gates

The technical release path is ready for final owner review when these local
gates pass:

- `docker compose build tempora`
- `docker compose run --rm tempora`
- `docker compose run --rm tempora python scripts/release_smoke.py --config configs/benchmark_smoke.yaml`

The full Docker gate runs pytest, Ruff, Ruff format checking, and Mypy. The
release smoke command regenerates `outputs/benchmark_smoke/metrics.json`,
`outputs/benchmark_smoke/config.yaml`, `outputs/benchmark_smoke/report.md`,
figures, checkpoints, and `outputs/benchmark_smoke/artifact_manifest.json`.

Generated files under `outputs/` remain uncommitted release artifacts except
for `outputs/.gitkeep`.

## Release Blockers

These items require project-owner decisions before a public tag:

- Choose the project license.
- Replace `LICENSE.md` with the selected license text.
- Update `pyproject.toml` license metadata.
- Review and replace placeholder author metadata in `CITATION.cff` if needed.
- Update citation version and date fields when the release tag is chosen.
- Change `CHANGELOG.md` from `0.1.0-alpha - Unreleased` to the final tag.
- Rerun the technical gates after those decisions.

## Tagging Criteria

The `v0.1.0-alpha` tag should be created only after the release blockers are
resolved and the technical gates pass on the final tagged source state.

No benchmark numbers should be copied into release notes unless they come from
a reviewed run with its config, seed, metrics file, report, and artifact
manifest.
