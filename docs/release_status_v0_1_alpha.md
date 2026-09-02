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

## Technical Gates

The technical release path is ready for an alpha tag when these local gates
pass on the final source state:

- `docker compose build tempora`
- `docker compose run --rm tempora`
- `docker compose run --rm tempora python scripts/release_smoke.py --config configs/benchmark_smoke.yaml`

The full Docker gate runs pytest, Ruff, Ruff format checking, and Mypy. The
release smoke command regenerates `outputs/benchmark_smoke/metrics.json`,
`outputs/benchmark_smoke/config.yaml`, `outputs/benchmark_smoke/report.md`,
figures, checkpoints, and `outputs/benchmark_smoke/artifact_manifest.json`.

Generated files under `outputs/` remain uncommitted release artifacts except
for `outputs/.gitkeep`.

After tagging, release provenance can be audited with:

```bash
docker compose run --rm tempora python scripts/release_provenance.py --tag v0.1.0-alpha
```

This writes `outputs/benchmark_smoke/release_provenance.json` and verifies that
the release tag commit matches the `git_commit` recorded in `metrics.json`, the
artifact manifest belongs to the same run, and the stored certificate gate
passed.

## Resolved Release Decisions

The release decisions for `v0.1.0-alpha` are:

- License: MIT.
- License text: `LICENSE.md`.
- Package license metadata: `MIT`.
- Citation file: `CITATION.cff`.
- Citation version: `0.1.0-alpha`.
- Citation release date: `2026-09-02`.
- Citation author metadata: project-level `TEMPORA Contributors`.
- Changelog file: `CHANGELOG.md`.
- Changelog release header: `0.1.0-alpha - 2026-09-02`.

## Tagging Criteria

The `v0.1.0-alpha` tag should be created only after the technical gates pass on
the final source state.

No benchmark numbers should be copied into release notes unless they come from
a reviewed run with its config, seed, metrics file, report, and artifact
manifest.
