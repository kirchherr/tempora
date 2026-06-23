# TEMPORA

TEMPORA is a Python research codebase for proof-guided temporal
self-organization in continuous-time neural systems.

The first milestone is a conservative research MVP. It will generate synthetic
dynamical systems, train a contractive continuous-time neural model, apply
projected local plasticity, compute contraction certificates and stability
diagnostics, and measure empirical topology distances between controlled input
and latent trajectories.

TEMPORA does not claim to prove general temporal semantics, general
homeomorphism preservation, real-world video understanding, or AGI-like
capability.

## Repository Status

This repository currently contains the `0.1.0-alpha` release-prep scope:

- Python package with `src/` layout
- pytest, ruff, and mypy configuration
- Docker environment for the research stack
- CI workflow
- conservative MVP and assumptions documentation
- project guidance for research roles and required mathematical tooling
- synthetic trajectory generators and perturbations
- contractive CTRNN model core with projection and Jacobian diagnostics
- machine-readable contraction, projected-learning, and topology certificates
- projected Oja-style plasticity and a small Circle smoke trainer
- persistent homology metrics for finite point-cloud comparisons
- stability and invariance diagnostics for synthetic trajectories
- GRU, unconstrained Neural ODE, and reservoir baselines
- CI-small synthetic smoke benchmark with metrics, figures, and report

Final release tagging and license selection are intentionally pending.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements/docker-cpu.txt
python -m pip install -e ".[dev]" --no-deps
```

On Linux or macOS, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Quickstart

Run the full check suite in Docker:

```bash
docker compose run --rm tempora
```

Run the CI-small synthetic smoke benchmark:

```bash
docker compose run --rm tempora python scripts/train_synth.py --config configs/benchmark_smoke.yaml
```

Example dataset and model configs live under `configs/`, including
`synth_circle.yaml`, `synth_torus.yaml`, `synth_lorenz.yaml`,
`synth_rossler.yaml`, and `contractive_ctrnn.yaml`.

Regenerate a Markdown report from saved metrics:

```bash
docker compose run --rm tempora python scripts/make_report.py outputs/benchmark_smoke/metrics.json
```

Check the mandatory certificate gate from saved benchmark metrics:

```bash
docker compose run --rm tempora python scripts/check_certificates.py outputs/benchmark_smoke/metrics.json
```

Generated metrics, reports, config snapshots, checkpoints, trajectory figures,
and persistence figures are written under `outputs/` and are not committed.

## Docker Setup

The Docker image installs TEMPORA with development and research dependencies,
including PyTorch, torchdiffeq, scipy, sympy, ripser, persim, matplotlib,
pytest, ruff, and mypy.

The Docker build uses the CPU PyTorch wheel index to keep the image suitable for
ordinary development machines and CI smoke checks.

```bash
docker compose --progress plain build tempora
docker compose run --rm tempora
```

Run an interactive shell:

```bash
docker compose run --rm tempora bash
```

## Standard Checks

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

## Research Direction

The MVP centers on the system

```text
dz/dt = -D z + W tanh(z) + B phi(u(t)) + b
```

with a sufficient contraction condition:

```text
min(D) > L_sigma * ||W||_2 + margin
```

The current alpha implements synthetic data generators, a contractive CTRNN,
projection operators, Oja-style plasticity, persistent homology metrics,
Lyapunov estimates, invariance tests, baselines, and a smoke benchmark.

See [docs/research_stack.md](docs/research_stack.md) for the working roles,
mathematical toolkit, and Python dependency rationale.
See [docs/experiments/synthetic_protocol.md](docs/experiments/synthetic_protocol.md)
for the synthetic data API introduced in Phase 1.
See [docs/experiments/plasticity_smoke_protocol.md](docs/experiments/plasticity_smoke_protocol.md)
for the projected-plasticity smoke protocol introduced in Phase 3.
See [docs/experiments/stability_invariance_protocol.md](docs/experiments/stability_invariance_protocol.md)
for the stability and invariance diagnostics introduced in Phase 5.
See [docs/experiments/baseline_comparison_protocol.md](docs/experiments/baseline_comparison_protocol.md)
for the baseline comparison smoke protocol introduced in Phase 6.
See [docs/experiments/benchmark_spec.md](docs/experiments/benchmark_spec.md)
for the synthetic smoke benchmark introduced in Phase 7.
See [docs/theory/assumptions.md](docs/theory/assumptions.md) for the theory
assumptions and theorem documents introduced through Phase 8.
See [docs/release_v0_1.md](docs/release_v0_1.md) for the v0.1 release checklist.

## Non-Goals for v0.1

- No real video datasets.
- No GPU requirement.
- No large transformer stack.
- No distributed training.
- No broad semantic-preservation proof.
- No invented benchmark results.

## Release Status

TEMPORA is currently `0.1.0-alpha` release-prep software. The license is pending
project-owner selection; see [LICENSE.md](LICENSE.md).
