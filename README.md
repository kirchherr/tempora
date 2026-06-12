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

This repository is initialized for Phase 0 of the master plan:

- Python package with `src/` layout
- pytest, ruff, and mypy configuration
- Docker environment for the research stack
- CI workflow
- conservative MVP and assumptions documentation
- project guidance for research roles and required mathematical tooling
- synthetic trajectory generators and perturbations
- contractive CTRNN model core with projection and Jacobian diagnostics
- projected Oja-style plasticity and a small Circle smoke trainer

TDA metrics, baselines, and benchmarks are intentionally not implemented yet.

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

Future phases will implement synthetic data generators, a contractive CTRNN,
projection operators, Oja-style plasticity, persistent homology metrics,
Lyapunov estimates, invariance tests, and baselines.

See [docs/research_stack.md](docs/research_stack.md) for the working roles,
mathematical toolkit, and Python dependency rationale.
See [docs/experiments/synthetic_protocol.md](docs/experiments/synthetic_protocol.md)
for the synthetic data API introduced in Phase 1.
See [docs/experiments/plasticity_smoke_protocol.md](docs/experiments/plasticity_smoke_protocol.md)
for the projected-plasticity smoke protocol introduced in Phase 3.

## Non-Goals for v0.1

- No real video datasets.
- No GPU requirement.
- No large transformer stack.
- No distributed training.
- No broad semantic-preservation proof.
- No invented benchmark results.
