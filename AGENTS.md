# TEMPORA Agent Instructions

## Project Goal

TEMPORA is a research codebase for proof-guided temporal self-organization in
continuous-time neural systems.

The first milestone is not a general AGI system and not a broad proof of
temporal semantics. The first milestone is a reproducible MVP that:

1. Generates synthetic dynamical-system datasets.
2. Trains a contractive CTRNN / Neural-ODE-like model.
3. Applies local plasticity with projection into a contractive parameter set.
4. Computes stability certificates.
5. Computes persistent homology metrics between input and latent trajectories.
6. Compares against simple baselines.

## Engineering Rules

- Prefer small, reviewable changes.
- Do not add large datasets to the repository.
- All public functions need type hints.
- Use deterministic seeds in tests.
- Keep experiments reproducible from CLI commands.
- Every feature must include tests.
- Avoid silent numerical failures; check for NaN and Inf.
- Use docstrings for mathematical functions.
- Keep theory claims conservative and explicitly state assumptions.
- Do not overclaim what the experiments prove.

## Standard Commands

Run these commands unless the repository later standardizes different commands:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

For the Docker environment:

```bash
docker compose build tempora
docker compose run --rm tempora
```

## Definition of Done

A task is done only when:

1. Tests pass or failures are clearly explained.
2. New behavior is covered by tests.
3. README or docs are updated if user-facing behavior changed.
4. Numerical assumptions are documented.
5. Any experiment produces reproducible artifacts under `outputs/`.

## Research Constraints

Do not claim that TEMPORA proves general temporal semantic preservation.
For now, claims are limited to:

- contraction certificates under explicit assumptions,
- empirical topology preservation via persistent homology metrics,
- reproducible synthetic benchmarks,
- comparison against simple baselines.

