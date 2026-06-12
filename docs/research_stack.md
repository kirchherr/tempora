# TEMPORA Research Stack

## Working Roles

TEMPORA work should be split into small, auditable roles:

- Research architect: keeps claims conservative and tied to assumptions.
- Numerical engineer: checks finite values, spectral norms, Jacobians, and ODE
  stability.
- Model engineer: implements contractive CTRNNs, projections, plasticity, and
  baselines.
- Experiment engineer: keeps seeds, configs, metrics, and outputs reproducible.
- Theory reviewer: rejects overbroad semantic or topological claims.

These roles are project guidance for Codex sessions and reviews. They do not
replace tests, documented assumptions, or reproducible experiments.

## Mathematical Toolkit

The first MVP should use proof-near numerical evidence:

- contraction margins from sufficient spectral-norm conditions,
- symmetric Jacobian eigenvalue checks for local contraction evidence,
- Lyapunov estimates as empirical stability diagnostics,
- projection operators that keep recurrent weights inside a certified set,
- persistent homology on finite point clouds for empirical topology comparison,
- controlled perturbation tests for time-warping, noise, and missing segments.

The toolkit supports falsifiable experiments. It does not establish universal
semantic preservation.

## Python Tooling

Core dependencies:

- `numpy`, `scipy`, and `scikit-learn` for numerical data generation and metrics,
- `PyYAML` for reproducible experiment configs,
- `torch` for neural models,
- `torchdiffeq` for Neural-ODE-style integration,
- `sympy` and `z3-solver` for symbolic sanity checks where useful,
- `ripser` and `persim` for persistent homology and diagram distances,
- `matplotlib` for reproducible figures,
- `pytest`, `ruff`, and `mypy` for engineering checks.

Optional future additions should be justified by a concrete phase requirement.
Do not add heavyweight dependencies only because they are interesting.

## Container Boundary

The default container is CPU-first and CI-friendly. GPU support can be added
later with a separate compose profile once a benchmark actually requires it.

