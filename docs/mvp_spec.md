# TEMPORA MVP Specification

## Purpose

TEMPORA v0.1 is a proof-guided research MVP for controlled temporal dynamics.
It will test whether a contractive continuous-time neural system can learn
stable latent representations for synthetic dynamical systems while measuring
empirical topology distances between controlled input and latent trajectories.

## Scope

The MVP includes:

- synthetic circle, torus, Lorenz, and Rossler trajectories,
- time warping, Gaussian noise, and missing-segment perturbations,
- a contractive CTRNN / Neural-ODE-like model,
- recurrent-weight projection into a contractive parameter set,
- Oja-style local plasticity with homeostasis,
- contraction certificates and Jacobian-based numerical diagnostics,
- persistent homology metrics for input and latent point clouds,
- Lyapunov and invariance estimates,
- simple baselines.

## Non-Scope

TEMPORA v0.1 does not include real video datasets, GPU-only training, large
transformers, distributed training, broad semantic preservation claims, or
formal proof of homeomorphism for arbitrary data.

## Core Model

The initial model family is:

```text
dz/dt = -D z + W tanh(z) + B phi(u(t)) + b
```

The sufficient contraction condition used by the MVP is:

```text
min(D) > L_sigma * ||W||_2 + margin
```

The implementation will compute:

```text
contraction_margin = min(D) - L_sigma * spectral_norm(W)
```

The recurrent weight matrix must be projected after unconstrained updates when
the sufficient condition is violated.

## Acceptance Criteria

- The package installs reproducibly.
- Tests, linting, and type checks run in local and Docker environments.
- New mathematical claims name assumptions and limitations.
- Experiments write reproducible artifacts under `outputs/`.
- Missing optional research dependencies fail with clear errors.
