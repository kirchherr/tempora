# Experiment: Projected Plasticity Smoke Training

## Purpose

Phase 3 verifies that TEMPORA can combine gradient updates and local Oja-style
plasticity while projecting recurrent weights back into the sufficient
contraction set after each update.

## Dataset

The smoke test uses a small deterministic circle trajectory. It is intentionally
too small to support benchmark claims.

## Models

The smoke loop uses `ContractiveCTRNN` with `latent_dim == observation_dim`, so
latent states can be compared directly to next-step observations.

## Metrics

- prediction mean squared error,
- contraction margin,
- recurrent weight norm,
- plasticity margin after local updates.

## Procedure

1. Project recurrent weights before training.
2. Run one gradient step on next-step prediction loss.
3. Project recurrent weights after the optimizer step.
4. Apply one Oja-style local plasticity update.
5. Project recurrent weights again.
6. Store JSON-serializable epoch metrics.

## Expected Failure Modes

- Loss does not decrease on the tiny smoke task.
- Projection margin becomes non-positive.
- Recurrent weight norms grow despite projection.
- Metrics contain non-finite values.

## Reproducibility Notes

Seeds are fixed in tests. This protocol checks mechanics only and should not be
reported as evidence of broad learning performance.

