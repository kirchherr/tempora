# Experiment: Baseline Comparison Smoke Protocol

## Purpose

Phase 6 adds simple baselines with a shared temporal model interface. The goal
is comparable mechanics, not tuned performance.

## Dataset

Baselines consume `TemporalDataset` directly through:

- `fit(dataset, seed)`
- `encode(dataset)`
- `predict(dataset)`

## Models

- `GRUBaseline`: teacher-forced GRU next-step model.
- `VanillaNeuralODEBaseline`: unconstrained continuous-time recurrent model
  without contraction projection.
- `ReservoirBaseline`: fixed random reservoir with ridge readout.
- `tempora_contractivectrnn`: optional smoke entry using the existing projected
  TEMPORA CTRNN trainer.

## Metrics

Every baseline returns:

- `prediction_mse`
- `reconstruction_mse`
- `fit_epochs`

## Procedure

1. Generate a small deterministic synthetic dataset.
2. Fit each model with a deterministic seed.
3. Collect common metric fields.
4. Compare returned metrics, not broad theoretical claims.

## Expected Failure Modes

- baseline output shape differs from observations,
- non-finite predictions,
- missing common metric fields,
- nondeterministic results for the same seed.

## Reproducibility Notes

The default baselines are intentionally small enough for CI. Future phases can
expand this into benchmark reports with saved artifacts.

