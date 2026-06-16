# Theorem 02: Stability Preservation Under Projected Updates

## Statement

If the TEMPORA recurrent weight matrix is projected into the sufficient
contractive parameter set after each optimizer or local plasticity update, then
the post-projection recurrent parameters satisfy the same sufficient contraction
condition used by Theorem 01, subject to finite-precision tolerance and positive
margin assumptions.

This is a statement about the parameter state after projection, not about the
unprojected update path.

## Assumptions

- The model class and contraction assumptions from Theorem 01 hold.
- Damping values remain strictly positive.
- The projection margin is positive and smaller than `min(D)`.
- The spectral norm used by projection is finite.
- Plasticity updates are finite.
- Projection is applied after every optimizer step or local plasticity update
  that may alter `W`.
- Claims concern the post-projection value of `W`, not transient intermediate
  values.

## Proof Sketch

TEMPORA projects recurrent weights by radial rescaling when:

```text
L_sigma * ||W||_2 >= min(D) - margin
```

The projected matrix is:

```text
W_projected = W * ((min(D) - margin) / (L_sigma * ||W||_2 + eps))
```

for violating weights. Under finite positive `eps` and a valid margin, this
places the scaled recurrent matrix at or inside the allowed spectral radius up
to numerical tolerance. Applying this projection after each update ensures that
the stored recurrent matrix is returned to the sufficient contraction set before
subsequent certified evaluations.

The Oja-style update itself is not claimed to be stable. The preservation claim
comes from enforcing the projection after the update.

## Implementation Correspondence

- `src/tempora/models/plasticity.py` implements `oja_delta` and
  `apply_projected_oja_update_`.
- `apply_projected_oja_update_` logs contraction margin before and after the
  projected local update.
- `src/tempora/training/trainer.py` calls `project_recurrent_weight_` after
  optimizer steps and applies projected Oja-style updates in the smoke loop.
- `src/tempora/training/callbacks.py` records JSON-serializable margins and
  losses.
- `src/tempora/proof/certificates.py` exposes JSON-serializable projected-update
  stability certificates based on post-projection margins.

## Empirical Checks

The tests verify that:

- repeated projected Oja updates preserve a positive contraction margin,
- recurrent weight norms remain bounded in a controlled setting,
- the small circle smoke task produces decreasing loss,
- training metrics are JSON-serializable,
- the benchmark runner records contraction margins in saved metrics.

## Limitations

This result does not imply that plastic learning is globally optimal, that the
unprojected update is stable, or that training will converge for every dataset.
It also does not prove general temporal semantics. The claim is limited to the
stored post-projection recurrent parameters under the documented numerical
assumptions.

## Related Tests

- `tests/test_plasticity.py`
- `tests/test_end_to_end_synth.py`
- `tests/test_run_synthetic.py`
- `tests/test_learning_stability_certificates.py`
