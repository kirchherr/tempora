# Theorem 01: Sufficient Contraction of the MVP CTRNN

## Statement

For the TEMPORA MVP latent dynamics

```text
dz/dt = -D z + W tanh(z) + B u(t) + b
```

the recurrent latent dynamics satisfy a sufficient contraction condition with
respect to `z` when:

```text
min(D) > L_sigma * ||W||_2
```

In implementation, TEMPORA reports the numerical margin:

```text
contraction_margin = min(D) - L_sigma * ||W||_2
```

A positive margin is treated as a sufficient certificate for this restricted
model class under the assumptions below.

## Assumptions

- `D` is diagonal and strictly positive.
- `W` is finite and square.
- The activation is `tanh` or another activation with documented Lipschitz
  constant `L_sigma`; for `tanh`, `L_sigma <= 1`.
- The contraction statement is only with respect to the latent state `z`.
- Inputs `u(t)` and bias `b` are finite forcing terms and do not alter the
  Jacobian with respect to `z`.
- Spectral norms are computed numerically with finite precision.
- Any strict numerical claim uses an explicit positive margin.

## Proof Sketch

The Jacobian of the latent dynamics with respect to `z` is:

```text
J(z) = -D + W diag(tanh'(z))
```

Because `|tanh'(z_i)| <= L_sigma`, the induced operator norm of the recurrent
term is bounded by:

```text
||W diag(tanh'(z))||_2 <= L_sigma * ||W||_2
```

The damping term contributes at least `min(D)` of diagonal contraction. If
`min(D) > L_sigma * ||W||_2`, the damping dominates the recurrent term in this
sufficient norm bound. This gives a conservative sufficient condition for
contraction of the latent dynamics with respect to `z`.

## Implementation Correspondence

- `src/tempora/models/projections.py` implements `spectral_norm`,
  `contraction_margin`, and `project_recurrent_weights`.
- `src/tempora/models/contractive_ctrnn.py` exposes
  `sufficient_contraction_margin` and `project_recurrent_weight_`.
- `src/tempora/metrics/contraction.py` exposes `model_contraction_margin`.
- `src/tempora/metrics/jacobian.py` provides sampled symmetric-Jacobian
  diagnostics. These are local numerical checks, not a replacement for the
  sufficient spectral condition.

## Empirical Checks

The tests verify that:

- projection reduces spectral norm when the sufficient condition is violated,
- projection restores a positive contraction margin,
- forward passes remain finite,
- gradients can be computed,
- sampled symmetric-Jacobian diagnostics are negative for a projected model in a
  controlled test setting.

## Limitations

This result is a sufficient condition, not a necessary one. It does not prove
optimal learning, semantic preservation, robustness to arbitrary data, or
homeomorphism between input and latent spaces. Numerical margins depend on
finite precision, spectral-norm computation, damping values, activation choice,
and the explicit margin used by the implementation.

## Related Tests

- `tests/test_projection.py`
- `tests/test_contractive_ctrnn.py`

