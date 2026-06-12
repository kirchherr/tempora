"""Jacobian diagnostics for latent neural dynamics."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch


def symmetric_jacobian_max_eigenvalue(
    dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    states: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Estimate max eigenvalue of the symmetric Jacobian part on samples.

    For each `(z, u)` sample, this computes the Jacobian of `dynamics(z, u)`
    with respect to `z`, symmetrizes it as `(J + J.T) / 2`, and returns the
    maximum eigenvalue over all samples. Negative values are local numerical
    evidence of contraction at the sampled states.
    """

    if states.ndim != 2:
        raise ValueError("states must have shape (n_samples, latent_dim).")
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape (n_samples, input_dim).")
    if states.shape[0] != inputs.shape[0]:
        raise ValueError("states and inputs must have matching sample counts.")
    if not torch.isfinite(states).all():
        raise ValueError("states must be finite.")
    if not torch.isfinite(inputs).all():
        raise ValueError("inputs must be finite.")

    eigenvalues: list[torch.Tensor] = []
    for state, input_value in zip(states, inputs, strict=True):
        state_for_grad = state.detach().clone().requires_grad_(True)
        input_detached = input_value.detach()

        def state_only(
            next_state: torch.Tensor,
            fixed_input: torch.Tensor = input_detached,
        ) -> torch.Tensor:
            return dynamics(next_state, fixed_input)

        jacobian = cast(
            torch.Tensor,
            torch.autograd.functional.jacobian(  # type: ignore[no-untyped-call]
                state_only,
                state_for_grad,
                create_graph=False,
                vectorize=True,
            ),
        )
        symmetric = 0.5 * (jacobian + jacobian.transpose(0, 1))
        eigenvalues.append(torch.linalg.eigvalsh(symmetric).max())
    return torch.stack(eigenvalues).max()
