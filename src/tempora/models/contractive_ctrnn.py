"""Contractive continuous-time recurrent neural network."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from tempora.models.projections import contraction_margin, project_recurrent_weights

try:
    from torchdiffeq import odeint as torchdiffeq_odeint
except ImportError:  # pragma: no cover - exercised only without optional dep
    torchdiffeq_odeint = None


class ContractiveCTRNN(nn.Module):
    """Contractive CTRNN for controlled temporal dynamics.

    The latent dynamics are:

    `dz/dt = -D z + W tanh(z) + B u + b`

    with `D > 0`. TEMPORA's MVP uses the sufficient condition
    `min(D) > L_sigma * ||W||_2 + margin` to keep the recurrent latent dynamics
    in a contractive parameter set. This implementation exposes an explicit
    projection method for `W`; callers should invoke it after unconstrained
    optimizer steps or plasticity updates.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        damping_init: float = 1.5,
        recurrent_scale: float = 0.05,
        margin: float = 0.05,
        lipschitz: float = 1.0,
        use_torchdiffeq: bool = False,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive.")
        if damping_init <= 0.0:
            raise ValueError("damping_init must be positive.")
        if recurrent_scale < 0.0:
            raise ValueError("recurrent_scale must be non-negative.")
        if margin <= 0.0:
            raise ValueError("margin must be positive.")
        if lipschitz <= 0.0:
            raise ValueError("lipschitz must be positive.")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.margin = float(margin)
        self.lipschitz = float(lipschitz)
        self.use_torchdiffeq = bool(use_torchdiffeq)

        raw_damping = torch.full(
            (self.latent_dim,),
            _inverse_softplus(float(damping_init)),
            dtype=torch.float32,
        )
        self.raw_damping = nn.Parameter(raw_damping)
        recurrent_weight = torch.empty(self.latent_dim, self.latent_dim)
        nn.init.orthogonal_(recurrent_weight)
        self._recurrent_weight = nn.Parameter(recurrent_scale * recurrent_weight)
        self.input_weight = nn.Parameter(torch.empty(self.latent_dim, self.input_dim))
        nn.init.xavier_uniform_(self.input_weight)
        self.bias = nn.Parameter(torch.zeros(self.latent_dim))

    @property
    def damping(self) -> torch.Tensor:
        """Positive diagonal damping vector `D`."""

        return F.softplus(self.raw_damping)

    @property
    def recurrent_weight(self) -> torch.Tensor:
        """Recurrent weight matrix `W`."""

        return self._recurrent_weight

    def sufficient_contraction_margin(self) -> torch.Tensor:
        """Return `min(D) - L_sigma * ||W||_2` for the current parameters."""

        return contraction_margin(
            damping=self.damping,
            recurrent_weights=self.recurrent_weight,
            lipschitz=self.lipschitz,
        )

    def project_recurrent_weight_(self) -> torch.Tensor:
        """Project `W` in-place into TEMPORA's sufficient contraction set."""

        with torch.no_grad():
            projected = project_recurrent_weights(
                self.recurrent_weight,
                self.damping,
                margin=self.margin,
                lipschitz=self.lipschitz,
            )
            self.recurrent_weight.copy_(projected)
        return self.sufficient_contraction_margin()

    def dynamics(self, state: torch.Tensor, input_value: torch.Tensor) -> torch.Tensor:
        """Evaluate `dz/dt` for batched latent states and inputs."""

        if state.shape[-1] != self.latent_dim:
            raise ValueError("state last dimension must equal latent_dim.")
        if input_value.shape[-1] != self.input_dim:
            raise ValueError("input_value last dimension must equal input_dim.")
        recurrent = torch.tanh(state) @ self.recurrent_weight.transpose(0, 1)
        forced = input_value @ self.input_weight.transpose(0, 1)
        return -self.damping * state + recurrent + forced + self.bias

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        times: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate the latent trajectory for input shape `(batch, steps, dim)`."""

        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, steps, input_dim).")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError("inputs last dimension must equal input_dim.")
        if inputs.shape[1] < 2:
            raise ValueError("inputs must contain at least two time steps.")
        if not torch.isfinite(inputs).all():
            raise ValueError("inputs must be finite.")

        resolved_times = self._resolve_times(
            times=times,
            steps=int(inputs.shape[1]),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        state = self._resolve_initial_state(
            initial_state=initial_state,
            batch_size=int(inputs.shape[0]),
            device=inputs.device,
            dtype=inputs.dtype,
        )

        states = [state]
        for step in range(int(inputs.shape[1] - 1)):
            dt = resolved_times[step + 1] - resolved_times[step]
            if bool((dt <= 0.0).item()):
                raise ValueError("times must be strictly increasing.")
            input_value = inputs[:, step, :]
            if self.use_torchdiffeq and torchdiffeq_odeint is not None:
                state = self._torchdiffeq_step(
                    state=state, input_value=input_value, dt=dt
                )
            else:
                state = _rk4_step(
                    dynamics=self.dynamics,
                    state=state,
                    input_value=input_value,
                    dt=dt,
                )
            if not torch.isfinite(state).all():
                raise RuntimeError("non-finite latent state encountered.")
            states.append(state)
        return torch.stack(states, dim=1)

    def _resolve_times(
        self,
        *,
        times: torch.Tensor | None,
        steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if times is None:
            return torch.arange(steps, device=device, dtype=dtype)
        if times.ndim != 1 or times.shape[0] != steps:
            raise ValueError("times must have shape (steps,).")
        if not torch.isfinite(times).all():
            raise ValueError("times must be finite.")
        return times.to(device=device, dtype=dtype)

    def _resolve_initial_state(
        self,
        *,
        initial_state: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if initial_state is None:
            return torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype)
        if initial_state.shape != (batch_size, self.latent_dim):
            raise ValueError("initial_state must have shape (batch, latent_dim).")
        if not torch.isfinite(initial_state).all():
            raise ValueError("initial_state must be finite.")
        return initial_state.to(device=device, dtype=dtype)

    def _torchdiffeq_step(
        self,
        *,
        state: torch.Tensor,
        input_value: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        if torchdiffeq_odeint is None:
            return _rk4_step(
                dynamics=self.dynamics,
                state=state,
                input_value=input_value,
                dt=dt,
            )

        time_grid = torch.stack(
            [
                torch.zeros((), device=state.device, dtype=state.dtype),
                dt.to(device=state.device, dtype=state.dtype),
            ]
        )

        def rhs(_time: torch.Tensor, ode_state: torch.Tensor) -> torch.Tensor:
            return self.dynamics(ode_state, input_value)

        integrated = cast(
            torch.Tensor,
            torchdiffeq_odeint(rhs, state, time_grid, method="rk4"),
        )
        return integrated[-1]


def _rk4_step(
    *,
    dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    input_value: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    half_dt = 0.5 * dt
    k1 = dynamics(state, input_value)
    k2 = dynamics(state + half_dt * k1, input_value)
    k3 = dynamics(state + half_dt * k2, input_value)
    k4 = dynamics(state + dt * k3, input_value)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _inverse_softplus(value: float) -> float:
    tensor = torch.tensor(value, dtype=torch.float32)
    return float(tensor + torch.log(-torch.expm1(-tensor)))
