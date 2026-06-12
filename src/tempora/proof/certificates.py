"""Conservative certificate payloads for TEMPORA theorem checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch

from tempora.models.projections import contraction_margin, spectral_norm
from tempora.proof.assumptions import CONTRACTION_ASSUMPTIONS


class SupportsContractionCertificate(Protocol):
    """Minimal protocol for models that expose contraction-relevant tensors."""

    @property
    def damping(self) -> torch.Tensor: ...

    @property
    def recurrent_weight(self) -> torch.Tensor: ...

    lipschitz: float
    margin: float


@dataclass(frozen=True)
class ContractionCertificate:
    """JSON-serializable sufficient-contraction certificate payload."""

    theorem: str
    damping_min: float
    recurrent_spectral_norm: float
    lipschitz: float
    contraction_margin: float
    required_margin: float
    is_certified: bool
    assumptions: tuple[str, ...]
    limitation: str

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the certificate."""

        payload = asdict(self)
        payload["assumptions"] = list(self.assumptions)
        return payload


def certify_model_contraction(
    model: SupportsContractionCertificate,
    *,
    required_margin: float | None = None,
) -> ContractionCertificate:
    """Build a sufficient-contraction certificate from a compatible model."""

    resolved_margin = model.margin if required_margin is None else required_margin
    return certify_sufficient_contraction(
        damping=model.damping,
        recurrent_weights=model.recurrent_weight,
        lipschitz=model.lipschitz,
        required_margin=resolved_margin,
    )


def certify_sufficient_contraction(
    *,
    damping: torch.Tensor,
    recurrent_weights: torch.Tensor,
    lipschitz: float = 1.0,
    required_margin: float = 0.0,
) -> ContractionCertificate:
    """Certify TEMPORA's sufficient spectral contraction condition.

    The certificate captures the numerical statement behind Theorem 01:
    `min(D) - L_sigma * ||W||_2 > required_margin`. It is deliberately only a
    sufficient local-dynamics certificate and not a semantic guarantee.
    """

    if required_margin < 0.0:
        raise ValueError("required_margin must be non-negative.")

    with torch.no_grad():
        margin = contraction_margin(
            damping=damping,
            recurrent_weights=recurrent_weights,
            lipschitz=lipschitz,
        )
        damping_min = torch.min(damping)
        recurrent_norm = spectral_norm(recurrent_weights)

    margin_value = float(margin.detach().item())
    required_margin_value = float(required_margin)
    return ContractionCertificate(
        theorem="theorem_01_sufficient_contraction",
        damping_min=float(damping_min.detach().item()),
        recurrent_spectral_norm=float(recurrent_norm.detach().item()),
        lipschitz=float(lipschitz),
        contraction_margin=margin_value,
        required_margin=required_margin_value,
        is_certified=margin_value > required_margin_value,
        assumptions=CONTRACTION_ASSUMPTIONS,
        limitation=(
            "Sufficient contraction certificate only; does not prove semantic "
            "preservation, homeomorphism, or real-world robustness."
        ),
    )
