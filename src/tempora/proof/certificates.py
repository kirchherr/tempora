"""Conservative certificate payloads for TEMPORA theorem checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch

from tempora.models.plasticity import PlasticityLog
from tempora.models.projections import contraction_margin, spectral_norm
from tempora.proof.assumptions import (
    CONTRACTION_ASSUMPTIONS,
    LEARNING_STABILITY_ASSUMPTIONS,
    TOPOLOGY_COMPARISON_ASSUMPTIONS,
)


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


@dataclass(frozen=True)
class LearningStabilityCertificate:
    """JSON-serializable certificate for one projected learning update."""

    theorem: str
    margin_before: float
    margin_after: float
    required_margin: float
    weight_norm_before: float
    weight_norm_after: float
    update_norm: float
    is_certified: bool
    assumptions: tuple[str, ...]
    limitation: str

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the certificate."""

        payload = asdict(self)
        payload["assumptions"] = list(self.assumptions)
        return payload


@dataclass(frozen=True)
class TopologyComparisonCertificate:
    """JSON-serializable certificate for finite TDA metric comparisons."""

    theorem: str
    metric: str
    homology_dim: int
    distance: float
    max_distance: float
    input_n_points: int
    latent_n_points: int
    input_dominant_lifetime: float
    latent_dominant_lifetime: float
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


def certify_topology_comparison(
    metrics: dict[str, Any],
    *,
    homology_dim: int = 1,
    max_distance: float,
) -> TopologyComparisonCertificate:
    """Certify an empirical persistence-diagram distance against a threshold."""

    if homology_dim < 0:
        raise ValueError("homology_dim must be non-negative.")
    if max_distance < 0.0:
        raise ValueError("max_distance must be non-negative.")

    metric = str(metrics.get("metric", "bottleneck"))
    distance_key = f"{metric}_h{homology_dim}"
    input_lifetime_key = f"input_h{homology_dim}_dominant_lifetime"
    latent_lifetime_key = f"latent_h{homology_dim}_dominant_lifetime"
    distance = _required_float_metric(metrics, distance_key)
    return TopologyComparisonCertificate(
        theorem="theorem_03_empirical_persistence_diagram_comparison",
        metric=metric,
        homology_dim=homology_dim,
        distance=distance,
        max_distance=float(max_distance),
        input_n_points=_required_int_metric(metrics, "input_n_points"),
        latent_n_points=_required_int_metric(metrics, "latent_n_points"),
        input_dominant_lifetime=_required_float_metric(metrics, input_lifetime_key),
        latent_dominant_lifetime=_required_float_metric(metrics, latent_lifetime_key),
        is_certified=distance <= float(max_distance),
        assumptions=TOPOLOGY_COMPARISON_ASSUMPTIONS,
        limitation=(
            "Empirical finite point-cloud comparison only; does not prove "
            "semantic equivalence, homeomorphism, or real-world robustness."
        ),
    )


def certify_projected_update_stability(
    log: PlasticityLog,
    *,
    required_margin: float,
) -> LearningStabilityCertificate:
    """Certify that one projected update preserved the required margin.

    The certificate is based on `PlasticityLog.margin_after`. It does not claim
    that the unprojected plasticity update is stable.
    """

    if required_margin < 0.0:
        raise ValueError("required_margin must be non-negative.")
    return LearningStabilityCertificate(
        theorem="theorem_02_projected_learning_stability",
        margin_before=log.margin_before,
        margin_after=log.margin_after,
        required_margin=float(required_margin),
        weight_norm_before=log.weight_norm_before,
        weight_norm_after=log.weight_norm_after,
        update_norm=log.update_norm,
        is_certified=log.margin_after > float(required_margin),
        assumptions=LEARNING_STABILITY_ASSUMPTIONS,
        limitation=(
            "Post-projection stability certificate only; does not prove "
            "convergence, optimality, semantic preservation, or stability of "
            "the unprojected update path."
        ),
    )


def _required_float_metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"metrics.{key} must be numeric.")
    return float(value)


def _required_int_metric(metrics: dict[str, Any], key: str) -> int:
    value = metrics.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"metrics.{key} must be an integer.")
    return value


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
