"""Metrics and numerical diagnostics for TEMPORA."""

from tempora.metrics.contraction import model_contraction_margin
from tempora.metrics.jacobian import symmetric_jacobian_max_eigenvalue
from tempora.metrics.tda import (
    OptionalDependencyError,
    PersistenceResult,
    compare_point_clouds,
    compute_persistence_diagrams,
    persistence_distance,
)

__all__ = [
    "OptionalDependencyError",
    "PersistenceResult",
    "compare_point_clouds",
    "compute_persistence_diagrams",
    "model_contraction_margin",
    "persistence_distance",
    "symmetric_jacobian_max_eigenvalue",
]
