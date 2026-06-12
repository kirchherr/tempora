"""Metrics and numerical diagnostics for TEMPORA."""

from tempora.metrics.contraction import model_contraction_margin
from tempora.metrics.invariance import (
    missing_segment_robustness_score,
    noise_robustness_score,
    representation_distance,
    representation_similarity_score,
    time_warp_invariance_score,
)
from tempora.metrics.jacobian import symmetric_jacobian_max_eigenvalue
from tempora.metrics.lyapunov import LyapunovEstimate, estimate_largest_lyapunov
from tempora.metrics.reconstruction import reconstruction_mse
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
    "LyapunovEstimate",
    "compare_point_clouds",
    "compute_persistence_diagrams",
    "estimate_largest_lyapunov",
    "missing_segment_robustness_score",
    "model_contraction_margin",
    "noise_robustness_score",
    "persistence_distance",
    "reconstruction_mse",
    "representation_distance",
    "representation_similarity_score",
    "symmetric_jacobian_max_eigenvalue",
    "time_warp_invariance_score",
]
