"""Persistent-homology metrics for finite TEMPORA point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
DistanceMetric = Literal["bottleneck", "wasserstein"]


class OptionalDependencyError(RuntimeError):
    """Raised when an optional TDA dependency is unavailable."""


@dataclass(frozen=True)
class PersistenceResult:
    """Persistence diagrams and metadata for a finite point cloud."""

    diagrams: tuple[FloatArray, ...]
    metadata: dict[str, Any]

    def diagram(self, homology_dim: int) -> FloatArray:
        if homology_dim < 0 or homology_dim >= len(self.diagrams):
            raise ValueError("homology_dim is not available in this result.")
        return self.diagrams[homology_dim]

    def dominant_lifetime(self, homology_dim: int) -> float:
        """Return the largest finite persistence lifetime in a homology degree."""

        diagram = finite_persistence_pairs(self.diagram(homology_dim))
        if diagram.size == 0:
            return 0.0
        lifetimes = diagram[:, 1] - diagram[:, 0]
        return float(np.max(lifetimes))


def compute_persistence_diagrams(
    point_cloud: npt.ArrayLike,
    *,
    maxdim: int = 1,
    thresh: float = np.inf,
) -> PersistenceResult:
    """Compute Vietoris-Rips persistence diagrams for a finite point cloud."""

    points = validate_point_cloud(point_cloud)
    if maxdim < 0:
        raise ValueError("maxdim must be non-negative.")
    if not np.isfinite(thresh) and not np.isinf(thresh):
        raise ValueError("thresh must be finite or infinity.")

    ripser = _load_ripser()
    raw = ripser(points, maxdim=maxdim, thresh=thresh)
    diagrams = tuple(np.asarray(diagram, dtype=np.float64) for diagram in raw["dgms"])
    return PersistenceResult(
        diagrams=diagrams,
        metadata={
            "n_points": int(points.shape[0]),
            "point_dim": int(points.shape[1]),
            "maxdim": int(maxdim),
            "thresh": float(thresh),
            "backend": "ripser",
        },
    )


def persistence_distance(
    first: PersistenceResult | tuple[FloatArray, ...],
    second: PersistenceResult | tuple[FloatArray, ...],
    *,
    homology_dim: int,
    metric: DistanceMetric = "bottleneck",
) -> float:
    """Compute a diagram distance for one homology dimension."""

    first_diagram = _diagram_from(first, homology_dim)
    second_diagram = _diagram_from(second, homology_dim)
    if metric == "bottleneck":
        distance_fn = _load_persim_function("bottleneck")
    elif metric == "wasserstein":
        distance_fn = _load_persim_function("wasserstein")
    else:
        raise ValueError("metric must be 'bottleneck' or 'wasserstein'.")
    distance = distance_fn(
        finite_persistence_pairs(first_diagram),
        finite_persistence_pairs(second_diagram),
    )
    return float(distance)


def compare_point_clouds(
    first_points: npt.ArrayLike,
    second_points: npt.ArrayLike,
    *,
    maxdim: int = 1,
    metric: DistanceMetric = "bottleneck",
) -> dict[str, float]:
    """Compare two finite point clouds using H0/H1 persistence distances."""

    first = compute_persistence_diagrams(first_points, maxdim=maxdim)
    second = compute_persistence_diagrams(second_points, maxdim=maxdim)
    metrics: dict[str, float] = {}
    for homology_dim in range(maxdim + 1):
        metrics[f"{metric}_h{homology_dim}"] = persistence_distance(
            first,
            second,
            homology_dim=homology_dim,
            metric=metric,
        )
    return metrics


def finite_persistence_pairs(diagram: npt.ArrayLike) -> FloatArray:
    """Return only finite birth-death pairs from a persistence diagram."""

    array = np.asarray(diagram, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("diagram must have shape (n_pairs, 2).")
    return cast(FloatArray, array[np.isfinite(array).all(axis=1)])


def validate_point_cloud(point_cloud: npt.ArrayLike) -> FloatArray:
    """Validate a finite point cloud with shape `(n_points, point_dim)`."""

    points = np.asarray(point_cloud, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("point_cloud must have shape (n_points, point_dim).")
    if points.shape[0] < 2:
        raise ValueError("point_cloud must contain at least two points.")
    if points.shape[1] < 1:
        raise ValueError("point_cloud must have at least one dimension.")
    if not np.all(np.isfinite(points)):
        raise ValueError("point_cloud must contain only finite values.")
    return points


def _diagram_from(
    value: PersistenceResult | tuple[FloatArray, ...],
    homology_dim: int,
) -> FloatArray:
    if homology_dim < 0:
        raise ValueError("homology_dim must be non-negative.")
    if isinstance(value, PersistenceResult):
        return value.diagram(homology_dim)
    if homology_dim >= len(value):
        raise ValueError("homology_dim is not available in this result.")
    return value[homology_dim]


def _load_ripser() -> Any:
    try:
        module = import_module("ripser")
    except ImportError as exc:
        raise OptionalDependencyError(
            "Persistent homology requires optional dependency 'ripser'. "
            "Install TEMPORA with the research extras or Docker image."
        ) from exc
    return module.ripser


def _load_persim_function(name: DistanceMetric) -> Any:
    try:
        module = import_module("persim")
    except ImportError as exc:
        raise OptionalDependencyError(
            "Persistence diagram distances require optional dependency 'persim'. "
            "Install TEMPORA with the research extras or Docker image."
        ) from exc
    return getattr(module, name)
