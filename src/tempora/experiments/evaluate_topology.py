"""Topology evaluation helpers for controlled TEMPORA point clouds."""

from __future__ import annotations

from typing import Any

import numpy.typing as npt

from tempora.metrics.tda import (
    DistanceMetric,
    compute_persistence_diagrams,
    persistence_distance,
)


def evaluate_topology_pair(
    input_points: npt.ArrayLike,
    latent_points: npt.ArrayLike,
    *,
    metric: DistanceMetric = "bottleneck",
    maxdim: int = 1,
) -> dict[str, Any]:
    """Compare input and latent point clouds via persistence diagrams.

    The returned metrics are empirical distances between finite point-cloud
    summaries. They do not prove semantic equivalence or homeomorphism.
    """

    input_result = compute_persistence_diagrams(input_points, maxdim=maxdim)
    latent_result = compute_persistence_diagrams(latent_points, maxdim=maxdim)
    metrics: dict[str, Any] = {
        "metric": metric,
        "maxdim": maxdim,
        "input_n_points": input_result.metadata["n_points"],
        "latent_n_points": latent_result.metadata["n_points"],
    }
    for homology_dim in range(maxdim + 1):
        metrics[f"{metric}_h{homology_dim}"] = persistence_distance(
            input_result,
            latent_result,
            homology_dim=homology_dim,
            metric=metric,
        )
        metrics[f"input_h{homology_dim}_dominant_lifetime"] = (
            input_result.dominant_lifetime(homology_dim)
        )
        metrics[f"latent_h{homology_dim}_dominant_lifetime"] = (
            latent_result.dominant_lifetime(homology_dim)
        )
    return metrics
