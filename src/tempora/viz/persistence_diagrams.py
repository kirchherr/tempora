"""Persistence diagram visualization helpers."""

from __future__ import annotations

import numpy as np

from tempora.metrics.tda import PersistenceResult, finite_persistence_pairs


def plot_persistence_diagram(
    result: PersistenceResult,
    *,
    homology_dim: int = 1,
) -> object:
    """Plot one finite persistence diagram and return a matplotlib figure."""

    diagram = finite_persistence_pairs(result.diagram(homology_dim))
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    if diagram.size > 0:
        axis.scatter(diagram[:, 0], diagram[:, 1], s=20)
        lower = float(np.min(diagram))
        upper = float(np.max(diagram))
    else:
        lower = 0.0
        upper = 1.0
    axis.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    axis.set_xlabel("birth")
    axis.set_ylabel("death")
    axis.set_title(f"H{homology_dim} persistence")
    return figure
