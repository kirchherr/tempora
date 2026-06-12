"""Plotting helpers for synthetic trajectories."""

from __future__ import annotations

from collections.abc import Sequence

from tempora.data.types import TemporalDataset


def plot_observations(
    dataset: TemporalDataset,
    *,
    axes: Sequence[int] = (0, 1),
) -> object:
    """Plot selected observation axes and return a matplotlib figure."""

    if len(axes) not in {2, 3}:
        raise ValueError("axes must contain two or three axis indices.")
    if any(axis < 0 or axis >= dataset.observation_dim for axis in axes):
        raise ValueError("axes must refer to observation dimensions.")

    import matplotlib.pyplot as plt

    figure = plt.figure()
    if len(axes) == 3:
        axis = figure.add_subplot(111, projection="3d")
        axis.plot(
            dataset.observations[:, axes[0]],
            dataset.observations[:, axes[1]],
            dataset.observations[:, axes[2]],
        )
    else:
        axis = figure.add_subplot(111)
        axis.plot(dataset.observations[:, axes[0]], dataset.observations[:, axes[1]])
    axis.set_title(str(dataset.metadata.get("system", "temporal dataset")))
    return figure
