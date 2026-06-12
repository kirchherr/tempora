"""Finite-trajectory Lyapunov diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class LyapunovEstimate:
    """Nearest-neighbor divergence summary for one finite trajectory."""

    largest_exponent: float
    horizons: tuple[float, ...]
    mean_log_divergence: tuple[float, ...]
    pair_count: int

    def to_metrics(self) -> dict[str, object]:
        return {
            "largest_lyapunov_estimate": self.largest_exponent,
            "lyapunov_horizons": list(self.horizons),
            "lyapunov_mean_log_divergence": list(self.mean_log_divergence),
            "lyapunov_pair_count": self.pair_count,
        }


def estimate_largest_lyapunov(
    trajectory: npt.ArrayLike,
    *,
    times: npt.ArrayLike | None = None,
    min_separation: int = 2,
    max_horizon: int = 6,
    eps: float = 1e-12,
) -> LyapunovEstimate:
    """Estimate the largest Lyapunov exponent from a finite trajectory.

    This is a compact Rosenstein-style diagnostic: for each valid point, find a
    temporally separated nearest neighbor, track mean log-distance growth over
    short horizons, and fit a line. It is empirical and sensitive to sampling,
    trajectory length, and noise.
    """

    points = validate_trajectory(trajectory)
    resolved_times = _resolve_times(times=times, n_steps=points.shape[0])
    if min_separation < 1:
        raise ValueError("min_separation must be at least 1.")
    if max_horizon < 1:
        raise ValueError("max_horizon must be at least 1.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    usable_steps = points.shape[0] - max_horizon
    if usable_steps <= min_separation + 1:
        raise ValueError("trajectory is too short for the requested horizons.")

    pairs = _nearest_temporally_separated_pairs(
        points[:usable_steps],
        min_separation=min_separation,
    )
    if not pairs:
        raise ValueError("no valid neighbor pairs found.")

    log_divergence: list[float] = []
    horizons: list[float] = []
    for horizon in range(1, max_horizon + 1):
        ratios = []
        for first, second in pairs:
            initial = np.linalg.norm(points[first] - points[second])
            future = np.linalg.norm(points[first + horizon] - points[second + horizon])
            ratios.append(np.log((future + eps) / (initial + eps)))
        log_divergence.append(float(np.mean(ratios)))
        horizon_dt = resolved_times[horizon] - resolved_times[0]
        horizons.append(float(horizon_dt))

    slope = float(
        np.polyfit(np.asarray(horizons), np.asarray(log_divergence), deg=1)[0]
    )
    if not np.isfinite(slope):
        raise RuntimeError("non-finite Lyapunov estimate.")
    return LyapunovEstimate(
        largest_exponent=slope,
        horizons=tuple(horizons),
        mean_log_divergence=tuple(log_divergence),
        pair_count=len(pairs),
    )


def validate_trajectory(trajectory: npt.ArrayLike) -> FloatArray:
    """Validate a finite trajectory with shape `(n_steps, dim)`."""

    points = np.asarray(trajectory, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("trajectory must have shape (n_steps, dim).")
    if points.shape[0] < 4:
        raise ValueError("trajectory must contain at least four samples.")
    if points.shape[1] < 1:
        raise ValueError("trajectory must have at least one dimension.")
    if not np.all(np.isfinite(points)):
        raise ValueError("trajectory must contain only finite values.")
    return points


def _resolve_times(*, times: npt.ArrayLike | None, n_steps: int) -> FloatArray:
    if times is None:
        return np.arange(n_steps, dtype=np.float64)
    resolved = np.asarray(times, dtype=np.float64)
    if resolved.shape != (n_steps,):
        raise ValueError("times must have shape (n_steps,).")
    if not np.all(np.isfinite(resolved)):
        raise ValueError("times must contain only finite values.")
    if not np.all(np.diff(resolved) > 0.0):
        raise ValueError("times must be strictly increasing.")
    return resolved


def _nearest_temporally_separated_pairs(
    points: FloatArray,
    *,
    min_separation: int,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for index, point in enumerate(points):
        distances = np.linalg.norm(points - point, axis=1)
        invalid = np.abs(np.arange(points.shape[0]) - index) < min_separation
        distances[invalid] = np.inf
        neighbor = int(np.argmin(distances))
        if np.isfinite(distances[neighbor]):
            pairs.append((index, neighbor))
    return pairs
