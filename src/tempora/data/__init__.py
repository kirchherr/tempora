"""Synthetic temporal data utilities for TEMPORA."""

from tempora.data.augmentations import add_gaussian_noise, mask_missing_segment
from tempora.data.circle import generate_circle
from tempora.data.lorenz import generate_lorenz
from tempora.data.rossler import generate_rossler
from tempora.data.time_warp import time_warp
from tempora.data.torus import generate_torus
from tempora.data.types import TemporalDataset

__all__ = [
    "TemporalDataset",
    "add_gaussian_noise",
    "generate_circle",
    "generate_lorenz",
    "generate_rossler",
    "generate_torus",
    "mask_missing_segment",
    "time_warp",
]
