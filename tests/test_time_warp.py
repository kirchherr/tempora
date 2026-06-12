import numpy as np
import pytest

from tempora.data import (
    add_gaussian_noise,
    generate_circle,
    mask_missing_segment,
    time_warp,
)


def test_time_warp_changes_times_but_preserves_trajectory_values() -> None:
    dataset = generate_circle(n_steps=64, duration=2.0, seed=71)

    warped = time_warp(dataset, duration_scale=1.5, exponent=2.0)

    assert warped.metadata["augmentation"] == "time_warp"
    assert warped.metadata["source_system"] == "circle"
    assert warped.times[-1] == pytest.approx(3.0)
    assert not np.allclose(warped.times, dataset.times)
    assert np.all(np.diff(warped.times) > 0.0)
    np.testing.assert_allclose(warped.observations, dataset.observations)
    np.testing.assert_allclose(warped.clean_states, dataset.clean_states)


def test_time_warp_rejects_invalid_scale() -> None:
    dataset = generate_circle(seed=72)

    with pytest.raises(ValueError, match="duration_scale"):
        time_warp(dataset, duration_scale=0.0)


def test_gaussian_noise_is_deterministic_and_keeps_clean_states() -> None:
    dataset = generate_circle(n_steps=40, seed=81)

    first = add_gaussian_noise(dataset, noise_std=0.1, seed=82)
    second = add_gaussian_noise(dataset, noise_std=0.1, seed=82)

    assert first.metadata["augmentation"] == "gaussian_noise"
    np.testing.assert_allclose(first.observations, second.observations)
    np.testing.assert_allclose(first.clean_states, dataset.clean_states)
    assert not np.allclose(first.observations, dataset.observations)
    assert np.all(np.isfinite(first.observations))


def test_missing_segment_mask_is_finite_and_records_bounds() -> None:
    dataset = generate_circle(n_steps=50, seed=91)

    masked = mask_missing_segment(dataset, fraction=0.2, seed=92, mask_value=-1.0)

    start = masked.metadata["missing_start"]
    stop = masked.metadata["missing_stop"]
    assert masked.metadata["augmentation"] == "missing_segment_mask"
    assert stop > start
    assert stop - start == 10
    np.testing.assert_allclose(masked.observations[start:stop], -1.0)
    np.testing.assert_allclose(masked.clean_states, dataset.clean_states)
    assert np.all(np.isfinite(masked.observations))


def test_missing_segment_mask_rejects_nan_mask_value() -> None:
    dataset = generate_circle(seed=93)

    with pytest.raises(ValueError, match="mask_value"):
        mask_missing_segment(dataset, fraction=0.2, mask_value=np.nan)
