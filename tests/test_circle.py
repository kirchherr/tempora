import numpy as np
import pytest

from tempora.data import generate_circle


def test_circle_shapes_metadata_and_finite_values() -> None:
    dataset = generate_circle(n_steps=64, duration=3.0, noise_std=0.01, seed=7)

    assert dataset.times.shape == (64,)
    assert dataset.observations.shape == (64, 2)
    assert dataset.clean_states.shape == (64, 2)
    assert dataset.metadata["system"] == "circle"
    assert dataset.metadata["seed"] == 7
    assert dataset.metadata["observation_shape"] == (64, 2)
    assert np.all(np.isfinite(dataset.times))
    assert np.all(np.isfinite(dataset.observations))
    assert np.all(np.isfinite(dataset.clean_states))
    assert np.all(np.diff(dataset.times) > 0.0)


def test_circle_seed_reproducibility() -> None:
    first = generate_circle(n_steps=32, noise_std=0.05, seed=11)
    second = generate_circle(n_steps=32, noise_std=0.05, seed=11)
    other = generate_circle(n_steps=32, noise_std=0.05, seed=12)

    np.testing.assert_allclose(first.times, second.times)
    np.testing.assert_allclose(first.observations, second.observations)
    np.testing.assert_allclose(first.clean_states, second.clean_states)
    assert not np.allclose(first.observations, other.observations)


def test_circle_rejects_invalid_radius() -> None:
    with pytest.raises(ValueError, match="radius"):
        generate_circle(radius=0.0)
