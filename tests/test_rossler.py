import numpy as np
import pytest

from tempora.data import generate_rossler


def test_rossler_shapes_metadata_and_finite_values() -> None:
    dataset = generate_rossler(n_steps=80, duration=4.0, noise_std=0.001, seed=51)

    assert dataset.times.shape == (80,)
    assert dataset.observations.shape == (80, 3)
    assert dataset.clean_states.shape == (80, 3)
    assert dataset.metadata["system"] == "rossler"
    assert dataset.metadata["integrator"] == "scipy.solve_ivp"
    assert dataset.metadata["seed"] == 51
    assert np.all(np.isfinite(dataset.observations))
    assert np.all(np.isfinite(dataset.clean_states))
    assert np.all(np.diff(dataset.times) > 0.0)


def test_rossler_seed_reproducibility() -> None:
    first = generate_rossler(n_steps=50, duration=2.0, noise_std=0.01, seed=61)
    second = generate_rossler(n_steps=50, duration=2.0, noise_std=0.01, seed=61)

    np.testing.assert_allclose(first.times, second.times)
    np.testing.assert_allclose(first.observations, second.observations)
    np.testing.assert_allclose(first.clean_states, second.clean_states)


def test_rossler_rejects_invalid_noise() -> None:
    with pytest.raises(ValueError, match="noise_std"):
        generate_rossler(noise_std=-0.1)
