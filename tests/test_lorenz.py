import numpy as np
import pytest

from tempora.data import generate_lorenz


def test_lorenz_shapes_metadata_and_finite_values() -> None:
    dataset = generate_lorenz(n_steps=80, duration=2.0, noise_std=0.001, seed=31)

    assert dataset.times.shape == (80,)
    assert dataset.observations.shape == (80, 3)
    assert dataset.clean_states.shape == (80, 3)
    assert dataset.metadata["system"] == "lorenz"
    assert dataset.metadata["integrator"] == "scipy.solve_ivp"
    assert dataset.metadata["seed"] == 31
    assert np.all(np.isfinite(dataset.observations))
    assert np.all(np.isfinite(dataset.clean_states))
    assert np.all(np.diff(dataset.times) > 0.0)


def test_lorenz_seed_reproducibility() -> None:
    first = generate_lorenz(n_steps=50, duration=1.0, noise_std=0.01, seed=41)
    second = generate_lorenz(n_steps=50, duration=1.0, noise_std=0.01, seed=41)

    np.testing.assert_allclose(first.times, second.times)
    np.testing.assert_allclose(first.observations, second.observations)
    np.testing.assert_allclose(first.clean_states, second.clean_states)


def test_lorenz_rejects_invalid_initial_state() -> None:
    with pytest.raises(ValueError, match="initial_state"):
        generate_lorenz(initial_state=(1.0, 2.0))
