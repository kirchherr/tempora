import numpy as np
import pytest

from tempora.data import generate_torus


def test_torus_shapes_metadata_and_finite_values() -> None:
    dataset = generate_torus(n_steps=72, duration=4.0, noise_std=0.01, seed=13)

    assert dataset.times.shape == (72,)
    assert dataset.observations.shape == (72, 3)
    assert dataset.clean_states.shape == (72, 3)
    assert dataset.metadata["system"] == "torus"
    assert dataset.metadata["seed"] == 13
    assert dataset.metadata["major_radius"] > dataset.metadata["minor_radius"]
    assert np.all(np.isfinite(dataset.observations))
    assert np.all(np.isfinite(dataset.clean_states))
    assert np.all(np.diff(dataset.times) > 0.0)


def test_torus_seed_reproducibility() -> None:
    first = generate_torus(n_steps=40, noise_std=0.02, seed=21)
    second = generate_torus(n_steps=40, noise_std=0.02, seed=21)

    np.testing.assert_allclose(first.times, second.times)
    np.testing.assert_allclose(first.observations, second.observations)
    np.testing.assert_allclose(first.clean_states, second.clean_states)


def test_torus_rejects_degenerate_radii() -> None:
    with pytest.raises(ValueError, match="minor_radius"):
        generate_torus(major_radius=1.0, minor_radius=1.0)
