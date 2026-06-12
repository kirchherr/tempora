# Theorem 03: Empirical Persistence-Diagram Comparison

## Statement

For finite point clouds sampled from controlled input and latent trajectories,
TEMPORA can compute persistent homology summaries and compare them with
bottleneck or Wasserstein distances when the optional TDA dependencies are
available.

## Assumptions

- Point clouds are finite arrays with shape `(n_points, point_dim)`.
- All point-cloud values are finite.
- Vietoris-Rips persistence is computed with `ripser`.
- Diagram distances are computed with `persim`.
- Distances are evaluated on selected homology degrees such as H0 and H1.

## Proof Sketch

This is an implementation correspondence claim, not a new mathematical theorem.
Given finite point clouds, persistent homology algorithms produce persistence
diagrams under their documented numerical assumptions. Standard diagram
distances then provide empirical distances between these summaries.

## Implementation Correspondence

- `compute_persistence_diagrams` validates finite point clouds and calls
  `ripser`.
- `persistence_distance` computes bottleneck or Wasserstein distances through
  `persim`.
- `evaluate_topology_pair` reports H0/H1 distances and dominant finite
  lifetimes for input and latent point clouds.

## Empirical Checks

The tests verify that:

- a clean circle has a dominant H1 feature,
- Gaussian noise changes persistence distances,
- an orthogonal transform of the same point cloud has near-zero diagram
  distance,
- missing optional dependencies raise clear errors.

## Limitations

Similar persistence diagrams do not prove semantic equivalence, homeomorphism,
or robust real-world understanding. TDA metrics are empirical summaries of
finite point clouds and can be sensitive to sampling density, noise, filtration
choice, and numerical tolerance.

## Related Tests

- `tests/test_tda_metrics.py`

