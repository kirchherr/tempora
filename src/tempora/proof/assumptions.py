"""Machine-readable assumption summaries for TEMPORA theorem checks."""

CONTRACTION_ASSUMPTIONS: tuple[str, ...] = (
    "D is diagonal and strictly positive.",
    "W is finite and square.",
    "The activation has documented Lipschitz constant L_sigma.",
    "The contraction statement is only with respect to latent state z.",
    "Inputs and bias are finite forcing terms.",
    "Spectral norms are numerical finite-precision estimates.",
    "Strict numerical claims use an explicit positive margin.",
)

LEARNING_STABILITY_ASSUMPTIONS: tuple[str, ...] = (
    "The Theorem 01 contraction assumptions hold before certification.",
    "Damping values remain strictly positive.",
    "The projection margin is positive and smaller than min(D).",
    "Plasticity updates and post-projection weights are finite.",
    "Projection is applied after every update that may alter W.",
    "The certificate concerns only stored post-projection recurrent weights.",
    "No stability claim is made about the unprojected update path.",
)

TOPOLOGY_COMPARISON_ASSUMPTIONS: tuple[str, ...] = (
    "Input and latent point clouds are finite arrays.",
    "All point-cloud values are finite.",
    "Persistence diagrams are computed by the configured TDA backend.",
    "Diagram distances are computed for an explicit homology degree.",
    "Certification uses an explicit maximum empirical diagram distance.",
    "The certificate is an empirical finite-sample comparison only.",
    "No semantic equivalence or homeomorphism claim is made.",
)
