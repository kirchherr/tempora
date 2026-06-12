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
