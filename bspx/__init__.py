"""Top-level package for bspx."""

# ---------------Uncomment for runtime type checking --------
from jaxtyping import install_import_hook

# with install_import_hook("bspx", "beartype.beartype"):
#     from .curves import bspline, bspline_derivative
#     from .utils import clamped_uniform_knot_vector
###---------------Use this for no runtime type checking (faster) --------
from .curves import (
    bspline,
    bspline_arclength_adjusted,
    bspline_arclength_subrange,
    bspline_derivative,
    bspline_lsq_fit,
    chord_length_params,
)
from .utils import clamped_uniform_knot_vector, greville_abscissae

__all__ = (
    "bspline",
    "bspline_arclength_adjusted",
    "bspline_arclength_subrange",
    "bspline_derivative",
    "bspline_lsq_fit",
    "chord_length_params",
    "clamped_uniform_knot_vector",
    "greville_abscissae",
)
__version__ = "0.0.0"
