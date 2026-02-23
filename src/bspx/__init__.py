"""Top-level package for bspx."""

##---------------Uncomment for runtime type checking --------
# from jaxtyping import install_import_hook
# with install_import_hook("bspx", "beartype.beartype"):
#     from .curves import bspline, bspline_derivative
#     from .utils_static import get_knots_static
###---------------Use this for no runtime type checking (faster) --------
from .curves import bspline, bspline_derivative
from .utils_static import get_knots_static

__all__ = ("bspline", "get_knots_static", "bspline_derivative")
__version__ = "0.0.0"
