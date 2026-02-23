"""Top-level package for bspx."""

from jaxtyping import install_import_hook

##---------------Uncomment for runtime type checking --------
# with install_import_hook("bspx", "beartype.beartype"):
#     from .bsplines import BSpline
#     from .utils_static import get_knots_static
###---------------Use this for no runtime type checking (faster) --------
from .bsplines import bspline
from .utils_static import get_knots_static

__all__ = ("bspline", "get_knots_static")
__version__ = "0.0.0"
