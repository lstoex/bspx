"""Type aliases for B-spline arguments."""

from typing import NewType

import numpy as np
from jaxtyping import Array, Float, Int

# Scalar parameters
Order = NewType("Order", int)
"""B-spline order k (e.g. 4 for cubic)."""

NumCtrl = NewType("NumCtrl", int)
"""Number of control points minus 1 (n)."""

NPoints = NewType("NPoints", int)
"""Number of evaluation points."""

# Array types — dual numpy/JAX support
ControlPoints = Float[Array | np.ndarray, "np1 d"]
"""Control points of shape (n+1, d)."""

KnotVector = Float[Array | np.ndarray, " mp1"]
"""Knot vector of length m+1 = n+k+1."""

Time = Float[Array | np.ndarray, " n_points"]
"""Parameter values t in [0, 1], shape (n_points,)."""

SpanIndices = Int[Array | np.ndarray, " n_points"]
"""Knot span indices j, shape (n_points,)."""

AlphaLUT = Float[np.ndarray, "n_points n_alphas"]
"""Precomputed alpha lookup table."""

CurvePoints = Float[Array, "n_points d"]
"""Evaluated points on a B-spline curve."""
