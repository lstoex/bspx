"""Utilities for B-spline evaluation."""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int


def get_indices_uniform(
    n: int, k: int, t: Float[Array | np.ndarray, " n_points"]
) -> Int[Array | np.ndarray, " n_points"]:
    """Compute the knot span index j for a uniform clamped knot vector.
    For uniform knots the interior spans are evenly spaced, so j is just a scaled floor.
    The clamped knot vector has k repeated knots at each end, so valid span indices run from k-1 to n (inclusive).
    Args:
        n: number of control points - 1
        k: order of the B-spline
        t: evaluation parameters, shape (n_points,)
    Returns:
        j: knot span indices for each evaluation point, shape (n_points,)
    """
    lib = jnp if isinstance(t, jnp.ndarray) else np
    n_spans = n - k + 2
    raw = lib.floor(t * n_spans).astype(lib.int32)
    j = lib.clip(raw + (k - 1), k - 1, n)
    return j


def get_indices_nonuniform(
    n: int, T: Float[Array | np.ndarray, " mp1"], t: Float[Array | np.ndarray, " n_points"]
) -> Int[Array | np.ndarray, " n_points"]:
    """Compute the knot span index j for a non-uniform (but strictly increasing!) knot vector T and parameter t.
    Args:
        n: number of control points - 1
        T: knot vector of length m+1, where m = n + k
        t: evaluation parameters, shape (n_points,)
    Returns:
        j: knot span indices for each evaluation point, shape (n_points,)

    """
    # For each t, find the largest j such that T[j] <= t. This is the knot span index.
    lib = jnp if isinstance(t, jnp.ndarray) else np
    m = T.shape[0] - 1
    k = m - n  # order of the B-spline
    j = lib.searchsorted(T, t, side="right") - 1
    # Clamp to valid range [k-1, n]
    j = lib.clip(j, k - 1, n)
    return j


def clamped_uniform_knot_vector(n: int, k: int, use_jax=False) -> Float[np.ndarray | Array, " {n+k+1}"]:
    """Generate a clamped uniform knot vector for n+1 control points and order k.
    Args:
        n: number of control points - 1
        k: order of the B-spline
        use_jax: whether to return a JAX array instead of a NumPy array
    Returns:
        T: knot vector of length m+1=n+k+1
    """
    assert n + 1 >= k, "Need at least k control points for order k"

    n_interior = n + 1 - k
    lib = jnp if use_jax else np
    if n_interior == 0:
        interior = lib.array([])
    else:
        interior = lib.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    T = lib.concatenate(
        [
            lib.zeros(k),
            interior,
            lib.ones(k),
        ]
    )
    return T


def compute_alpha(
    k: int,
    r: int,
    i: int,
    j: Int[np.ndarray | Array, " n_points"],
    T: Float[np.ndarray | Array, " mp1"],
    t: Float[np.ndarray | Array, " n_points"],
) -> Float[np.ndarray | Array, "..."]:
    """Compute the alpha value for the r-th iteration and i-th control point in de Boor's algorithm.
    Args:
        k: order of the B-spline
        r: current iteration (1-based)
        i: current control point index (0-based)
        j: knot span indices for each evaluation point, shape (n_points,)
        T: knot vector, shape (m+1,)
        t: evaluation parameters, shape (n_points,)
    Returns:
        alpha: shape (n_points,)
    """
    numerator = t - T[j - k + 1 + i]
    denominator = T[j + i - r + 1] - T[j - k + 1 + i]
    alpha = (
        np.where(np.isclose(denominator, 0.0), 0.0, numerator / denominator)
        if isinstance(T, np.ndarray)
        else jnp.where(jnp.isclose(denominator, 0.0), 0.0, numerator / denominator)
    )
    return alpha


def build_alpha_lut(
    k: int, j: Int[np.ndarray, " n_points"], T: Float[np.ndarray, " mp1"], t: Float[np.ndarray, " n_points"]
) -> Float[np.ndarray, "n_points {(k-1)*k//2}"]:
    """Build a lookup table for the alphas used in de Boor's algorithm for uniform B-splines. Since the LUT is triangular, we save it as a row first flat array with size (k+1)*k//2.
    Args:
        k: order of the B-spline
        j: knot span indices for each evaluation point, shape (n_points,)
        T: knot vector, shape (m+1,)
        t: evaluation parameters, shape (n_points,)
    Returns:
        alphas: shape (n_points, (k-1)*k//2)
    """
    alphas = []
    for r in range(1, k):
        for i in range(r, k):
            alpha_r_i = compute_alpha(k, r, i, j, T, t)
            alphas.append(alpha_r_i)
    return np.stack(alphas, axis=-1)  # shape (n_points, n_alphas_per_point)


def flat_triangular_index(
    i: Int[Array | np.ndarray, "..."] | int, j: Int[Array | np.ndarray, "..."] | int, size: int
) -> Int[Array | np.ndarray, "..."] | int:
    """Convert 2D upper-left triangular indices to flat index.
    Args:
        i: Row index (0-based)
        j: Column index (must be size-1 - j > i)
        size: Number of columns in the triangular matrix
    Returns:
        Flat index corresponding to the (i, j) position in the upper-left triangular matrix.
    """
    return i * size - (i * (i + 1)) // 2 + j
