"""Utilities for B-spline evaluation."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .utils_static import get_knots_static


def make_uniform_knot_vector(n: int, k: int) -> Float[Array, " {n+k+1}"]:
    """
    Build a uniform, CLAMPED knot vector for n+1 control points and order k.

    Structure:
      [ 0, ..., 0,  t_k, t_{k+1}, ..., t_{n},  1, ..., 1 ]
        k zeros        (n-k+1) interior knots        k ones

    m = n + k  ->  m+1 knots total.

    The interior knots are uniformly spaced in (0, 1).
    Number of interior knots = n + 1 - k  (zero when n+1 == k, i.e. Bezier).
    """
    return jnp.array(get_knots_static(n, k))


def get_relevant_points(index: Int[Array, ""], P: Float[Array, "np1 d"], k: int) -> Float[Array, "k d"]:
    """Get the k control points relevant for the span index j. Works on JAX tracers."""
    start_P = index - k + 1
    return jax.lax.dynamic_slice_in_dim(P, start_P, k, axis=0)


def get_blending_weight_for_stage(
    k: int, r: int, i: Int[Array, ""], j: Int[Array, ""], T: Float[Array, " mp1"], t: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Compute the blending weight alpha for the r-th blending stage and i-th point in the span,
    for a given span index j, knot vector T, and parameter t."""
    numerator = t - T[j - k + 1 + i]
    denominator = T[j + i - r + 1] - T[j - k + 1 + i]
    safe_denom = jnp.where(denominator == 0.0, 1.0, denominator)
    alpha = jnp.where(denominator == 0.0, 0.0, numerator / safe_denom)
    return alpha


def get_alphas(k: int, j: Int[Array, ""], T: Float[Array, "..."], t: Float[Array, "..."]) -> list[Float[Array, "..."]]:
    """Compute the blending weights for all stages of the de Boor algorithm for a given span index j,
    knot vector T, and parameter t."""
    alphas = []
    for r in range(1, k):
        alpha_r = jax.vmap(lambda i: get_blending_weight_for_stage(k, r, i, j, T, t))(jnp.arange(r, k))
        alphas.append(alpha_r)
    return alphas


def get_indices(n: int, k: int, t: Float[Array, "..."]) -> Int[Array, "..."]:
    """Compute the knot span index j for a uniform clamped knot vector.
    For uniform knots the interior spans are evenly spaced, so j is just a scaled floor.
    The clamped knot vector has k repeated knots at each end, so valid span indices run from k-1 to n (inclusive).
    """
    n_spans = n - k + 2
    raw = jnp.floor(t * n_spans).astype(jnp.int32)
    j = jnp.clip(raw + (k - 1), k - 1, n)
    return j
