"""Variant of De Boor's algorithm for evaluating B-spline curves, with a compile-time precomputation."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .utils import (
    get_alphas,
    get_indices_uniform,
    get_relevant_points,
    make_uniform_knot_vector,
    get_indices_nonuniform,
)
from .utils_static import precompute_aligned_alphas


def de_boor_static(P: Float[Array, "np1 d"], k: int, n_points: int) -> Float[Array, " {n_points} d"]:
    """
    Evaluate a B-spline curve at n_points uniformly spaced parameter values in [0, 1].
    Computes as much as possible at compile time for maximum performance when t values are known at compile time.
    Args:
    P: control points, shape (n+1, d)
    k: order of the B-spline
    n_points: number of points to evaluate at
    """
    with jax.ensure_compile_time_eval():
        n = P.shape[0] - 1
        ts = np.linspace(0, 1, n_points)
        alphas_padded, js = precompute_aligned_alphas(n, k, ts)

    # we only need to do the blending at runtime using the precomputed alphas and indices.
    def blend(j, a, P):
        d = get_relevant_points(j, P, k)
        for r_ in range(1, k):  # blending stages
            for i in range(
                k - 1, r_ - 1, -1
            ):  # blend in reverse order to avoid overwriting points we still need to read.
                alpha = a[r_ - 1, i - r_]
                d = d.at[i].set((1.0 - alpha) * d[i - 1] + alpha * d[i])
        return d[k - 1]

    return jax.vmap(blend, in_axes=(0, 0, None))(js, alphas_padded, P)


def de_boor(P: Float[Array, "np1 d"], k: int, n_points: int) -> Float[Array, " n_points d"]:
    """
    Evaluate a B-spline curve at the given parameter values ts.
    This version computes the blending weights at runtime, so it can handle dynamic t values.
    Args:
    P: control points, shape (n+1, d)
    k: order of the B-spline
    n_points: number of points to evaluate at
    """
    n = P.shape[0] - 1
    T = make_uniform_knot_vector(n, k)
    ts = jnp.linspace(0, 1, n_points)

    def blend(t, P):
        j = get_indices_uniform(n, k, t)
        a = get_alphas(k, j, T, t)
        d = get_relevant_points(j, P, k)
        for r_ in range(1, k):  # blending stages
            for i in range(
                k - 1, r_ - 1, -1
            ):  # blend in reverse order to avoid overwriting points we still need to read.
                alpha = a[r_ - 1][i - r_]
                d = d.at[i].set((1.0 - alpha) * d[i - 1] + alpha * d[i])
        return d[k - 1]

    return jax.vmap(blend, in_axes=(0, None))(ts, P)


def diff_spline(
    P: Float[Array, "np1 d"], k: int, T: Float[Array, " np1+{k}"]
) -> tuple[Float[Array, "n d"], int, Float[Array, " n+{k}-1"]]:
    """Compute the control points and knot vector for the derivative of a B-spline curve.
    Args:
        P: control points of the original B-spline curve, shape (n+1, d)
        k: order of the original B-spline curve
        T: knot vector of the original B-spline curve, shape (n+k+1,)
    Returns:
        Q: control points of the derivative B-spline curve, shape (n, d)
        k-1: order of the derivative B-spline curve
        T[1:n+k]: knot vector of the derivative B-spline curve, shape (n+k-1,)
    """
    assert k > 1, "Cannot compute derivative of a B-spline curve of order 1 or less."
    n = P.shape[0] - 1
    Q = jnp.diff(P, axis=0) * (k - 1)
    denom = T[k : n + k] - T[1 : n + 1]
    Q = jnp.where(jnp.allclose(denom, 0.0), 0.0, Q / denom[:, None])
    return Q, k - 1, T[1:-1]
