"""Variant of De Boor's algorithm for evaluating B-spline curves, with a compile-time precomputation."""

import jax
import numpy as np
from jaxtyping import Array, Float

from .utils import get_alphas, get_indices, get_relevant_points, make_uniform_knot_vector
from .utils_static import precompute_aligned_alphas


def de_boor_static(P: Float[Array, "np1 d"], k: int, n_points: int) -> Float[Array, " n_points d"]:
    """
    Evaluate a B-spline curve at n_points uniformly spaced parameter values in [0, 1].
    Computes as much as possible at compile time for maximum performance when t values are known at compile time.
    Args:
    P: control points, shape (n+1, d)
    k: order of the B-spline
    n_points: number of points to evaluate on the curve
    """
    with jax.ensure_compile_time_eval():
        n = P.shape[0] - 1
        alphas_padded, js = precompute_aligned_alphas(n, k, np.linspace(0.0, 1.0, n_points))

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


def de_boor(P: Float[Array, "np1 d"], k: int, ts: Float[Array, " n_points"]) -> Float[Array, " n_points d"]:
    """
    Evaluate a B-spline curve at the given parameter values ts.
    This version computes the blending weights at runtime, so it can handle dynamic t values.
    Args:
    P: control points, shape (n+1, d)
    k: order of the B-spline
    ts: parameter values to evaluate at, shape (n_points,)
    """
    n = P.shape[0] - 1
    T = make_uniform_knot_vector(n, k)

    def blend(t, P):
        j = get_indices(n, k, t)
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
