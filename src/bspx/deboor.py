"""Variant of De Boor's algorithm for evaluating B-spline curves, with a compile-time precomputation."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from .utils import compute_alpha, flat_triangular_index


def init_d(P: Float[Array | np.ndarray, "np1 d"], j: Int[np.ndarray | Array, " n_points"], k: int):
    """Initialize the d array for de Boor's algorithm.
    For each evaluation point, we need to select k control points based on the index j.
    Args:
    P: Control points, shape (n+1, d)
    j: Knot span indices for each evaluation point, shape (n_points,)
    k: Order of the B-spline
    """
    if isinstance(j, np.ndarray):
        i = np.arange(k)
        return P[j[:, None] - k + 1 + i]  # shape (n_points, k)
    elif isinstance(j, jnp.ndarray):
        start_P = j - k + 1  # shape (n_points,)

        def index(start):
            return jax.lax.dynamic_slice_in_dim(P, start, k, axis=0)  # shape (k, d)

        return jax.vmap(index)(start_P)  # shape (n_points, k


def propagate(
    P: Float[Array | np.ndarray, "np1 d"],
    k: int,
    j: Int[np.ndarray | Array, " n_points"],
    alphas_precomputed_flat: Float[np.ndarray, "n_points n_alphas"] | None,
    T: Float[Array | np.ndarray, " mp1"] | None = None,
    t: Float[Array | np.ndarray, " n_points"] | None = None,
):
    d = init_d(P, j, k)  # shape (n_points, k)
    for r in range(1, k):
        for i in range(k - 1, r - 1, -1):
            a = r - 1
            b = i - 1
            if T is not None and t is not None and alphas_precomputed_flat is None:
                alpha = compute_alpha(k, r, i, j, T, t)
            elif alphas_precomputed_flat is not None and T is None and t is None:
                alpha = alphas_precomputed_flat[:, flat_triangular_index(a, b, k - 1)]
            else:
                raise ValueError("Either T and t or alphas_flat must be provided.")
            if isinstance(d, np.ndarray):
                d[:, i] = (1.0 - alpha[..., None]) * d[:, i - 1] + alpha[..., None] * d[:, i]
            else:
                d = d.at[:, i].set((1.0 - alpha[..., None]) * d[:, i - 1] + alpha[..., None] * d[:, i])
    return d[:, k - 1]


def differentiate(
    P: Float[Array | np.ndarray, "np1 d"], k: int, T: Float[Array | np.ndarray, " np1+{k}"]
) -> tuple[Float[Array | np.ndarray, "n d"], int, Float[Array | np.ndarray, " n+{k}-1"]]:
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
    lib = np if isinstance(P, np.ndarray) else jnp
    Q = lib.diff(P, axis=0) * (k - 1)
    denom = (T[k : n + k] - T[1 : n + 1])[:, None]  # shape (n, 1) for broadcasting
    Q = lib.where(lib.isclose(denom, 0.0), lib.zeros_like(Q), Q / denom)
    return Q, k - 1, T[1:-1]
