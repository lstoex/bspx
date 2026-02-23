"""Utilities for B-spline evaluation during compile time."""

import numpy as np
from jaxtyping import Float, Int


def get_knots_static(n: int, k: int) -> Float[np.ndarray, " {n+k+1}"]:
    assert n + 1 >= k, "Need at least k control points for order k"

    n_interior = n + 1 - k
    if n_interior == 0:
        interior = np.array([])
    else:
        interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    T = np.concatenate(
        [
            np.zeros(k),
            interior,
            np.ones(k),
        ]
    )
    return T


def get_indices_static(n: int, k: int, t: Float[np.ndarray, "..."]) -> Int[np.ndarray, "..."]:
    n_spans = n - k + 2
    raw = np.floor(t * n_spans).astype(np.int32)
    j = np.clip(raw + (k - 1), k - 1, n)
    return j


def get_blending_weight_for_stage_static(
    k: int, r: int, i: int, j: np.int32, T: Float[np.ndarray, " mp1"], t: np.float64
) -> Float[np.ndarray, "..."]:
    numerator = t - T[j - k + 1 + i]
    denominator = T[j + i - r + 1] - T[j - k + 1 + i]
    alpha = np.where(denominator == 0.0, 0.0, numerator / denominator)
    return alpha


def get_alphas_static(
    k: int, j: np.int32, T: Float[np.ndarray, " mp1"], t: np.float64
) -> list[Float[np.ndarray, "..."]]:
    alphas = []
    for r in range(1, k):
        alpha_r = np.array([get_blending_weight_for_stage_static(k, r, i, j, T, t) for i in range(r, k)])
        alphas.append(alpha_r)
    return alphas


def precompute_aligned_alphas(
    n: int, k: int, ts: Float[np.ndarray, " n_points"]
) -> tuple[Float[np.ndarray, "n_points max_alpha_len km1"], Int[np.ndarray, " n_points"]]:
    """Precompute the alphas for all points and pad them to have the same shape for efficient indexing at runtime."""
    T = get_knots_static(n, k)
    js = get_indices_static(n, k, ts)  # precompute for n_points
    alphas = [get_alphas_static(k, j, T, t) for j, t in zip(js, ts)]
    alphas_i = alphas[0]  # all elements have the same shape, so we can just take the first one to get the shape
    max_alpha_len = max(len(alpha) for alpha in alphas_i) if alphas_i else 0
    alphas_padded = []
    for i, a in enumerate(alphas):
        alphas_padded.append([np.concatenate([a_, np.inf * np.ones((max_alpha_len - len(a_),))]) for a_ in a])
    return np.array(alphas_padded), js
