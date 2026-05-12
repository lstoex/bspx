"""A module for evaluating B-spline curves using de Boor's algorithm. This is where the user interface lives."""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from jaxtyping import Array, Float

from ._typing import ControlPoints, CurvePoints, KnotVector, Order, Time
from .deboor import differentiate, propagate
from .utils import build_alpha_lut, clamped_uniform_knot_vector, get_indices_nonuniform, get_indices_uniform


@jax.jit(static_argnames=["k", "n_points"])
def bspline(
    control_points: ControlPoints,
    n_points: int,
    k: Order,
    T: KnotVector | None = None,
    t: Time | None = None,
) -> CurvePoints:
    """Evaluate a B-spline curve using de Boor's algorithm.
    Args:
        control_points: An array of shape (np1, d) representing the control points of the B-spline curve.
        n_points: The number of output points to generate on the B-spline curve.
        k: The order of the B-spline curve (default is 4 for cubic).
        T: Optional knot vector of shape (n_in + k + 1,). If not provided, a uniform knot vector will be used.
        t: Optional array of shape (n_points,) representing the parameter values at which to evaluate the B-spline curve. If not provided, uniform parameter values will be used.
    Returns:
        An array of shape (n_points, d) representing the points on the B-spline curve.
    """

    n = control_points.shape[0] - 1
    if T is None and t is None:
        with jax.ensure_compile_time_eval():
            t_ = np.linspace(0.0, 1.0, n_points)
            T_ = clamped_uniform_knot_vector(n, k, use_jax=False)
            j = get_indices_uniform(n, k, t_)
            assert isinstance(j, np.ndarray) and isinstance(T_, np.ndarray) and isinstance(t_, np.ndarray)
            alphas_flat = build_alpha_lut(k, j, T_, t_)
        return propagate(control_points, k, j, alphas_flat)
    elif T is not None and t is not None:
        j = get_indices_nonuniform(n, T, t)
        return propagate(control_points, k, j, None, T, t)
    elif T is None and t is not None:
        T_ = clamped_uniform_knot_vector(n, k, use_jax=True)
        j = get_indices_nonuniform(n, T_, t)
        return propagate(control_points, k, j, None, T_, t)
    else:
        raise ValueError("Both T and t or just t must be provided.")


@jax.jit(static_argnames=["k", "n_points", "n_fine"])
def bspline_arclength_adjusted(
    control_points: ControlPoints,
    n_points: int,
    k: Order,
    n_fine: int | None = None,
) -> tuple[CurvePoints, Time]:
    """B-spline sampled at approximately uniform arc-length spacing.

    Dense pass estimates arc length, then re-evaluates at re-parameterized t.
    Returns ``(curve, t)``. ``n_fine`` defaults to ``n_points``.
    """
    n_fine = n_fine or n_points
    c_fine = bspline(control_points, n_points=n_fine, k=k)
    segments = jnp.linalg.norm(jnp.diff(c_fine, axis=0), axis=-1)
    cum_len = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(segments)])
    t_old = jnp.linspace(0.0, 1.0, n_fine)
    cum_len_norm = cum_len / cum_len[-1]
    t_new = jnp.interp(jnp.linspace(0.0, 1.0, n_points), cum_len_norm, t_old)
    t_new = jax.lax.stop_gradient(t_new)  # block gradients through reparam
    return bspline(control_points, n_points=n_points, k=k, t=t_new), t_new


def chord_length_params(
    data: Float[Array, "n d"],
) -> Time:
    """Chord-length parameterization of data points → t in [0, 1]."""
    diffs = jnp.diff(data, axis=0)
    seg_lengths = jnp.sqrt(jnp.sum(diffs**2, axis=-1))
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg_lengths)])
    total = cum[-1]
    return jnp.where(total > 0, cum / total, jnp.linspace(0.0, 1.0, data.shape[0]))


def _ridge_solve(M, b, lam: float, solver: str = "qr"):
    """Solve ``min ‖M x − b‖² + λ‖x‖²`` with M tall (rows ≥ cols).

    Two backends:

    * ``solver="qr"`` (default, accurate). Householder QR on M (lam=0) or on
      the augmented design ``[M; √λ I]`` (lam>0) → triangular solve. κ stays
      linear in σ_max(M)/√λ.

    * ``solver="cholesky"`` (fast, requires lam>0). Cholesky on the small
      ``n_ctrl × n_ctrl`` shifted normal equations ``MᵀM + λI``. Banded
      structure of B-spline ``BᵀB`` keeps the matrix small and the factor
      cheap — empirically 2-12× faster than QR on GPU. Squares conditioning
      (κ → κ²+1/λ), so fp32 precision floor is ~√(κ²·ε² + ε/λ). Fine for
      ``λ ≳ 1e-3``; not safe for ``λ < 1e-4`` at high n_ctrl.
    """
    if solver == "cholesky":
        if lam <= 0.0:
            raise ValueError("solver='cholesky' requires lam > 0 for positive-definiteness.")
        n = M.shape[1]
        lam_d = jnp.asarray(lam, dtype=M.dtype)
        A = M.T @ M + lam_d * jnp.eye(n, dtype=M.dtype)
        L = jnp.linalg.cholesky(A)
        return jsl.cho_solve((L, True), M.T @ b)
    if solver != "qr":
        raise ValueError(f"solver must be 'qr' or 'cholesky', got {solver!r}")
    if lam > 0.0:
        n = M.shape[1]
        sqrt_lam = jnp.sqrt(jnp.asarray(lam, dtype=M.dtype))
        M = jnp.concatenate([M, sqrt_lam * jnp.eye(n, dtype=M.dtype)], axis=0)
        zeros = jnp.zeros((n,) + b.shape[1:], dtype=b.dtype)
        b = jnp.concatenate([b, zeros], axis=0)
    Q, R = jnp.linalg.qr(M, mode="reduced")
    return jsl.solve_triangular(R, Q.T @ b, lower=False)


@jax.jit(static_argnames=["n_ctrl", "order", "lam", "solver"])
def bspline_lsq_fit(
    *,
    data: Float[Array, "n d"],
    n_ctrl: int,
    order: Order = 4,
    anchors_t: Float[Array, " m"] | None = None,
    anchors_y: Float[Array, "m d"] | None = None,
    lam: float = 0.0,
    solver: str = "qr",
) -> ControlPoints:
    """Fit a B-spline with ``n_ctrl`` control points to ``data``.

    Solves ``min ‖B P − data‖² + λ‖P‖²`` (chord-length-parameterized ``B``),
    optionally subject to equality constraints ``C P = anchors_y`` where
    ``C[i,:] = N(anchors_t[i])``. Four orthogonal knobs:

    * ``anchors_t=None`` → plain unconstrained LSQ. Endpoints float by O(noise).
    * Endpoint clamp → ``anchors_t=jnp.array([0., 1.])``,
      ``anchors_y=jnp.stack([data[0], data[-1]])``.
    * Arbitrary anchors → any ``(t, y)`` pairs (interior, endpoint, or mix).
    * ``lam > 0`` → Tikhonov ridge. **Required for stable cross-device results
      when n_ctrl approaches n_data**: chord-length parameterization can leave
      knot intervals empty (Schoenberg–Whitney violated) → B rank-deficient →
      lstsq returns a different min-norm element of the null space per
      LAPACK/cuSolver pivot order. ``λ ≈ 1e-6`` is enough to make the result
      deterministic at the cost of an O(λ/σ²) shrinkage on well-determined
      modes. Default ``0.0`` preserves bias-free behavior in the
      well-conditioned regime.

    Algorithm: see ``_ridge_solve``. ``solver="qr"`` (default) uses
    Householder QR on the (possibly augmented) design — accurate, works at
    ``lam=0``. ``solver="cholesky"`` solves the small ``n_ctrl × n_ctrl``
    shifted normal equations — 2-12× faster on GPU, requires ``lam>0``,
    fp32-safe only for ``lam ≳ 1e-3``. Constrained mode uses the null-space
    method: ``P = P_part + Q_2 y`` where ``Q_2`` spans null(C); the reduced
    LSQ for ``y`` is solved by the chosen backend. Anchors enforced exactly.
    """
    if (anchors_t is None) != (anchors_y is None):
        raise ValueError("anchors_t and anchors_y must be both given or both None.")

    n_data = data.shape[0]
    t_data = chord_length_params(data)
    eye_ctrl = jnp.eye(n_ctrl)
    B = bspline(eye_ctrl, n_points=n_data, k=order, t=t_data)

    if anchors_t is None:
        return _ridge_solve(B, data, lam, solver)

    n_anchors = anchors_t.shape[0]
    C = bspline(eye_ctrl, n_points=n_anchors, k=order, t=anchors_t)

    Q, R = jnp.linalg.qr(C.T, mode="complete")
    R_top = R[:n_anchors, :]
    Q1 = Q[:, :n_anchors]
    Q2 = Q[:, n_anchors:]

    v = jsl.solve_triangular(R_top.T, anchors_y, lower=True)
    P_part = Q1 @ v

    y = _ridge_solve(B @ Q2, data - B @ P_part, lam, solver)
    return P_part + Q2 @ y


@jax.jit(static_argnames=["n_points", "k", "n_fine"])
def bspline_arclength_subrange(
    *,
    control_points: ControlPoints,
    n_points: int,
    k: Order,
    t_lo: Float[Array, ""],
    t_hi: Float[Array, ""],
    n_fine: int | None = None,
) -> CurvePoints:
    """B-spline with arc-length spacing over ``[t_lo, t_hi]``. ``n_fine`` defaults to ``n_points * 4``."""
    if n_fine is None:
        n_fine = n_points * 4

    t_fine = t_lo + (t_hi - t_lo) * jnp.linspace(0.0, 1.0, n_fine)
    c_fine = bspline(control_points, n_points=n_fine, k=k, t=t_fine)

    seg = jnp.sqrt(jnp.sum(jnp.diff(c_fine, axis=0) ** 2, axis=-1))
    cum_arc = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
    cum_arc_norm = cum_arc / jnp.maximum(cum_arc[-1], 1e-12)

    t_uniform = jnp.interp(jnp.linspace(0.0, 1.0, n_points), cum_arc_norm, t_fine)
    t_uniform = jax.lax.stop_gradient(t_uniform)
    return bspline(control_points, n_points=n_points, k=k, t=t_uniform)


@jax.jit(static_argnames=["k", "n_points", "derivative_order", "emit_intermediates"])
def bspline_derivative(
    control_points: ControlPoints,
    n_points: int,
    k: Order,
    T: KnotVector | None = None,
    t: Time | None = None,
    derivative_order: int = 1,
    emit_intermediates=False,
) -> list[CurvePoints] | CurvePoints:
    """Evaluate the derivative of a B-spline curve using de Boor's algorithm.
    Args:
        control_points: An array of shape (n+1, d) representing the control points of the B-spline curve.
        k: The order of the B-spline curve (default is 4 for cubic).
        T: Optional knot vector of shape (n + k + 1,). If not provided, a uniform knot vector will be used.
        t: Optional array of shape (n_points,) representing the parameter values at which to evaluate the B-spline curve. If not provided, uniform parameter values will be used.
        n_points: The number of output points to generate on the B-spline curve.
        derivative_order: The order of the derivative to compute (default is 1 for first derivative).
    Returns:
        An array of shape (n_points, d) representing the points on the derivative of the B-spline curve.
    """

    n = control_points.shape[0] - 1
    T_ = clamped_uniform_knot_vector(n, k) if T is None else T
    t_ = jnp.linspace(0.0, 1.0, n_points) if t is None else t
    k_ = k
    Q = control_points

    def assemble(Q, k_, T):
        return bspline(Q, n_points, k_, T=T_, t=t_)

    out = []
    for _ in range(derivative_order):
        Q, k_, T_ = differentiate(Q, k_, T_)
        if emit_intermediates:
            out.append(assemble(Q, k_, T))
    out = assemble(Q, k_, T) if not emit_intermediates else out
    return out
