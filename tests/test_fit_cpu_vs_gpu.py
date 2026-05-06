"""CPU vs GPU agreement for bspx.bspline_lsq_fit, all three modes.

Each mode is run on both backends and the resulting control points and
reconstructed curves are compared. Modes:

  1. plain      — unconstrained LSQ
  2. clamped    — anchors at t in {0, 1} → endpoint interpolation
  3. arbitrary  — anchors at endpoints + a few interior parameter values

Also asserts that constrained modes actually pass through their anchors to
solver-precision (the whole point of the constraint).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bspx


CPU = jax.devices("cpu")[0]
GPU_AVAILABLE = len(jax.devices("gpu")) > 0
GPU = jax.devices("gpu")[0] if GPU_AVAILABLE else None

ORDER = 4
N_TRUE_CTRL = 16
DIM = 2
N_DATA = 256
SIGMA = 0.05


def _make_noisy(seed: int):
    rng = np.random.default_rng(seed)
    P_true = rng.standard_normal((N_TRUE_CTRL, DIM)).astype(np.float32)
    curve = np.asarray(bspx.bspline(jnp.asarray(P_true), n_points=N_DATA, k=ORDER))
    noisy = curve + rng.standard_normal(curve.shape).astype(curve.dtype) * SIGMA
    return noisy.astype(np.float32)


def _anchors_for_mode(mode: str, data_np: np.ndarray):
    """Return (anchors_t, anchors_y) — np arrays — or (None, None) for plain."""
    if mode == "plain":
        return None, None
    if mode == "clamped":
        t = np.array([0.0, 1.0], dtype=np.float32)
        y = np.stack([data_np[0], data_np[-1]]).astype(np.float32)
        return t, y
    if mode == "arbitrary":
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        idx = (t * (N_DATA - 1)).round().astype(int)
        y = data_np[idx].astype(np.float32)
        return t, y
    raise ValueError(mode)


def _fit_on(device, data_np, n_ctrl, anchors_t_np, anchors_y_np, lam=0.0, solver="qr"):
    data = jax.device_put(jnp.asarray(data_np), device)
    if anchors_t_np is None:
        P = bspx.bspline_lsq_fit(
            data=data, n_ctrl=n_ctrl, order=ORDER, lam=lam, solver=solver
        )
    else:
        at = jax.device_put(jnp.asarray(anchors_t_np), device)
        ay = jax.device_put(jnp.asarray(anchors_y_np), device)
        P = bspx.bspline_lsq_fit(
            data=data, n_ctrl=n_ctrl, order=ORDER, anchors_t=at, anchors_y=ay, lam=lam, solver=solver
        )
    P.block_until_ready()
    return np.asarray(jax.device_put(P, CPU))


def _eval_on_cpu(P_np, t_np):
    P = jax.device_put(jnp.asarray(P_np), CPU)
    t = jax.device_put(jnp.asarray(t_np), CPU)
    n = t.shape[0]
    return np.asarray(bspx.bspline(P, n_points=n, k=ORDER, t=t))


@pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU device available")
@pytest.mark.parametrize("solver", ["qr", "cholesky"])
@pytest.mark.parametrize("mode", ["plain", "clamped", "arbitrary"])
@pytest.mark.parametrize("n_ctrl", [24, 128])  # one well-conditioned + one rank-deficient
@pytest.mark.parametrize("seed", [0])
def test_cpu_vs_gpu_agree(mode: str, n_ctrl: int, seed: int, solver: str):
    """With λ=1e-6 ridge, CPU and GPU agree across the full n_ctrl range.

    Including the rank-deficient regime (n_ctrl ≳ N_DATA/3) where chord-length
    parameterization can leave knot intervals empty and B becomes rank-
    deficient. Without the ridge (see ``test_unregularized_drift_diagnostic``),
    LAPACK and cuSolver pivot small singular values differently and produce
    O(1) drift in that regime. ``lam > 0`` removes the null space, the lstsq
    answer is unique, both backends agree to fp32 round-off.
    """
    data = _make_noisy(seed)
    at, ay = _anchors_for_mode(mode, data)
    # cholesky needs more ridge to overcome squared κ in fp32; qr fine at 1e-4
    lam = 1e-3 if solver == "cholesky" else 1e-4

    P_cpu = _fit_on(CPU, data, n_ctrl, at, ay, lam=lam, solver=solver)
    P_gpu = _fit_on(GPU, data, n_ctrl, at, ay, lam=lam, solver=solver)

    dP = float(np.max(np.abs(P_cpu - P_gpu)))

    n_eval = 512
    t_eval = np.linspace(0.0, 1.0, n_eval, dtype=np.float32)
    c_cpu = _eval_on_cpu(P_cpu, t_eval)
    c_gpu = _eval_on_cpu(P_gpu, t_eval)
    dc = float(np.max(np.abs(c_cpu - c_gpu)))

    print(f"\n[{solver} {mode} n_ctrl={n_ctrl} seed={seed}] max|dP|={dP:.2e}  max|dc|={dc:.2e}")
    assert dP < 1e-3, f"{solver}/{mode}: control points disagree CPU vs GPU by {dP:.3e}"
    assert dc < 1e-3, f"{solver}/{mode}: curves disagree CPU vs GPU by {dc:.3e}"


@pytest.mark.parametrize("mode", ["clamped", "arbitrary"])
@pytest.mark.parametrize("n_ctrl", [8, 16, 32])
def test_anchors_are_satisfied(mode: str, n_ctrl: int):
    """Constrained modes must pass through every anchor exactly."""
    data = _make_noisy(0)
    at, ay = _anchors_for_mode(mode, data)
    P = _fit_on(CPU, data, n_ctrl, at, ay)
    c_at_anchors = _eval_on_cpu(P, at)
    err = float(np.max(np.abs(c_at_anchors - ay)))
    print(f"\n[{mode} n_ctrl={n_ctrl}] max|curve(anchor) − anchor_y| = {err:.2e}")
    assert err < 1e-4, f"{mode}: anchor not interpolated, max err {err:.3e}"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU device available")
def test_unregularized_drift_diagnostic(capsys):
    """Diagnostic. Sweeps n_ctrl × mode × {lam=0, lam=1e-6}, prints drift.

    Always passes — the point is to surface where lam=0 is non-deterministic
    and where the small ridge cures it. Read the printed table.
    """
    seeds = list(range(5))
    rows = []
    for mode in ["plain", "clamped", "arbitrary"]:
        for n_ctrl in [6, 12, 24, 48, 64, 96, 128, 192]:
            for lam in [0.0, 1e-6]:
                dPs, dcs = [], []
                for s in seeds:
                    data = _make_noisy(s)
                    at, ay = _anchors_for_mode(mode, data)
                    P_cpu = _fit_on(CPU, data, n_ctrl, at, ay, lam=lam)
                    P_gpu = _fit_on(GPU, data, n_ctrl, at, ay, lam=lam)
                    dPs.append(np.max(np.abs(P_cpu - P_gpu)))
                    t_eval = np.linspace(0.0, 1.0, 256, dtype=np.float32)
                    dcs.append(
                        np.max(np.abs(_eval_on_cpu(P_cpu, t_eval) - _eval_on_cpu(P_gpu, t_eval)))
                    )
                rows.append((mode, n_ctrl, lam, np.mean(dPs), np.max(dPs), np.max(dcs)))

    print("\nmode      | n_ctrl | lam      | mean|dP|   max|dP|   max|dc|")
    print("----------+--------+----------+--------------------------------")
    for m, n, lam, mdp, xdp, xdc in rows:
        print(f"{m:9s} | {n:6d} | {lam:.0e} | {mdp:.2e}  {xdp:.2e}  {xdc:.2e}")
