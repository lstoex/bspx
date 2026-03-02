import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import BSpline as SciPyBSpline

from bspx import bspline, bspline_derivative
from bspx.utils_static import get_knots_static

rng = np.random.default_rng(42)


@pytest.mark.parametrize(
    "n_in, order, n_output, use_ts",
    [
        (4, 3, 10, False),  # cubic Bezier, n_output (static path)
        (5, 4, 20, False),  # cubic B-spline
        (6, 5, 15, False),  # quartic B-spline
        (7, 6, 25, False),  # quintic B-spline
        (3, 2, 5, False),  # linear B-spline
        (4, 5, 10, False),  # invalid case: order > n_in + 1
        (4, 3, 10, True),  # cubic Bezier, ts (dynamic path)
        (5, 4, 20, True),  # cubic B-spline
        (6, 5, 15, True),  # quartic B-spline
        (7, 6, 25, True),  # quintic B-spline
        (3, 2, 5, True),  # linear B-spline
        (4, 5, 10, True),  # invalid case: order > n_in + 1
    ],
)
def test_bspline_against_scipy(n_in, order, n_output, use_ts):
    if order > n_in + 1:
        with pytest.raises(AssertionError):
            control_points = rng.random((n_in + 1, 2))
            if use_ts:
                bspline(control_points, order=order, ts=jnp.linspace(0, 1, n_output))
            else:
                bspline(control_points, n_output=n_output, order=order)
    else:
        control_points = rng.random((n_in + 1, 2))

        if use_ts:
            result = bspline(control_points, order=order, ts=jnp.linspace(0, 1, n_output))
        else:
            result = bspline(control_points, n_output=n_output, order=order)

        k = order
        T = get_knots_static(n_in, order)
        scipy_bspline = SciPyBSpline(T, np.array(control_points), k - 1)
        expected = scipy_bspline(np.linspace(0, 1, n_output))

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "n_in, order, n_output, use_ts, derivative_order",
    [
        (5, 4, 20, False, 1),  # cubic B-spline first derivative, n_output
        (5, 4, 20, True, 1),  # cubic B-spline first derivative, ts
        (6, 5, 15, False, 2),  # quartic B-spline second derivative, n_output
        (6, 5, 15, True, 2),  # quartic B-spline second derivative, ts
        (5, 4, 20, True, 3),  # cubic B-spline third derivative
        (5, 4, 20, False, 4),  # fourth derivative (should throw an error)
    ],
)
def test_bspline_derivative_against_scipy(n_in, order, n_output, use_ts, derivative_order):
    if derivative_order >= order:
        with pytest.raises(AssertionError):
            control_points = rng.random((n_in + 1, 2))
            bspline_derivative(
                jnp.array(control_points),
                n_output=n_output,
                order=order,
                derivative_order=derivative_order,
            )
    else:
        control_points = rng.random((n_in + 1, 2))

        if use_ts:
            result = bspline_derivative(
                jnp.array(control_points),
                order=order,
                ts=jnp.linspace(0, 1, n_output),
                derivative_order=derivative_order,
            )
        else:
            result = bspline_derivative(
                jnp.array(control_points),
                n_output=n_output,
                order=order,
                derivative_order=derivative_order,
            )

        k = order
        T = get_knots_static(n_in, order)
        scipy_bspline = SciPyBSpline(T, np.array(control_points), k - 1)
        expected = scipy_bspline(np.linspace(0, 1, n_output), nu=derivative_order)

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_bspline_1d_control_points():
    """B-spline should work with 1D output dimension."""
    control_points = jnp.array([[0.0], [1.0], [0.5], [2.0], [1.5]])
    result = bspline(control_points, n_output=20, order=4)
    assert result.shape == (20, 1)
    # Clamped: first and last output should match first and last control point
    np.testing.assert_allclose(result[0], control_points[0], atol=1e-6)
    np.testing.assert_allclose(result[-1], control_points[-1], atol=1e-6)


def test_bspline_boundary_interpolation():
    """Clamped B-spline must interpolate the first and last control points."""
    control_points = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0], [7.0, 1.0]])
    # Test both paths
    result_static = bspline(control_points, n_output=50, order=3)
    result_dynamic = bspline(control_points, order=3, ts=jnp.linspace(0, 1, 50))
    for result in [result_static, result_dynamic]:
        np.testing.assert_allclose(result[0], control_points[0], atol=1e-6)
        np.testing.assert_allclose(result[-1], control_points[-1], atol=1e-6)


def test_bspline_derivative_emit_intermediates():
    """emit_intermediates=True should return a list of all derivative orders."""
    control_points = jnp.array([[0.0, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 0.0], [4.0, -1.0]])
    derivatives = bspline_derivative(
        control_points, n_output=20, order=4, derivative_order=3, emit_intermediates=True
    )
    assert isinstance(derivatives, list)
    assert len(derivatives) == 3
    for d in derivatives:
        assert d.shape == (20, 2)

    # Each intermediate should match calling bspline_derivative separately
    for i, d in enumerate(derivatives):
        single = bspline_derivative(control_points, n_output=20, order=4, derivative_order=i + 1)
        np.testing.assert_allclose(d, single, atol=1e-6)


def test_bspline_arbitrary_ts():
    """Evaluating at arbitrary ts should match scipy at the same points."""
    control_points = rng.random((6, 2))
    ts = jnp.array([0.0, 0.1, 0.33, 0.5, 0.77, 1.0])
    result = bspline(jnp.array(control_points), order=4, ts=ts)

    T = get_knots_static(5, 4)
    scipy_bspline = SciPyBSpline(T, control_points, 3)
    expected = scipy_bspline(np.array(ts))

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_bspline_n_output_and_ts_exclusive():
    """Passing both n_output and ts should raise ValueError."""
    P = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match="either"):
        bspline(P, n_output=10, order=3, ts=jnp.linspace(0, 1, 10))


def test_bspline_requires_n_output_or_ts():
    """Passing neither n_output nor ts should raise ValueError."""
    P = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match="either"):
        bspline(P, order=3)


def test_gradient_through_bspline():
    """jax.grad through bspline should produce correct gradients (vs finite differences)."""
    with jax.enable_x64(True):
        control_points = jnp.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]], dtype=jnp.float64)

        def loss(P):
            curve = bspline(P, order=4, ts=jnp.linspace(0.0, 1.0, 32))
            return jnp.sum(curve**2)

        grad_fn = jax.grad(loss)
        analytic_grad = grad_fn(control_points)

        # Finite differences
        eps = 1e-7
        fd_grad = np.zeros_like(control_points)
        for i in range(control_points.shape[0]):
            for j in range(control_points.shape[1]):
                P_plus = control_points.at[i, j].add(eps)
                P_minus = control_points.at[i, j].add(-eps)
                fd_grad[i, j] = (loss(P_plus) - loss(P_minus)) / (2 * eps)

        np.testing.assert_allclose(analytic_grad, fd_grad, rtol=1e-5, atol=1e-5)
