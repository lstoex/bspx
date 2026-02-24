import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import BSpline as SciPyBSpline

from bspx import bspline, bspline_derivative
from bspx.utils_static import get_knots_static

rng = np.random.default_rng(42)


@pytest.mark.parametrize(
    "n_in, order, n_output, static",
    [
        (4, 3, 10, True),  # cubic Bezier
        (5, 4, 20, True),  # cubic B-spline
        (6, 5, 15, True),  # quartic B-spline
        (7, 6, 25, True),  # quintic B-spline
        (3, 2, 5, True),  # linear B-spline
        (4, 5, 10, True),  # invalid case: order > n_in + 1, should raise an error
        (4, 3, 10, False),  # cubic Bezier
        (5, 4, 20, False),  # cubic B-spline
        (6, 5, 15, False),  # quartic B-spline
        (7, 6, 25, False),  # quintic B-spline
        (3, 2, 5, False),  # linear B-spline
        (4, 5, 10, False),  # invalid case: order > n_in + 1, should raise an error
    ],
)
def test_bspline_against_scipy(n_in, order, n_output, static):
    if order > n_in + 1:
        with pytest.raises(AssertionError):
            control_points = rng.random((n_in + 1, 2))
            bspline(control_points, n_output=n_output, order=order, use_static=static)
    else:
        control_points = rng.random((n_in + 1, 2))

        result = bspline(control_points, n_output=n_output, order=order, use_static=static)

        k = order
        T = get_knots_static(n_in, order)
        scipy_bspline = SciPyBSpline(T, np.array(control_points), k - 1)
        expected = scipy_bspline(np.linspace(0, 1, n_output))

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "n_in, order, n_output, static, derivative_order",
    [
        (5, 4, 20, True, 1),  # cubic B-spline first derivative
        (5, 4, 20, False, 1),  # cubic B-spline first derivative with use_static=False
        (6, 5, 15, True, 2),  # quartic B-spline second derivative
        (6, 5, 15, False, 2),  # quartic B-spline second derivative with use_static=False
        (5, 4, 20, False, 3),  # cubic B-spline third derivative (should be piecewise constant)
        (5, 4, 20, True, 4),  # cubic B-spline fourth derivative (should throw an error since it's not defined)
    ],
)
def test_bspline_derivative_against_scipy(n_in, order, n_output, static, derivative_order):
    if derivative_order >= order:
        with pytest.raises(AssertionError):
            control_points = rng.random((n_in + 1, 2))
            bspline_derivative(
                jnp.array(control_points),
                n_output=n_output,
                order=order,
                use_static=static,
                derivative_order=derivative_order,
            )
    else:
        control_points = rng.random((n_in + 1, 2))

        result = bspline_derivative(
            jnp.array(control_points),
            n_output=n_output,
            order=order,
            use_static=static,
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
    for static in [True, False]:
        result = bspline(control_points, n_output=50, order=3, use_static=static)
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


def test_gradient_through_bspline():
    """jax.grad through bspline should produce correct gradients (vs finite differences)."""
    with jax.enable_x64(True):
        control_points = jnp.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]], dtype=jnp.float64)

        def loss(P):
            curve = bspline(P, n_output=32, order=4, use_static=False)
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
