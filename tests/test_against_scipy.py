import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import BSpline as SciPyBSpline

import bspx


@pytest.mark.parametrize(
    "n, k, n_points",
    [
        (4, 3, 10),  # cubic Bezier
        (5, 4, 20),  # cubic B-spline
        (6, 5, 15),  # quartic B-spline
        (7, 6, 25),  # quintic B-spline
        (3, 2, 5),  # linear B-spline
        (4, 5, 10),  # invalid case: k > n + 1, should raise an error
    ],
)
def test_bspline_uniform_against_scipy(n, k, n_points):
    import bspx

    T = bspx.clamped_uniform_knot_vector(n, k)
    t = np.linspace(0, 1, n_points)

    if k > n + 1:
        with pytest.raises(AssertionError):
            control_points = jnp.random.rand(n + 1, 2)  # 2D control points
            bspx.bspline(control_points, n_points=n_points, k=k)
    else:
        # Create random control points
        control_points = np.random.rand(n + 1, 2)  # 2D control points

        # Create our B-spline and evaluate
        result = bspx.bspline(control_points, n_points=n_points, k=k)

        # Create SciPy B-spline and evaluate
        k = k

        scipy_bspline = SciPyBSpline(T, np.array(control_points), k - 1)
        expected = scipy_bspline(t)

        # Compare results
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "n, k, n_points",
    [
        (4, 3, 10),  # cubic Bezier
        (5, 4, 20),  # cubic B-spline
        (6, 5, 15),  # quartic B-spline
        (7, 6, 25),  # quintic B-spline
        (3, 2, 5),  # linear B-spline
        (4, 5, 10),  # invalid case: k > n + 1, should raise an error
    ],
)
def test_bspline_nonuniform_against_scipy(n, k, n_points):
    import bspx

    m = n + k
    T_nonuniform = np.concatenate(
        [
            np.zeros(k),  # start with repeated knots for clamping
            np.sort(np.random.rand(m + 1 - 2 * k)),  # random internal knots
            np.ones(k),  # end with repeated knots for clamping
        ]
    )
    t_nonuniform = np.sort(np.random.rand(n_points))

    if k > n + 1:
        with pytest.raises(AssertionError):
            control_points = jnp.random.rand(n + 1, 2)  # 2D control points
            bspx.bspline(control_points, n_points=n_points, k=k, T=T_nonuniform, t=t_nonuniform)
    else:
        # Create random control points
        control_points = np.random.rand(n + 1, 2)  # 2D control points

        # Create our B-spline and evaluate
        result = bspx.bspline(control_points, n_points=n_points, k=k, T=T_nonuniform, t=t_nonuniform)

        # Create SciPy B-spline and evaluate
        k = k

        scipy_bspline = SciPyBSpline(T_nonuniform, np.array(control_points), k - 1)
        expected = scipy_bspline(t_nonuniform)

        # Compare results
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "n, k, n_points, derivative_order",
    [
        (5, 4, 20, 1),  # cubic B-spline first derivative
        (6, 5, 15, 2),  # quartic B-spline second derivative
        (5, 4, 20, 3),  # cubic B-spline third derivative (should be piecewise constant)
        (5, 4, 20, 4),  # cubic B-spline fourth derivative (should throw an error since it's not defined)
    ],
)
def test_bspline_derivative_uniform_against_scipy(n, k, n_points, derivative_order):
    import bspx

    T = bspx.clamped_uniform_knot_vector(n, k)
    t = np.linspace(0, 1, n_points)

    if derivative_order >= k:
        with pytest.raises(AssertionError):
            control_points = np.random.rand(n + 1, 2)  # 2D control points
            from bspx import bspline_derivative

            bspx.bspline_derivative(
                jnp.array(control_points), k=k, n_points=n_points, derivative_order=derivative_order
            )
    else:
        # Create random control points
        control_points = np.random.rand(n + 1, 2)  # 2D control points

        # Create our B-spline derivative and evaluate

        result = bspx.bspline_derivative(
            jnp.array(control_points), k=k, n_points=n_points, derivative_order=derivative_order
        )

        # Create SciPy B-spline and evaluate its derivative
        k = k
        scipy_bspline = SciPyBSpline(T, np.array(control_points), k - 1)
        expected = scipy_bspline(t, nu=derivative_order)

        # Compare results
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "n, k, n_points, derivative_order",
    [
        (5, 4, 20, 1),  # cubic B-spline first derivative
        (6, 5, 15, 2),  # quartic B-spline second derivative
        (5, 4, 20, 3),  # cubic B-spline third derivative (should be piecewise constant)
        (5, 4, 20, 4),  # cubic B-spline fourth derivative (should throw an error since it's not defined)
    ],
)
def test_bspline_derivative_nonuniform_against_scipy(n, k, n_points, derivative_order):
    import bspx

    m = n + k
    T_nonuniform = np.concatenate(
        [
            np.zeros(k),  # start with repeated knots for clamping
            np.sort(np.random.rand(m + 1 - 2 * k)),  # random internal knots
            np.ones(k),  # end with repeated knots for clamping
        ]
    )
    t_nonuniform = np.sort(np.random.rand(n_points))

    if derivative_order >= k:
        with pytest.raises(AssertionError):
            control_points = np.random.rand(n + 1, 2)  # 2D control points

            bspx.bspline_derivative(
                jnp.array(control_points),
                k=k,
                n_points=n_points,
                derivative_order=derivative_order,
                T=T_nonuniform,
                t=t_nonuniform,
            )
    else:
        # Create random control points
        control_points = np.random.rand(n + 1, 2)  # 2D control points

        # Create our B-spline derivative and evaluate

        result = bspx.bspline_derivative(
            jnp.array(control_points),
            k=k,
            n_points=n_points,
            derivative_order=derivative_order,
            T=T_nonuniform,
            t=t_nonuniform,
        )

        # Create SciPy B-spline and evaluate its derivative
        k = k
        scipy_bspline = SciPyBSpline(T_nonuniform, np.array(control_points), k - 1)
        expected = scipy_bspline(t_nonuniform, nu=derivative_order)

        # Compare results
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_ctrl, k", [(6, 4), (8, 4), (8, 5)])
def test_greville_abscissae_needed(n_ctrl, k):
    """Uniformly-spaced control points on a line do NOT reproduce that line
    for order k >= 3, but Greville-spaced control points do."""

    def line(t):
        return np.column_stack([t, 2.0 * t + 1.0])

    n_points = 50

    # --- uniform spacing: NOT a straight line ---
    t_uniform = np.linspace(0.0, 1.0, n_ctrl)
    P_uniform = jnp.array(line(t_uniform))
    curve_uniform = np.array(bspx.bspline(P_uniform, n_points=n_points, k=k))

    # the expected straight line evaluated at the same output parameters
    t_out = np.linspace(0.0, 1.0, n_points)
    expected = line(t_out)

    max_err_uniform = np.max(np.abs(curve_uniform - expected))
    assert max_err_uniform > 1e-3, (
        f"Uniform spacing should deviate from the line, but max error was only {max_err_uniform:.2e}"
    )

    # --- Greville spacing: IS a straight line ---
    t_greville = bspx.greville_abscissae(n_ctrl, k)
    P_greville = jnp.array(line(t_greville))
    curve_greville = np.array(bspx.bspline(P_greville, n_points=n_points, k=k))

    np.testing.assert_allclose(curve_greville, expected, atol=1e-5)
