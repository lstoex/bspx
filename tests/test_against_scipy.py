import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import BSpline as SciPyBSpline


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

    T = bspx.utils.clamped_uniform_knot_vector(n, k)
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

    T = bspx.utils.clamped_uniform_knot_vector(n, k)
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
