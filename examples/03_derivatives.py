# %%
import jax
import jax.numpy as jnp
import plotly.graph_objects as go

from bspx import bspline
from functools import partial

# %%
P = jnp.array([[0.0, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 0.0], [4.0, -1.0]])

# 64 output points, cubic B-spline (order 4 - 1 = degree 3)
b = partial(bspline, n_output=64, order=4, use_static=False)
curve_points = b(P)

fig = go.Figure()
fig.add_trace(go.Scatter(x=curve_points[:, 0], y=curve_points[:, 1], mode="lines+markers", name="B-spline Curve"))
fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="lines+markers", name="Control Points", line=dict(dash="dash")))
fig.update_layout(title="B-spline Curve and Control Points", xaxis_title="x", yaxis_title="y", legend_title="Legend")
fig.show()

# %%
# Test differentiability of the curve w.r.t. control points.
# Start from a flat curve and optimize to match the target shape.

P_start = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
P_target = P
b = partial(bspline, n_output=64, order=4, use_static=False)


def loss_fn(P):
    # Closest-point distance from the curve to the target points
    curve_points = b(P)
    ref_curve_points = b(P_target)
    dists = jnp.linalg.norm(curve_points - ref_curve_points, axis=-1, ord=1)
    return jnp.mean(dists)


@jax.jit
def step(P, lr=1e-3):
    loss = loss_fn(P)
    grads = jax.grad(loss_fn)(P)
    return P - lr * grads, loss


# Optimize the control points to minimize the loss
P_opt = P_start
iters = 2000
lr = 1e-2
for i in range(iters):
    P_opt, loss = step(P_opt, lr)
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss:.4f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=b(P_opt)[:, 0], y=b(P_opt)[:, 1], mode="lines+markers", name="Optimized B-spline Curve"))
fig.add_trace(
    go.Scatter(x=P_target[:, 0], y=P_target[:, 1], mode="lines+markers", name="Target Points", line=dict(dash="dash"))
)
fig.add_trace(go.Scatter(x=P_opt[:, 0], y=P_opt[:, 1], mode="markers", name="Optimized Control Points"))
fig.update_layout(
    title="Optimized B-spline Curve and Target Points", xaxis_title="x", yaxis_title="y", legend_title="Legend"
)
fig.show()

# %%
# Disturb a single output point and observe smooth compensation.
fig = go.Figure()
original_curve = b(P)
disturbed_curve = original_curve.at[32].add(jnp.array([0.5, 0.5]))
fig.add_trace(
    go.Scatter(x=original_curve[:, 0], y=original_curve[:, 1], mode="lines+markers", name="Original B-spline Curve")
)
fig.add_trace(
    go.Scatter(
        x=disturbed_curve[:, 0],
        y=disturbed_curve[:, 1],
        mode="lines+markers",
        name="Disturbed B-spline Curve",
        line=dict(dash="dash"),
    )
)
fig.update_layout(title="Disturbed B-spline Curve", xaxis_title="x", yaxis_title="y", legend_title="Legend")
fig.show()


# Apply the gradient of the disturbance to the control points
def disturbance_loss_fn(P):
    curve_points = b(P)
    dist_for_point = jnp.linalg.norm(curve_points[32] - disturbed_curve[32], ord=1)
    return dist_for_point


@jax.jit
def disturbance_step(P, lr=1e-3):
    loss = disturbance_loss_fn(P)
    grads = jax.grad(disturbance_loss_fn)(P)
    return P - lr * grads, loss


P_disturbed = P
for i in range(1000):
    P_disturbed, loss = disturbance_step(P_disturbed, lr=1e-2)
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss:.4f}")
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=b(P_disturbed)[:, 0], y=b(P_disturbed)[:, 1], mode="lines+markers", name="Compensated B-spline Curve")
)
fig.add_trace(
    go.Scatter(
        x=disturbed_curve[:, 0],
        y=disturbed_curve[:, 1],
        mode="lines+markers",
        name="Disturbed B-spline Curve",
        line=dict(dash="dash"),
    )
)
fig.update_layout(title="Compensated B-spline Curve", xaxis_title="x", yaxis_title="y", legend_title="Legend")
fig.show()
