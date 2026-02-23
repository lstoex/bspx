# %%
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
# 3D example in plotly
P_3d = jnp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 1.0], [2.0, 2.0, 0.5], [3.0, 0.0, 1.5], [4.0, -1.0, 0.0]])
b_3d = partial(bspline, n_output=128, order=4, use_static=False)
curve_points_3d = b_3d(P_3d)
fig_3d = go.Figure()
fig_3d.add_trace(
    go.Scatter3d(
        x=curve_points_3d[:, 0],
        y=curve_points_3d[:, 1],
        z=curve_points_3d[:, 2],
        mode="lines+markers",
        name="B-spline Curve",
    )
)
fig_3d.add_trace(go.Scatter3d(x=P_3d[:, 0], y=P_3d[:, 1], z=P_3d[:, 2], mode="markers", name="Control Points"))

fig_3d.update_layout(
    title="3D B-spline Curve and Control Points",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
    legend_title="Legend",
)
fig_3d.show()
