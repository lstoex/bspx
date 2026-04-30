# %%
"""B-spline sampling strategies: Greville reproduction + arc-length reparameterization."""
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bspx import bspline, bspline_arclength_adjusted, bspline_derivative, greville_abscissae

# %%
# Greville vs uniform control-point spacing for linear reproduction.
n_ctrl = 5
k = 4  # cubic
n_points = 100


def line(t):
    return np.column_stack([t, 2.0 * t + 1.0])


t_uniform = np.linspace(0.0, 1.0, n_ctrl)
P_uniform = jnp.array(line(t_uniform))
curve_uniform = np.array(bspline(P_uniform, n_points=n_points, k=k))

t_greville = greville_abscissae(n_ctrl, k)
P_greville = jnp.array(line(t_greville))
curve_greville = np.array(bspline(P_greville, n_points=n_points, k=k))

t_ref = np.linspace(0.0, 1.0, n_points)
ref = line(t_ref)

fig = go.Figure()
fig.add_trace(go.Scatter(x=P_uniform[:, 0], y=P_uniform[:, 1], mode="markers", name="Uniform control points"))
fig.add_trace(go.Scatter(x=curve_uniform[:, 0], y=curve_uniform[:, 1], mode="lines+markers", name="Uniform curve"))
fig.add_trace(go.Scatter(x=curve_greville[:, 0], y=curve_greville[:, 1], mode="lines+markers", name="Greville curve"))
fig.add_trace(go.Scatter(x=ref[:, 0], y=ref[:, 1], mode="lines", name="Reference line"))
fig.update_layout(title=f"Linear reproduction with {n_ctrl} control points, order {k}", xaxis_title="x", yaxis_title="y", width=700, height=500)
fig.show()

err_uniform = np.linalg.norm(curve_uniform - ref, axis=1)
err_greville = np.linalg.norm(curve_greville - ref, axis=1)
fig = go.Figure()
fig.add_trace(go.Scatter(x=t_ref, y=err_uniform, mode="lines", name=f"Uniform (max {err_uniform.max():.2e})"))
fig.add_trace(go.Scatter(x=t_ref, y=err_greville, mode="lines", name=f"Greville (max {err_greville.max():.2e})"))
fig.update_yaxes(type="log")
fig.update_layout(title="Distance to target line", xaxis_title="t", yaxis_title="distance", width=800, height=500)
fig.show()

# %%
# Arc-length reparameterization on a curvy spline.
P = jnp.array([[0.0, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 0.0], [4.0, -1.0]])
n = 64

curve = bspline(P, n_points=n, k=4)
curve_arclen, t_arclen = bspline_arclength_adjusted(P, n_points=n, k=4)

fig = make_subplots(rows=2, cols=1, subplot_titles=("Curve + control points", "Segment-length histogram"))
fig.add_trace(go.Scatter(x=curve[:, 0], y=curve[:, 1], mode="lines+markers", name="Uniform t"), row=1, col=1)
fig.add_trace(go.Scatter(x=curve_arclen[:, 0], y=curve_arclen[:, 1], mode="lines+markers", name="Arc-length t"), row=1, col=1)
fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="lines+markers", name="Control points", line=dict(dash="dash")), row=1, col=1)

deltas_uniform = jnp.linalg.norm(jnp.diff(curve, axis=0), axis=-1)
deltas_arclen = jnp.linalg.norm(jnp.diff(curve_arclen, axis=0), axis=-1)
fig.add_trace(go.Histogram(x=np.array(deltas_uniform), name="Uniform t", opacity=0.75), row=2, col=1)
fig.add_trace(go.Histogram(x=np.array(deltas_arclen), name="Arc-length t", opacity=0.75), row=2, col=1)
fig.show()

# %%
# Derivatives evaluated at the arc-length-uniform t values.
velocity = bspline_derivative(P, n_points=n, k=4, t=t_arclen, derivative_order=1)
acceleration = bspline_derivative(P, n_points=n, k=4, t=t_arclen, derivative_order=2)

fig = go.Figure()
fig.add_trace(go.Scatter(x=curve_arclen[:, 0], y=curve_arclen[:, 1], mode="lines+markers", name="Curve (arc-length t)"))
fig.add_trace(go.Scatter(x=velocity[:, 0], y=velocity[:, 1], mode="lines+markers", name="Velocity"))
fig.add_trace(go.Scatter(x=acceleration[:, 0], y=acceleration[:, 1], mode="lines+markers", name="Acceleration"))
fig.update_layout(title="Curve and derivatives at arc-length-uniform t", xaxis_title="x", yaxis_title="y")
fig.show()
