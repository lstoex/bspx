# %%
import scipy
import jax
import jax.numpy as jnp
import plotly.graph_objects as go

from bspx import bspline
from functools import partial

# %%
from typing import Optional, Union


def safe_norm(
    x: jax.typing.ArrayLike,
    min_norm: jax.typing.ArrayLike,
    ord: Optional[Union[int, float, str]] = None,
    axis: Union[None, tuple[int, ...], int] = None,
    keepdims: bool = False,
) -> jax.Array:
    """Taken from optax, to avoid NaNs when computing norms of small vectors."""
    norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=True)
    x = jnp.where(norm <= min_norm, jnp.ones_like(x), x)
    norm = jnp.squeeze(norm, axis=axis) if not keepdims else norm
    masked_norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    return jnp.where(norm <= min_norm, min_norm, masked_norm)


def resample(data, n_new: int, t=None, xp_indices: tuple[int, ...] | None = None) -> jnp.ndarray:
    # Calculate cumulative arc length
    xp_indices = list(range(data.shape[-1])) if xp_indices is None else list(xp_indices)
    # xp_indices gives dims which participate in the distance calculation.
    xdata = data[..., xp_indices]
    distances = safe_norm(xdata[1:] - xdata[:-1], axis=-1, ord=2, min_norm=1e-4)
    cumulative_distances = jnp.cumsum(distances)
    cumulative_distances = jnp.insert(cumulative_distances, 0, 0)

    # Normalize to [0, 1]
    normalized_distances = cumulative_distances / cumulative_distances[-1]

    # Equally spaced points in the normalized distance
    t = jnp.linspace(0, 1, n_new) if t is None else t
    return jax.vmap(jnp.interp, in_axes=(None, None, 1))(t, normalized_distances, data).T


# %%
P = jnp.array([[0.0, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 0.0], [4.0, -1.0]])

# 64 output points, cubic B-spline (k 4 - 1 = degree 3)
b = partial(bspline, n_points=64, k=4)
curve_points = b(P)
curve_points_resampled = resample(curve_points, n_new=64)

trace_curve = go.Scatter(x=curve_points[:, 0], y=curve_points[:, 1], mode="lines+markers", name="B-spline Curve")
trace_resampled = go.Scatter(
    x=curve_points_resampled[:, 0],
    y=curve_points_resampled[:, 1],
    mode="lines+markers",
    name="Resampled B-spline Curve",
)
trace_control = go.Scatter(x=P[:, 0], y=P[:, 1], mode="lines+markers", name="Control Points", line=dict(dash="dash"))

# make one subplots for the curves and one for the histograms
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=1, subplot_titles=("B-spline Curve and Control Points", "Histogram of Deltas"))
fig.add_trace(trace_curve, row=1, col=1)
fig.add_trace(trace_resampled, row=1, col=1)
fig.add_trace(trace_control, row=1, col=1)

deltas_before = jnp.linalg.norm(jnp.diff(curve_points, axis=0), axis=-1)
deltas_after = jnp.linalg.norm(jnp.diff(curve_points_resampled, axis=0), axis=-1)
trace_hist_before = go.Histogram(x=deltas_before, name="Deltas Before Resampling", opacity=0.75)
trace_hist_after = go.Histogram(x=deltas_after, name="Deltas After Resampling", opacity=0.75)
fig.add_trace(trace_hist_before, row=2, col=1)
fig.add_trace(trace_hist_after, row=2, col=1)
fig.show()

# %%
from bspx import bspline_derivative

curve_points = bspline(P, n_points=64, k=4)
velocity_points = bspline_derivative(P, n_points=64, k=4)
acceleration_points = bspline_derivative(P, n_points=64, k=4, derivative_order=2)
trace_curve = go.Scatter(x=curve_points[:, 0], y=curve_points[:, 1], mode="lines+markers", name="B-spline Curve")
trace_velocity = go.Scatter(
    x=velocity_points[:, 0], y=velocity_points[:, 1], mode="lines+markers", name="Velocity (1st Derivative)"
)
trace_acceleration = go.Scatter(
    x=acceleration_points[:, 0], y=acceleration_points[:, 1], mode="lines+markers", name="Acceleration (2nd Derivative)"
)
fig = go.Figure(data=[trace_curve, trace_velocity, trace_acceleration])
fig.update_layout(title="B-spline Curve and its Derivatives", xaxis_title="X", yaxis_title="Y")
fig.show()

# %%
data = jnp.hstack([curve_points, velocity_points, acceleration_points])
data_resampled = resample(data, n_new=64, xp_indices=(0, 1))
curve_points_resampled = data_resampled[:, :2]
velocity_resampled = data_resampled[:, 2:4]
acceleration_resampled = data_resampled[:, 4:6]
trace_curve_resampled = go.Scatter(
    x=curve_points_resampled[:, 0],
    y=curve_points_resampled[:, 1],
    mode="lines+markers",
    name="Resampled B-spline Curve",
)
trace_velocity_resampled = go.Scatter(
    x=velocity_resampled[:, 0],
    y=velocity_resampled[:, 1],
    mode="lines+markers",
    name="Resampled Velocity (1st Derivative)",
)
trace_acceleration_resampled = go.Scatter(
    x=acceleration_resampled[:, 0],
    y=acceleration_resampled[:, 1],
    mode="lines+markers",
    name="Resampled Acceleration (2nd Derivative)",
)
fig = go.Figure(data=[trace_curve_resampled, trace_velocity_resampled, trace_acceleration_resampled])
fig.update_layout(title="Resampled B-spline Curve and its Derivatives", xaxis_title="X", yaxis_title="Y")
fig.show()
# %%
