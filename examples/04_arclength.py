# %%
import trace
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

def resample(data, n_new: int, t=None) -> jnp.ndarray:
    # Calculate cumulative arc length
    distances = safe_norm(data[1:] - data[:-1], axis=-1, ord=2, min_norm=1e-4)
    cumulative_distances = jnp.cumsum(distances)
    cumulative_distances = jnp.insert(cumulative_distances, 0, 0)

    # Normalize to [0, 1]
    normalized_distances = cumulative_distances / cumulative_distances[-1]

    # Equally spaced points in the normalized distance
    t = jnp.linspace(0, 1, n_new) if t is None else t
    q_resampled = jax.vmap(jnp.interp, in_axes=(None, None, 1))(t, normalized_distances, data).T
    return q_resampled


# %%
P = jnp.array([[0.0, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 0.0], [4.0, -1.0]])

# 64 output points, cubic B-spline (order 4 - 1 = degree 3)
b = partial(bspline, n_output=64, order=4, use_static=False)
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

#make one subplots for the curves and one for the histograms
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
