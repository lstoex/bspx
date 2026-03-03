# %%
"""Show why Greville abscissae are needed for linear reproduction."""
from click.decorators import P
import jax.numpy as jnp
import numpy as np

from bspx import bspline, greville_abscissae
import plotly.graph_objects as go

# %%
n_ctrl = 5
k = 4  # cubic
n_points = 100


def line(t):
    return np.column_stack([t, 2.0 * t + 1.0])


# uniform control-point spacing
t_uniform = np.linspace(0.0, 1.0, n_ctrl)
P_uniform = jnp.array(line(t_uniform))
curve_uniform = np.array(bspline(P_uniform, n_points=n_points, k=k))

# Greville control-point spacing
t_greville = greville_abscissae(n_ctrl, k)
P_greville = jnp.array(line(t_greville))
curve_greville = np.array(bspline(P_greville, n_points=n_points, k=k))

# reference straight line
t_ref = np.linspace(0.0, 1.0, n_points)
ref = line(t_ref)
#%%
fig = go.Figure()
fig.add_trace(go.Scatter(x=P_uniform[:, 0], y=P_uniform[:, 1], mode="markers", name="Uniform control points"))
fig.add_trace(go.Scatter(x=curve_uniform[:, 0], y=curve_uniform[:, 1], mode="lines+markers", name="Uniform curve"))
fig.add_trace(go.Scatter(x=curve_greville[:, 0], y=curve_greville[:, 1], mode="lines+markers", name="Greville curve"))
fig.add_trace(go.Scatter(x=ref[:, 0], y=ref[:, 1], mode="lines", name="Reference line"))
fig.update_layout(title=f"Linear reproduction with {n_ctrl} control points, order {k}", xaxis_title="x", yaxis_title="y", width=700, height=500)
fig.show()

# %%
import matplotlib.pyplot as plt
import jax
err_uniform = np.linalg.norm(curve_uniform - ref, axis=1) #Hier liegt der Hase im Pfeffer
err_greville = np.linalg.norm(curve_greville - ref, axis=1)
def distance_to_line(point, line_start, line_end):
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    #assume line_len > 0
    line_unitvec = line_vec / line_len
    point_vec = point - line_start
    t = jnp.dot(point_vec, line_unitvec)
    t = jnp.clip(t, 0.0, line_len)
    nearest = line_start + t * line_unitvec
    return jnp.linalg.norm(point - nearest)


err_uniform_point2line = np.asarray(jax.vmap(lambda p: distance_to_line(p, ref[0], ref[-1]))(curve_uniform))

def resample(
    data,
    n_new: int,
    t=None,
    xp_indices=None,
):
    # from ikx trajectory module
    i = list(range(data.shape[-1])) if xp_indices is None else list(xp_indices)
    xdata = data[..., i]
    # Calculate cumulative arc length
    distances = jnp.linalg.norm(xdata[1:] - xdata[:-1], axis=-1)
    cumulative_distances = jnp.cumsum(distances)
    cumulative_distances = jnp.insert(cumulative_distances, 0, 0)

    # Normalize to [0, 1]
    normalized_distances = cumulative_distances / cumulative_distances[-1]

    # Equally spaced points in the normalized distance
    t = jnp.linspace(0, 1, n_new) if t is None else t
    return jax.vmap(jnp.interp, in_axes=(None, None, 1))(t, normalized_distances, data).T
    
err_resampled_uniform = np.linalg.norm(resample(curve_uniform, n_points) - ref, axis=1)

fig = go.Figure()
fig.add_trace(go.Scatter(x=t_ref, y=err_uniform, mode="lines", name=f"Uniform (max {err_uniform.max():.2e})", line=dict(width=2)))

fig.add_trace(go.Scatter(x=t_ref, y=err_greville, mode="lines", name=f"Greville (max {err_greville.max():.2e})", line=dict(width=2)))

fig.add_trace(
    go.Scatter(
        x=t_ref,
        y=err_uniform_point2line,
        mode="lines",
        name=f"Uniform corrected (max {err_uniform_point2line.max():.2e})",
        line=dict(width=2, dash="dash"),
    )
)

fig.add_trace(go.Scatter(x=t_ref, y=err_resampled_uniform, mode="lines", name=f"Resampled uniform (max {err_resampled_uniform.max():.2e})", line=dict(width=2, dash="dot")))
fig.update_yaxes(type="log")
fig.update_layout(title=f"Distance to target line: {n_ctrl} control points, order {k}", xaxis_title="t", yaxis_title="Distance to line", width=800, height=500)
fig.show()
#%%
#line without correction, and the reference line. Shift the curve by a fixed y offset and connect the points(per index) with an arrow to visualize the error direction. This is a bit hacky, but it should work.
y_offset = 0.5
fig = go.Figure()
fig.add_trace(go.Scatter(x=curve_uniform[:, 0], y=curve_uniform[:, 1] + y_offset, mode="lines+markers", name="Uniform curve", line=dict(width=2)))
fig.add_trace(go.Scatter(x=ref[:, 0], y=ref[:, 1], mode="lines", name="Reference line", line=dict(width=2)))
for i in range(len(curve_uniform)):
    fig.add_trace(go.Scatter(x=[curve_uniform[i, 0], ref[i, 0]], y=[curve_uniform[i, 1] + y_offset, ref[i, 1]], mode="lines", name="Error vector", line=dict(width=1, color="red")))
fig.update_layout(title=f"Error vectors for uniform control points", xaxis_title="x", yaxis_title="y", width=800, height=500)
fig.show()