# %%
"""Show why Greville abscissae are needed for linear reproduction."""
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

from bspx import bspline, greville_abscissae

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

# %%
err_uniform = np.linalg.norm(curve_uniform - ref, axis=1)
err_greville = np.linalg.norm(curve_greville - ref, axis=1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_ref, err_uniform, "-", linewidth=2, label=f"Uniform (max {err_uniform.max():.2e})")
ax.plot(t_ref, err_greville, "-", linewidth=2, label=f"Greville (max {err_greville.max():.2e})")
ax.set_xlabel("t")
ax.set_ylabel("Distance to line")
ax.set_yscale("log")
ax.set_title(f"Distance to target line: {n_ctrl} control points, order {k}")
ax.legend()
fig.tight_layout()
plt.show()
