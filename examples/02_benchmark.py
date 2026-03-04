# %%
from functools import partial

import jax  # noqa: F401
import jax.numpy as jnp
import numpy as np  # noqa: F401
import plotly.graph_objects as go  # noqa: F401
import prettytable as pt

from bspx import bspline, bspline_arclength_adjusted


# %%
def timeit(func, *args, **kwargs):
    import time

    n_runs = kwargs.pop("n_runs", 1000)
    start = time.time()
    output_form = func(*args, **kwargs)
    out_is_list = isinstance(output_form, (list, tuple))
    for _ in range(n_runs):
        func(*args, **kwargs).block_until_ready() if not out_is_list else func(*args, **kwargs)[0].block_until_ready()
    end = time.time()
    avg = (end - start) / n_runs
    # auto precision
    if avg < 1e-6:
        precision = "ns"
        avg *= 1e9
    elif avg < 1e-3:
        precision = "µs"
        avg *= 1e6
    elif avg < 1:
        precision = "ms"
        avg *= 1e3
    else:
        precision = "s"
    # name = func.__name__ if hasattr(func, "__name__") else str(func)
    return avg, precision


def manually_resampled_bspline(P, n_points=128, k=4):
    curve = bspline(P, n_points=n_points, k=k)
    # compute arclengths
    diffs = jnp.diff(curve, axis=0)
    segment_lengths = jnp.linalg.norm(diffs, axis=1)
    cumulative_lengths = jnp.cumsum(segment_lengths)
    cumulative_lengths = jnp.insert(cumulative_lengths, 0, 0.0)  # add starting point
    normalized_lengths = cumulative_lengths / cumulative_lengths[-1]
    # resample at uniform arclength intervals
    target_lengths = jnp.linspace(0, 1, n_points)
    return jax.vmap(jnp.interp, in_axes=(None, None, 1))(target_lengths, normalized_lengths, curve)


from bspx.utils import clamped_uniform_knot_vector

P = np.random.rand(64, 14)  # 64 control points in 14D
T = clamped_uniform_knot_vector(len(P) - 1, k=4)
t = jnp.linspace(0, 1, 128)
b_static = jax.jit(partial(bspline, n_points=128, k=4))
b_dynamic = jax.jit(partial(bspline, n_points=128, k=4))
b_arclength_adjusted = jax.jit(partial(bspline_arclength_adjusted, n_points=128, k=4))
b_manual = jax.jit(partial(manually_resampled_bspline, n_points=128, k=4))

print("Timing static version...")
avg_static = timeit(b_static, P, n_runs=1000)

print("Timing dynamic version...")
avg_dyn = timeit(b_dynamic, P, T=T, t=t, n_runs=1000)

print("Timing arclength adjusted version...")
avg_ac = timeit(b_arclength_adjusted, P, n_runs=1000)

print("Timing manually resampled version...")
avg_manual = timeit(b_manual, P, n_runs=1000)

# %%
# compare number of flops
n_flops_static = b_static.lower(P).compile().cost_analysis()["flops"]
n_flops_dynamic = b_dynamic.lower(P, T=T, t=t).compile().cost_analysis()["flops"]
n_flops_arclength_adjusted = b_arclength_adjusted.lower(P).compile().cost_analysis()["flops"]
n_flops_manual = b_manual.lower(P).compile().cost_analysis()["flops"]

table = pt.PrettyTable()
table.field_names = ["Version", "Avg Time", "Flops"]
table.add_row(["Static", f"{avg_static[0]:.2f} {avg_static[1]}", f"{n_flops_static}"])
table.add_row(["Dynamic", f"{avg_dyn[0]:.2f} {avg_dyn[1]}", f"{n_flops_dynamic}"])
table.add_row(["Arclength Adjusted", f"{avg_ac[0]:.2f} {avg_ac[1]}", f"{n_flops_arclength_adjusted}"])
table.add_row(["Manual Resampled", f"{avg_manual[0]:.2f} {avg_manual[1]}", f"{n_flops_manual}"])
print(table)
print(f"Number of control points: {len(P)} with dimension {P.shape[1]}")
