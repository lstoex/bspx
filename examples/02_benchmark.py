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
    times = np.ones(n_runs) * np.nan
    output_form = func(*args, **kwargs)
    out_is_list = isinstance(output_form, (list, tuple))
    for i in range(n_runs):
        start = time.time()
        func(*args, **kwargs).block_until_ready() if not out_is_list else func(*args, **kwargs)[0].block_until_ready()
        end = time.time()
        times[i] = end - start
    avg = np.median(times)
    return avg * 1e3, "ms"


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

n_points = 64
k = 4
n = 63
dof = 14
P = np.random.rand(n + 1, dof)  # 64 control points in 14D
T = clamped_uniform_knot_vector(len(P) - 1, k=k)
t = jnp.linspace(0, 1, n_points)
b_static = jax.jit(partial(bspline, n_points=n_points, k=k))
b_dynamic = jax.jit(partial(bspline, n_points=n_points, k=k))
b_arclength_adjusted = jax.jit(partial(bspline_arclength_adjusted, n_points=n_points, k=k))
b_manual = jax.jit(partial(manually_resampled_bspline, n_points=n_points, k=k))

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
table = pt.PrettyTable()
table.field_names = ["Control Points", "Order", "n_points"]
table.add_row([len(P), k, n_points])
print(table)

# %%
# perform full grid search
import sys

arg = sys.argv[1] if len(sys.argv) > 1 else None
if arg is None:
    sys.exit(0)
print("Running full grid search...")
from itertools import product
import pandas as pd

n_points_list = [16, 32, 64, 128]
k_list = [2, 3, 4, 5]
n_list = [15, 31, 63, 127]
results = []
key = jax.random.PRNGKey(0)
for i, (n_points, k, n) in enumerate(product(n_points_list, k_list, n_list)):
    print(
        f"Running benchmark {i + 1}/{len(n_points_list) * len(k_list) * len(n_list)}: n_points={n_points}, k={k}, n={n}, dof={dof}"
    )
    P = jax.random.uniform(key, (n + 1, dof))
    T = clamped_uniform_knot_vector(len(P) - 1, k=k)
    t = jnp.linspace(0, 1, n_points)
    b_static = jax.jit(partial(bspline, n_points=n_points, k=k))
    b_dynamic = jax.jit(partial(bspline, n_points=n_points, k=k))
    b_arclength_adjusted = jax.jit(partial(bspline_arclength_adjusted, n_points=n_points, k=k))
    b_manual = jax.jit(partial(manually_resampled_bspline, n_points=n_points, k=k))
    avg_static = timeit(b_static, P, n_runs=1000)
    avg_dyn = timeit(b_dynamic, P, T=T, t=t, n_runs=1000)
    avg_ac = timeit(b_arclength_adjusted, P, n_runs=1000)
    avg_manual = timeit(b_manual, P, n_runs=1000)
    results.append((n_points, k, n, avg_static[0], avg_dyn[0], avg_ac[0], avg_manual[0]))

df = pd.DataFrame(
    results,
    columns=[
        "n_points",
        "k",
        "n",
        "static_time_ms",
        "dynamic_time_ms",
        "arclength_adjusted_time_ms",
        "manual_time_ms",
    ],
)
# save to csv
df.to_csv("benchmark_results.csv", index=False)
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
variants = ["static_time_ms", "dynamic_time_ms", "arclength_adjusted_time_ms", "manual_time_ms"]
variant_labels = ["Static", "Dynamic", "Arclength Adj.", "Manual"]
params = ["n_points", "k", "n"]
df = pd.read_csv("../benchmark_results.csv")
df_long = df.melt(id_vars=["n_points", "k", "n"], value_vars=variants, var_name="method", value_name="time_ms")

method_labels = {
    "static_time_ms": "Static",
    "dynamic_time_ms": "Dynamic",
    "arclength_adjusted_time_ms": "Arclength Adj.",
    "manual_time_ms": "Manual",
}
df_long["method"] = df_long["method"].map(method_labels)
for param in ["n_points", "k", "n"]:
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df_long, x=param, y="time_ms", hue="method", palette="Set2", ax=ax)
    ax.set_title(f"Time distribution by {param}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()