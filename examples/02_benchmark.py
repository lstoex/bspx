#%%
import jax  # noqa: F401
import jax.numpy as jnp
import numpy as np  # noqa: F401
import plotly.graph_objects as go  # noqa: F401

from bspx import BSpline


#%%
def timeit(func, *args, **kwargs):
    import time

    n_runs = kwargs.pop("n_runs", 1000)
    start = time.time()
    for _ in range(n_runs):
        func(*args, **kwargs).block_until_ready()  # for JAX functions, ensure we wait for completion
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
    print(f"{func.__name__} took {avg:.2f} {precision} on average over {n_runs} runs")
    return avg


# Time the static version
b_static = BSpline(128, 4, use_static=True)
P = jnp.array([[0.0, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 0.0], [4.0, -1.0]])

def static_variant(P):
    return b_static(P)

def dynamic_variant(P):
    return b_dynamic(P)

timeit(static_variant, P, n_runs=1000)
# Time the dynamic version
b_dynamic = BSpline(128, 4, use_static=False)
timeit(dynamic_variant, P, n_runs=1000)

#%%
# compare number of flops
n_flops_static = b_static.__call__.lower(b_static, P).compile().cost_analysis()["flops"]
n_flops_dynamic = b_dynamic.__call__.lower(b_dynamic, P).compile().cost_analysis()["flops"]
print(f"Static version flops: {n_flops_static}")
print(f"Dynamic version flops: {n_flops_dynamic}")
print(
    f"Static version is {n_flops_dynamic / n_flops_static:.2f} times more efficient in terms of flops"
)
