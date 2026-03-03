# BSPX - B-Splines in JAX
> Efficient, differentiable B-splines in JAX with [De Boor's](https://en.wikipedia.org/wiki/De_Boor%27s_algorithm) algorithm.

## Features
- JAX-traceable De Boor implementation - completely sparse
- Jitable, vmapable, etc.
- Compile-time precomputation for minimal FLOPS
- Transparent shapes and types with Jaxtyping

## Quick start
```python
from bspx import bspline

P = jnp.array([0., 1., 1., 2., 4., 2. , 3.])
curve = bspline(P, n_points=42, k=4)
```
When no knots and evaluation times are given, we assume cardinal B-splines, i.e. uniform knots and evaluation times. This allows us to precompute the blending factors at compile time, resulting in efficient evaluation during runtime. For this, most functions in `bspx` process both NumPy arrays and JAX tracers, so we can use the same code for precomputation and runtime evaluation.
**To enable beartype, have a look at `src/bspx/__init__.py`**

## Linear reproduction and Greville abscissae

A B-spline reproduces linear functions exactly **only** when the control
points are placed at the [Greville abscissae](https://en.wikipedia.org/wiki/De_Boor%27s_algorithm#Greville_abscissae) — the average of each basis function's `k-1` consecutive interior knots:

$$\xi_i = \frac{1}{k-1}\sum_{j=1}^{k-1} T_{i+j}$$

To reproduce `f(t) = a + b*t`, set `P_i = f(ξ_i)`.  **Uniformly-spaced**
control points do **not** yield a straight line (except when `n_ctrl = k`,
i.e. the Bezier case where Greville and uniform coincide).

```python
from bspx import bspline, greville_abscissae
import jax.numpy as jnp

n_ctrl, order = 5, 4
xi = greville_abscissae(n_ctrl, order)   # [0, 1/6, 1/2, 5/6, 1]
uniform = jnp.linspace(0, 1, n_ctrl)     # [0, .25, .5, .75, 1]

# Linear control points at Greville positions → exact straight line
P_greville = xi[:, None] * jnp.array([[3.0, 7.0]])
curve = bspline(P_greville, n_output=100, order=order)
# max deviation from true linear: ~1e-7 (machine epsilon)

# Linear control points at uniform positions → NOT a straight line
P_uniform = uniform[:, None] * jnp.array([[3.0, 7.0]])
curve_bad = bspline(P_uniform, n_output=100, order=order)
# max deviation from true linear: ~0.08 per unit span
```

This matters whenever the initial B-spline curve should be a known function
(e.g. a straight-line initialization for path planning).  After optimization
moves the control points freely, Greville positions no longer apply — they
are only relevant for initialization.

## Notation
| Symbol | Description |
|--------|-------------|
| $k$ | order (= degree + 1) |
| $n$ | index of last control point → $n+1$ control points |
| $m$ | index of last knot → $m+1$ knots, $m = n + k$ |
| $t$ | parameter value |
| $T$ | knot vector, shape (m+1,) |
| $P$ | control points, shape (n+1, d) |

## TODOs
**Other:**
- [x] Check if a custom vjp might be faster than autodiff -> No its not!
