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
curve = bspline(P, n_output=42, order=4)
```

We always use clamped knots, so that the curves coincide with the first and last control point.

**To enable beartype, have a look at `src/bspx/__init__.py`**

## Notation
| Symbol | Description |
|--------|-------------|
| $k$ | order (= degree + 1) |
| $n$ | index of last control point → $n+1$ control points |
| $m$ | index of last knot → $m+1$ knots, $m = n + k$ |
| $t$ | parameter value |
| $T$ | knot vector, shape (m+1,) |
| $P$ | control points, shape (n+1, d) |
