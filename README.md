# BSPX - B-Splines in JAX
> Fast, differentiable B-splines in JAX with de Boor evaluation.

## Features
- JAX-traceable De Boor implementation - completely sparse
- Jitable, vmapable, etc.
- Compile-time precomputation for minimal FLOPS

## Quick start
```python
from bspx import BSpline

P = jnp.array([0., 1., 1., 2., 4., 2. , 3.])
b = BSpline(n_output=42, order=4)
curve = b(P)
```

We always use clamped knots, so that the curves coincide with the first and last control point.

## Notation
| Symbol | Description |
|--------|-------------|
| $k$ | order (= degree + 1) |
| $n$ | index of last control point → $n+1$ control points |
| $m$ | index of last knot → $m+1$ knots, $m = n + k$ |
| $t$ | parameter value |
| $T$ | knot vector, shape (m+1,) |
| $P$ | control points, shape (n+1, d) |
