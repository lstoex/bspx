## B-spline curve evaluation

We optimize control points for downstream tasks by leveraging the de Boor algorithm instead of precomputing basis functions. Since each curve point depends on only k control points, we avoid sparse matrix storage.

JAX's JIT compiler lets us precompute the knot vector, indices, and blending factors at compile time, yielding efficient evaluation.

| Symbol | Description |
|--------|-------------|
| $k$ | order (= degree + 1) |
| $n$ | index of last control point → $n+1$ control points |
| $m$ | index of last knot → $m+1$ knots, $m = n + k$ |
| $t$ | parameter value |
| $T$ | knot vector, shape (m+1,) |
| $P$ | control points, shape (n+1, d) |

