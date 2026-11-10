# Implementing expressions

These are useful guidelines to follow when making or editing expressions.

## Testing

Prefer testing from the python side, as this saves a lot of compilation time.
That way, you are also much more sure you are actually testing behaviour,
rather than internal implementation details :)

Rust tests can still be necessary if a certain functionality is only reachable from rust directly.

## Naming kernel output types

Polars allows you to pass a function which gets the output dtype of the kernel,
e.g. `#[polars_expr(output_type_func=kernel_output_type_area_planar)]`.
These should always be named `kernel_output_type_*`.
Functions that wrap these should be `kernel_output_type_wrapper_*`

## Kernels live in `geopolars-compute`

Kernels used to work directly on Polars' Arrow arrays.
We are moving them to the `../geopolars-compute` crate.
They take the views of `../geopolars-core`.
That way, the 'geo' logic is separated from the Polars plugin layer.
Write new kernels there.
Some kernels still live in `src/expr/` until they are migrated.
See [its README](../geopolars-compute/README.md) for how to write a kernel.

What remains in the plugin is a thin layer per expression in `src/expr/`.

- It checks the input's dtype with `describe`, and picks a kernel per `Kind`.
- It wraps each chunk of the storage in the views from `geopolars-core`
  (`List`, `ValidCoords`, ...).
- It calls the kernel, which returns a Polars array.
- It turns kernel errors into a `PolarsError`.

`src/expr/is_ccw.rs` is a simple example.

## Implementation tiers

Every operation is implemented at one of three tiers:

- **A-tier**: a native Rust kernel.
- **B-tier**: a Rust kernel that forwards to an external library.
- **C-tier**: plain Polars expressions.

Which to pick:

- Prefer **A-tier** for complex operations.
- Prefer **C-tier** for simple operations: ones that don't need `list.eval`,
  or that take very few Polars nodes to express
  (not counting the boilerplate to reach the underlying storage).
- Fall back to **B-tier** when writing our own kernel isn't worth it, or is very complicated.

But don't take my word as gospel! Benchmark, benchmark, benchmark!

### A. Native kernel

This functionality has a dedicated Rust kernel in `geopolars-compute`'s `native` module,
working directly on our storage.
This makes them very fast, but they cannot be subdivided at plan-time.
These are ideally element-wise if possible.
Before implementing one, read the [geopolars-compute README](../geopolars-compute/README.md#writing-kernels) first.

### B. Forward to external

Like A-tier, but the Rust kernel hands our data to an external library,
putting it into that library's own data type.
This usually means at least one allocation and copy pass
for every item processed this way, so prefer A-tier where it is feasible.
In return we don't have to re-implement (and maintain) well-tested algorithms.
Kernels forwarding to GEOS live in `geopolars-compute`'s `geos` module.
Before implementing one, read the [geopolars-compute README](../geopolars-compute/README.md#writing-kernels) first.

Examples are:

- [PROJ](https://proj.org/), to reproject coordinates.
  `to_crs` hands it each run of valid coordinates in a single call through the `proj-sys` C API,
  and only retries the coordinates PROJ failed on one at a time, to get their error
  (see `geopolars-compute/src/proj.rs`).
  Prefer this over one call per coordinate.
- [geographiclib](https://geographiclib.sourceforge.io/), to measure along an ellipsoid.
- The [geo ecosystem](https://georust.org/)
  (which we refer to as *rsgeo* in this crate to avoid confusion with our own `geo`),
  which has very fast implementations for many geo algorithms,
  but whose data encoding is not compatible with ours.
  In addition, rsgeo has a couple of limitations:
  - it only supports 2D operations
  - no M (measure) values
  - no CRS-awareness on a data level (this has to be carried separately)
  - no curved geometries

### C. Polars expressions

This functionality can be expressed directly using existing Polars nodes.
An example: The mean coordinate.

We can directly access the individual coordinate columns,
and use Polars' mean implementation on it.
This way you never re-invent the wheel,
and the planner will be able to apply further optimisations.
It only pays off for simple operations, though:
every extra node costs, and nested `list.eval` runs per row,
which loses badly to an A-tier kernel.

A C-tier operation may also be put together from other operations,
including ones that call an A-tier or B-tier kernel. `rotate_about_centroid`, for example,
hands `centroid` (A) as the origin to the `affine_about` kernel (A).
It adds no kernel of its own, so it is C-tier, but its speed is that of the kernels it calls.

You might want to change exactly what you want to put here using `pipe_with_dtype`
(through `on_geometry`) to change what you wish to lower it to based on the metadata.
For that, you can dispatch on the input's geometry through `on_geometry` in `geo/_dispatch.py`.
This wraps `Expr.pipe_with_dtype`, so the callback runs once at plan time
with the concrete geometry dtype and returns a plain expression for that geometry only.
The dtype carries the column's metadata (such as the CRS).
A callback that builds a geometry has to pass it on with `_with_metadata_of`,
or the result silently loses its CRS (`tests/unit/operations/test_metadata.py` checks this).
`on_geometry_pair` does the same for operations between two geometries.
Wrong input (not a geometry, or a geometry we don't support) therefore fails while the plan is built,
rather than partway through execution.
The dispatch also makes it so that you can pass a `List[T]`, and you get a `List[Res]` back.
The callback can return Polars expressions or a kernel call into Rust,
imported using `register_plugin_function`.

### Switch implementation depending on geometry

You can use the same trick with `pipe_with_dtype` that we use with a C-tier implementation
to switch implementation based on the geometry/metadata.
This is useful for:

- early exits (e.g. for counting coordinates,
  'return row count if counting points (native), or use my kernel x otherwise' ).
- switching kernels based on geometry or metadata (e.g. for `area`, forward to `area_planar`
  for equal projection CRSes, but use `area_geodesic` for other stuff)
