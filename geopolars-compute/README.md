# geopolars-compute

Kernels for GeoArrow.
They take the views of [`geopolars-core`](../geopolars-core/README.md),
and return Polars arrays.

The crate holds:

- the metrics that kernels measure with (`Planar`, `GeodesicMetric`).
- kernels for simple operations implemented directly on the arrow type in `native`, behind the `native` feature.
- kernels forwarding to GEOS in `geos`, behind the `geos` feature.
- kernels forwarding to rsgeo (the Rust `geo` crate) in `rsgeo`, behind the `rsgeo` feature.
- kernels forwarding to PROJ in `proj`.
- `is_valid_geoarrow`'s kernels in `geoarrow_validity`, which check the coordinates and the rings.
  These need the `native` feature.
- `is_valid_ogc`'s kernels in `ogc_validity`, which run the checks of `native`.
  Then they run those of `geos`, on the rows that pass.
  These need both features.
- the pass that moves every null value below a row out to the row in `inner_nulls`,
  or marks the parts holding one in a mask instead.
- the fold of each level's mask up to the rows in `passing`.
  `inner_nulls` and the validity kernels use it.

## Writing kernels

### Valid input

A kernel can always assume its input is valid according to the GeoArrow spec.
Only the outer level may have nulls.
All other levels, including the coordinate values, contain none.
There is therefore no need to branch on what to do with nulls below the rows.
Making sure of this is up to whoever builds the views.
The one exception is the view inside a `MaskedParts`:
the parts unset in its mask may hold arbitrary values where the input had a null.
Only code that takes a `MaskedParts` reads that view.
It skips those parts, or masks out what it computed for them.
It may hand the view to a plain kernel only if that kernel gives one value per part,
from that part alone, as the mask then drops the values of the unset parts.
Every kernel in turn has to give valid output.

This does **not** shield you from NaNs!
A valid Polygon cannot have null coordinates, but it might still have e.g. a NaN x-coordinate.
They show up in two ways.

- Individual NaN values (e.g. `x = 32.1, y = NaN`).
- *Empty* points (NaN on every axis).

How to handle these is up to the kernel.
The rules for single coordinates (empty, not finite, outside the bounds) are in `native/coordinate_selection.rs`:
reuse those rather than writing them again.

### Untrimmed views

Views from `new` are trimmed, and a view a kernel builds has to be too.
A cheap kernel may take an `Untrimmed` view instead.
See the [geopolars-core README](../geopolars-core/README.md#trimming).

### Branching inside a loop

Try to keep branching outside of kernel loops.
If you can already do it outside, do it outside.
Sometimes you conditionally need to do something extra
(e.g. also check for overlaps in a validation loop).
Write that as a generic function instead, and dispatch to its specific versions.
The `with_const_flags!` macro is very useful here.
Be careful you don't blow it up with combinatorics though.
Describe each flag once, in a ``## `FLAG` `` sub-section under `# Const generic flags` in the module doc (see `native/count_coordinates.rs`).
A runtime `bool` that dispatches to it has the same name in lowercase, so it needs no doc of its own.

### Fallible functions

How to handle errors depends on where they are raised.

- An error before looping over the rows would make *all rows* null.
  Return it, so the user sees it.
- An error in a single row does not affect the other rows.
  Switch on an `ignore_per_row_errors` flag.
  Errors are the exception, and we cannot pre-filter for them.
  So here it is fine to check the `bool` in the hot path, rather than make it a const generic.
  If `ignore_per_row_errors=false`, which is the default, return the error.
  This gives the user a way to inspect errors if they want.
  If `ignore_per_row_errors=true`, mark the row as null instead (not as an empty geometry!).
  The `geopolars-core` output functions take the rows that failed for this.
- An accumulator-like function combines many rows into far fewer results.
  Ask the caller how to handle errors here, with an `IGNORE_ERRORS` const `bool`.
  The default should be to return the error.

### Exiting early before GEOS

Converting a geometry to GEOS and calling GEOS on it is expensive.
A GEOS kernel that can tell the result of a row without GEOS therefore gives it before converting,
such as a null or empty geometry on either side of a binary operation.
That row then reaches GEOS on neither side, where it could also fail (see `geos/shortest_line.rs`).

### GEOS kernel standard shape

All kernels should have this shape:

```text
geoarrow -> GeosArray -> (geos kernel) -> GeosArray -> geoarrow

GeosArray = {Vec<GeosGeom>, validity bitmap }
GeosGeom = enum {geos geometry, empty, null, left_unchanged, right_unchanged}
```

We can create a standard way to do the shared steps.
