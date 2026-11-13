# Kernel implementation guide

This document serves as a guide for implementing compute kernels in `geopolars-compute`.
The information contained here is for both `native` kernels and `GEOS` forwarding kernels.
Both of these have additional implementation guides,which can be found as
the [Native kernel implementation guide]("../docs/Native kernel implementation guide.md")
and the [GEOS kernel implementation guide]("../docs/GEOS kernel implementation guide.md").

## Valid input

Kernels should only take valid input as [view types]("../docs/View Types.md").
These are guaranteed to follow the GeoArrow spec, where only the outer level may have nulls.
All other levels, including the coordinate values, cannot contain nulls.
This means that you do should not add any intermediate validity checking.
Ensuring validity is the task of whoever builds the views.
This puts the burden on the kernel to make sure that the output is valid too though.

Note that validity does **not** shield you from *NaNs* or *(nested *empty* geometries.
A valid Polygon cannot have null coordinates, but it might still have e.g. a NaN x-coordinate.
They show up in two ways.

- Individual NaN values (e.g. `x = 32.1, y = NaN`).
- *Empty* points (NaN on every axis).
- A polygon might also have an empty ring (or similar, depending on the geometry type).

How to handle these is up to the kernel.
Rules and helpers for single coordinates (empty, not finite, outside the bounds)
you can use are in `native/coordinate_selection.rs`.

## Kernels should work on collections, not on single items

Avoid going to `ScalarPolygon` etc. if you don't really need them.
Prefer working directly on the array data.
For nested types, we'd have to start saving the offsets and nestings, which is all extra overhead.
That way we only need to have helpers on one side too :)

## Use view types

Write kernels to take in view types, and output view types,
rather than using type-erased ones like `Box<dyn Array>`.
This makes it clear what type of data the kernel expects,
which functions both in documenting the code as well as reducing the chance to introduce bugs.

## Error handling

How to handle errors depends on where they are raised.

- An error that would invalidate all rows (i.e. raised before looping) should be returned.
  Such errors should try to be caught at plan time already if possible.
  This includes some common arguments like `allow_degree_distances=false`.
- An error in a single should follow the rules outlined by `on_row_error`.
- An accumulator-like function combines many rows into far fewer results.
  Ask the caller how to handle errors here, with an `IGNORE_ERRORS` const `bool`.
  The default should be to return the error.

## MaskedParts

The one exception to the validity rule is the view inside a `MaskedParts`.
The parts unset in its mask may hold arbitrary values where the input had a null.
Only code that takes a `MaskedParts` reads that view.
It skips those parts, or masks out what it computed for them.
It may hand the view to a plain kernel only if that kernel gives one value per part,
from that part alone, as the mask then drops the values of the unset parts.
Every kernel in turn has to give valid output.

## Untrimmed views

Views from `new` are trimmed, and a view a kernel builds has to be too.
A cheap kernel may take an `Untrimmed` view instead.
See the [geopolars-views README](../geopolars-views/README.md#trimming).
