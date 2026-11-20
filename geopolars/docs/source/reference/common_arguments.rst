================
Common arguments
================

There are a few arguments that show up in many expressions.
Rather than repeating them everywhere (and risking doc drift), they are all described here.
An expression uses these where they fit,
rather than defining a new argument for the same thing in slightly different words.

The sections describe the target API.
A *Legacy* note names the argument a section replaces (from the old/current design).
The table at the end lists arguments that only a few expressions share, as they are now.

Errors
~~~~~~

.. _arg-on_row_error:

`on_row_error: Literal["null", "raise"] = "raise"`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some kernels can error depending on the data in a row.
Something could be out of bounds, or a line might cross itself where an expression requires it not to.
With the `on_row_error` variable, you can set what to do if that happens.

- `on_row_error="raise"` stops processing, and returns the error that was raised.
  This is useful if you want to inspect the exact error that was raised.
  This is the (strict) default to agree with Polars.
- `on_row_error="null"` instructs the kernel to simply mark the resulting row as null.
  The total rows coming out of the expression then does not change.

This parameter does not affect how error handling outside the per-row part is done.
If there is an error in the setup, it can still fail the entire query even if `on_row_error="null"` is set.

.. note::

   Legacy: this replaces `ignore_per_row_errors: bool = False`.

.. _arg-early_coord_check:

`early_coord_check: Literal["finite", "off", "finite_and_empty"]`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most operations will return a non-finite result if the input is also non-finite.
This argument controls an early check for this in this kernel.
Checking this requires an extra pass over the geometry though.
See this as a safety check you can turn off for extra speed in some cases.
Turning this off in data with very few NaNs might be faster, but if your data contains a lot of NaNs,
you might be better off leaving this one as this can skip expensive calculations.

- `early_coord_check="off"` skips this extra pass.
- `early_coord_check="finite"` (default for native kernels) means that the kernel will
  check for such values in the operation. If it finds one, it will mark it as an error,
  which can be picked up by `on_row_error`. An empty point does not trigger this.
- `early_coord_check="finite_and_empty"` (default for GEOS kernels) does the same as the above,
  but also goes to `on_row_error` for an empty point inside a geometry.

This argument serves an important purpose for kernels that forward to GEOS.
GEOS supports neither empty points inside a geometry nor any other NaN or infinite x or y
(see "NaN and infinity" in `docs/source/user-guide/geo/concepts/nulls-nan-and-empty.md`).
Here the check is embedded into the conversion into a GEOS geometry.
With checking, it will will trigger `on_row_error`,
while on skipping it might give some unexpected result.
You read that right, setting `early_coord_check="off"` can also silently affect the results,
due to GEOS handling invalid states differently (and sometimes inconsistently).
An example would be GEOS silently returning (4, 0) for a distance of 0.5 on a line with a NaN vertex.

With `early_coord_check="finite"`, an empty point inside a geometry goes to GEOS.
What GEOS does with it depends on the operation, and often on where the point is:

.. list-table::
   :header-rows: 1

   * - Operation
     - An empty point inside a geometry
   * - `buffer`, `offset_curve`
     - is left out
   * - `simplify`
     - is left out, but kept as the first vertex
   * - `simplify` with `preserve_topology=True`
     - fails the row
   * - `intersection`, `difference`, `shortest_line_between`
     - fails the row
   * - `relate`
     - fails the row, unless it is the first vertex
   * - `convex_hull`
     - is left out, but gives a null as the first vertex
   * - `line_interpolate_point`
     - gives a wrong point
   * - `is_topologically_equal`
     - gives a wrong answer

A geometry that is itself the empty point passes either check,
and works as you'd expect it to, as it is a standard GEOS type.

If this is a function that also takes `include_m`/`include_z`,
the check only includes the coordinates that these flags specify.

.. note::

   Legacy: this replaces `early_finite_check: bool = False`.

CRS
~~~

.. _arg-crs:

`crs: str | None = None`
^^^^^^^^^^^^^^^^^^^^^^^^

The CRS to label the output with,
in any form described under "Declaring a CRS" in `docs/source/user-guide/geo/concepts/crs.md`.

For a constructor built from parts, it sets the CRS of parts that declare none,
and refuses parts that already declare another (`MetadataKwargs::apply` in `src/geoarrow/crs.rs`).
It only labels, and never reprojects (except for the kernel that actually projects, of course).

Related: `shortest_line_between(in_crs=None)` reprojects both sides to `in_crs` before measuring.
It is marked experimental.

.. _arg-distance_method:

`distance_method: Literal["auto", "planar", "geodesic"] = "auto"`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Passed to functions that *measure* a distance.
Why distances in degrees are rarely what you want is described under "Geographic and projected"
in `docs/source/user-guide/geo/concepts/crs.md`.

- `distance_method="planar"` naively looks at the coordinate data as-is.
  This makes sense if the CRS already defines its distances in a linear unit (e.g. metres, feet),
  but otherwise you might end up with distances in degrees.
- `distance_method="geodesic"` gets the distances on the CRS-defined ellipsoid.
- `distance_method="auto"` (default) looks at the CRS to switch between the other two modes.
  If the CRS is a geographic one, it uses the `"geodesic"` option,
  while it uses `"planar"` on anything that has a projected CRS, or no CRS at all.
  On a projected CRS, the geodesic and planar distances are usually very close enough,
  so it makes sense to use the faster option.

.. note::

   this replaces `unique_z: bool = True`, `unique_m: bool = False`,
   as well as `ignore_z` and `ignore_m`.
   It replaces method: Literal["planar", "geodesic"] | None = None.
   None becomes "auto".
   Today's default sends any declared CRS through "geodesic", so the result changes for a projected CRS.

.. _arg-allow_geographic:

`allow_geographic: bool = False`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Passed to kernels that take a distance as an *input*.
Some methods really only make sense to do in a planar reference frame.
Operations like buffer, simplify and offset_curve take distances in the argument.
By default, we want to be strict. Raising an error at plan-time is clear.

- `allow_geographic = False` raises an error at plan time if you try to run
  a function with data in a geographic CRS.
- `allow_geographic = True` shows that you know what you are doing,
  and intentionally want to express distances in degrees.

Comparing coordinates
~~~~~~~~~~~~~~~~~~~~~

.. _arg-abs_tol:

`abs_tol: float | None = None`, `rel_tol: float | None = None`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are used to denote tolerances for equality/coincidence checks.
These are modelled after Polars' arguments of the same name.
Setting either only requires them to be close, as in `pl.Expr.is_close`.
By default (both `None`), they have to be *exactly* equal (`0.0`).
Values are compared *on every axis*, not as a vector magnitude.
Neither value can be negative or NaN. This will return an error at plan time.

.. _arg-include_z:

`include_z: bool = True`, `include_m: bool = False`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These flags control whether a certain axis is included in a comparison
(e.g. equality, distance).
The x and y axes are always included, but sometimes it is better to leave the z and m values out.

z is compared by default, m is not (being a measure rather than a position).
Either flag is ignored for a geometry without that axis.

This also ties into `early_coord_check`. See its description for more details.

.. note::

   Legacy: this replaces `unique_z: bool = True`, `unique_m: bool = False`,
   as well as `ignore_z` and `ignore_m`.

Building and repairing geometries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _arg-close:

`close: CloseRings | None = None`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Closes open rings before they are checked, as `close_rings` does.
`CloseRings(snap_z=False, snap_m=False)` carries the flags of `close_rings`.
With `close=None`, an open ring is not a ring, so its polygon is null.

.. _arg-null_parts:

`null_parts: Literal["drop", "null"] = "null"`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This argument is used in constructors and functions that add to a geometry like `add_parts`.
It determines what happens when one of the parts or a coordinate is null.

- `null_parts="drop"` simply filters them. The other parts of the row are still added.
  What happens to linestrings and rings that are now too short is set by `short_lines`.
- `null_parts="null"` makes the resulting row null.

.. TODO: maybe add an 'empty' option here too? not a priority

.. _arg-short_lines:

`short_lines: Literal["null", "keep", "empty"] = "null"`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some operations remove coordinates. This can leave a linestring too short (single vertex),
or a ring that does not enclose any area (<4 vertices, due to needing to repeat the last vertex).
This argument says what should happen to such a thing.

- `short_lines="null"` makes this entire geometry (row) null.
- `short_lines="keep"` keeps the line as it is. This can cause invalid OGC geometries.
- `short_lines="empty"` turns the line into an empty line.
  For a single linestring, this makes the outer geometry an empty geometry.
  For a multilinestring, this makes that one line an empty line.
  For a (multi)polygon, this means the polygon will now have an empty ring.
  If there are still non-empty holes,
  what is done to an empty outer ring depends on the `empty_polygon_exterior` value.

.. note::

   Legacy: this replaces `too_short: Literal["null", "keep", "empty"] = "null"`.

.. _arg-empty_polygon_exterior:

`empty_polygon_exterior: Literal["keep", "null", "error", "empty"] = "null"`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GeoArrow allows you to have an empty polygon exterior but a non-empty hole.
Most of the time this is not a meaningful polygon though.

.. list-table::
   :header-rows: 1

   * - `empty_polygon_exterior`
     - a polygon a kernel would give an empty exterior and a hole with vertices
   * - `"null"` (the default)
     - is null
   * - `"error"`
     - goes to `on_row_error`
   * - `"empty"`
     - loses its holes, so it is the empty polygon
   * - `"keep"`
     - keeps its holes: valid GeoArrow, but not valid OGC

With `"keep"`, running such a polygon through other expressions might give unexpected results.

.. note::

   Legacy: this replaces `empty_exterior: Literal["null", "keep"] = "null"`.
   Its `"keep"` is now `"empty"`.

Output geometry
~~~~~~~~~~~~~~~

.. _arg-new_ring_orientation:

`new_ring_orientation: Literal["keep", "ccw", "cw"] = "keep"`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some functions add new rings,
which might run another way than the rest of your geometry
(see "New rings" in `docs/source/user-guide/geo/concepts/winding-order.md`).
In these cases, you can ask what should be done with the newly added rings:

- `new_ring_orientation="keep"` keeps the rings as-is. This is the cheapest option.
- `new_ring_orientation="ccw"` forces the rings to be oriented counter-clockwise.
- `new_ring_orientation="cw"` forces the rings to be oriented clockwise.

What counts as a new ring:

- GEOS output: all rings.
- Constructors: all rings.
- add_holes: only the added holes.

.. note::

   Legacy: this replaces `ccw_outer: bool = False`.
   `ccw_outer=True` is now `new_ring_orientation="ccw"`.

.. _arg-new_holes:

`new_holes: Literal["opposite", "same"] = "opposite"`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This complements the `new_ring_orientation` argument,
so it only affects newly transformed/built/converted rings.
If `new_holes="opposite"`, then holes are made to go the other direction from exterior rings.
I.e. if `new_ring_orientation="ccw"`, then the holes are made to go clockwise.

This option does nothing if `new_ring_orientation="keep"`

.. _arg-promote_to_multi:

`promote_to_multi: Literal["never", "always"] = "never"`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some operations can result in a shape falling into several parts.
For example, taking the difference between two polygons can cause the polygon to be cut into two.
A column holds a single kind of geometry
(see "One kind per column" in `docs/source/user-guide/geo/concepts/geometry-types.md`),
so this argument decides what happens in such a case.
This is mostly a concern when using GEOS kernels.

- `promote_to_multi="never"` expects every output to be a single geometry.
  Anything that results in multiple geometries is considered an error.
  How this error is then handled depends on the setting of the `on_row_error` argument.
- `promote_to_multi="always"` always outputs a `Multi*` (e.g. a `MultiPolygon` instead of a `Polygon`,
  even if the input types are only polygons). This is more flexible,
  but in many cases might make your data harder to work with.

If the input geometry is already a `Multi*` (e.g. `MultiPolygon`), this argument has no effect.
These will always output a `Multi*`.

.. note::

   Legacy: this replaces `allow_splitting: bool = False`.

Affine transformations
~~~~~~~~~~~~~~~~~~~~~~

.. _arg-origin:

`origin: Origin | IntoExprColumn = (0.0, 0.0, 0.0)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The point an affine transformation is done about:
one point for the whole column, as `(x, y)` or `(x, y, z)` (a `z` left out being 0),
or a `geoarrow.point` column with a point for every geometry.

.. _arg-unit:

`unit: Literal["deg", "rad"] = "deg"`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The unit of an angle: degrees or radians.

.. note::

   Legacy: this replaces `unit: Literal["deg", "pi"] = "deg"`,
   where `"pi"` meant multiples of π radians rather than radians.

Arguments that come together
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some arguments make another one necessary.

- An expression that can remove points takes `short_lines`,
  as dropping coordinates or parts can make rings and linestrings too short.
  So one that takes `null_parts` takes `short_lines` too.
- An expression that takes polygons (or multipolygons) and `short_lines`
  takes `empty_polygon_exterior` too,
  as removing points from a ring can otherwise invalidate a polygon.
- An expression that takes `new_ring_orientation` and can give polygons takes `new_holes` too.
- An expression that takes `promote_to_multi` takes `on_row_error` too,
  as with `"never"`, a result in several parts is an error.
- An expression that takes `early_coord_check` takes `on_row_error` too,
  as the check hands what it finds to it.
- An expression that takes `empty_polygon_exterior` takes `on_row_error` too,
  as `empty_polygon_exterior="error"` hands the error to it.

Shared by two or three expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are not (yet) common, but already share a name and meaning across expressions.
They are described as they are now.

.. list-table::
   :header-rows: 1

   * - argument
     - default
     - taken by
     - meaning
   * - `allow_non_polygons`
     - `False`
     - `area`, `area_planar`, `area_geodesic`
     - give a non-polygon `0.0` rather than refusing it
   * - `allow_non_lines`
     - `False`
     - `length`
     - as `allow_non_polygons`, for non-lines
   * - `action`
     - `"drop"`
     - `filter_coordinates`, `fix_out_of_bounds` (inside `OutOfBounds`)
     - what happens to a selected coordinate: `"drop"`, `"set_empty"`, `"set_null"`,
       and `"clamp"` for `fix_out_of_bounds`
   * - `out_of_bounds`
     - `None`
     - `is_valid_geoarrow`, `is_valid_ogc`, and their `null_out_invalid_*`
     - check coordinates against the bounds `fix_out_of_bounds` would fix
   * - `check_null_coordinates`
     - `True`
     - `is_valid_geoarrow`, `is_valid_ogc`, and their `null_out_invalid_*`
     - look for null values below the row
   * - `quadrant_segments`, `join_style`, `mitre_limit`
     - `8`, `"round"`, `None`
     - `buffer`, `offset_curve`
     - the shape of the curve GEOS draws
   * - `dtype`
     - required
     - `from_wkb`, `from_wkt`
     - the concrete geometry type to decode into
