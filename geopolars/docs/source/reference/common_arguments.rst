================
Common arguments
================

There are a few arguments that show up in many expressions.
Rather than repeating them everywhere (and risking doc drift), they are all described here.
An expression uses these where they fit,
rather than defining a new argument for the same thing in slightly different words.

The sections describe the target API.
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

.. _arg-early_coord_check:

`early_coord_check: Literal["finite", "off", "finite_and_empty"]`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This argument controls a shortcut path to allow faster execution in some cases.
It controls a pass over the coordinates of a row to find non-finite and/or empty values
in operations that do a lot of work that would be thrown away if there is a NaN.
This allows it to save needing to do that work in the first place.
It costs a little to check it for every row, so that some rows do not have to be worked on at all. 
Turn it off when you know your data holds (almost) no invalid data.
Leave it on when it holds many, as skipping their work is then the bigger saving.

- `early_coord_check="off"` skips this extra pass.
- `early_coord_check="finite"` (default for native kernels) checks for a NaN or infinite `x` or `y`.
  The empty point (which is represented as an all-NaN) is allowed.
  It then either marks the row as NaN for single-value returns, or errors if the row result is not a float.
  For example, a GEOS kernel marks the row as an error instead.
  `on_row_error` handles the error case.
- `early_coord_check="finite_and_empty"` (default for GEOS kernels) does the same as the above,
  but also for an empty point inside a geometry.
  It will try to honour the 'empty' where possible.

This argument is very useful for kernels that forward to GEOS.
GEOS cannot handle an empty point inside a geometry, nor any other NaN or infinite x or y
(see `Non-finite and inner empties in GEOS-backed kernels <https://github.com/pola-rs/geopolars/blob/main/docs/source/user-guide/geo/concepts/nulls-nan-and-empty.md#non-finite-and-inner-empties-in-geos-backed-kernels>`_ in the user guide).
These kernels run the check while they convert a row into a GEOS geometry.

With the check on, a NaN or infinite coordinate never comes out as a plausible finite number
(see `Non-finite <https://github.com/pola-rs/geopolars/blob/main/docs/source/user-guide/geo/concepts/nulls-nan-and-empty.md#non-finite>`_ in the user guide).
Turning it off gives the same result for most rows,
but it is not guaranteed to always give the same values.
A native kernel may give a different non-finite outcome,
such as infinity instead of NaN, or an error that the check would have caught first.

.. note::

   With `early_coord_check="off"`, a GEOS kernel cannot guarantee that the result stays non-finite,
   as GEOS sometimes handles these internally inconsistently.
   For example, it silently returns (4, 0) for a distance of 0.5 along a line with a NaN vertex.

With `early_coord_check="finite"`, an empty point inside a geometry goes to GEOS.
What GEOS does with it depends on the operation, and often on where the point is:

.. TODO: these should be on the expressions they describe, not here.

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


If this is a function that also takes `include_m`/`include_z`,
the check only includes the coordinates that these flags specify.

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

Passed to functions that *measure* a distance or an area.
Why distances in degrees are rarely what you want is described under "Geographic and projected"
in `docs/source/user-guide/geo/concepts/crs.md`.

- `distance_method="planar"` naively looks at the coordinate data as-is.
  This makes sense if the CRS already defines its distances in a linear unit (e.g. metres, feet),
  but otherwise you might end up with distances in degrees.
- `distance_method="geodesic"` gets the distances on the CRS-defined ellipsoid.
- `distance_method="auto"` (default) looks at the CRS to switch between the other two modes.
  It uses `"planar"` on anything that has a projected CRS, or no CRS at all,
  and `"geodesic"` on any other CRS.
  A CRS that is not defined on longitude/latitude, such as a geocentric one,
  has no ellipsoid to measure along, so it is refused at plan time unless `"planar"` is asked for.
  A projected CRS made for measuring such as a UTM zone or a national grid like EPSG:28992
  keeps distances to within about 0.1% inside its area of use,
  so the faster planar distance is close enough there.
  Others do not: Web Mercator (EPSG:3857) stretches distances by 1/cos(latitude),
  about 1.6 times at 52°N.
  Pass `distance_method="geodesic"` for such a CRS.

`area` resolves `"auto"` differently.
It only measures planar without a CRS or in an equal-area one (see `area`).

`centroid` does not measure geodesically yet, so it defaults to `"planar"`.
Its `"auto"` resolves as `length` does for a linestring, and as `area` does for a polygon.
It refuses to measure geodesically at plan time.

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

.. _arg-on_out_of_bounds:

`on_out_of_bounds: Literal["auto", "ignore", "error", "null"] = "auto"`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A coordinate outside the bounds of its CRS (a latitude of 91°, say) is not a place,
so measuring it gives NaN, `0.0`, or a wrong number without warning.
This argument checks for such coordinates.
`crs_bounds` picks the bounds to check against, with the same values as in `fix_out_of_bounds`.
Its default here is `"crs_wrapped_longitude"`, which lets longitudes from 180 to 360 pass.
Data stored from 0 to 360 then measures as it would from -180 to 180,
as does a box across the antimeridian (see `to_polygon`).

An infinite `x` or `y` lies outside the bounds.
A NaN one never does, so neither does an empty point inside a geometry.
Infinite values will be caught by `early_coord_check` instead if the expression uses it (and it is enabled),
as that runs before this check.

- `on_out_of_bounds="auto"` (default) is `"error"` when measuring geodesic,
  and `"ignore"` when measuring planar, which then does not ask PROJ for any bounds.
- `on_out_of_bounds="error"` fails the row, and `on_row_error` handles it.
- `on_out_of_bounds="null"` makes the row null, whatever `on_row_error` says.
- `on_out_of_bounds="ignore"` skips the check, and its extra pass over the coordinates.

A column without a CRS, or with a projected CRS and bounds not taken from its area of use,
has no bounds, so it is not checked.

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

`include_z: bool = True`, `include_m: bool = True`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These flags control whether a certain axis is included in a comparison
(e.g. equality, distance).
The x and y axes are always included, but sometimes it is better to leave the z and m values out.

Both are included by default.
Either flag is ignored for a geometry without that axis.

This also ties into `early_coord_check`. See its description for more details.

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

.. _arg-new_holes:

`new_holes: Literal["opposite", "same"] = "opposite"`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This complements the `new_ring_orientation` argument,
so it only affects newly transformed/built/converted rings.
If `new_holes="opposite"`, then holes are made to go the other direction from exterior rings.
I.e. if `new_ring_orientation="ccw"`, then the holes are made to go clockwise.

This option does nothing if `new_ring_orientation="keep"`

.. _arg-promote_to_multi:

`promote_to_multi: Literal["never", "auto", "always"] = "auto"`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some operations can result in a shape falling into several parts.
For example, taking the difference between two polygons can cause the polygon to be cut into two.
A column holds a single kind of geometry
(see "One kind per column" in `docs/source/user-guide/geo/concepts/geometry-types.md`),
so this argument decides what happens in such a case.
This is mostly a concern when using GEOS kernels.

- `promote_to_multi="never"` expects every output to be a single geometry.
  Anything that results in multiple geometries is considered an error.
  How this error is then handled depends on the setting of the `on_row_error` argument.
- `promote_to_multi="auto"` (default) outputs a `Multi*` where the result can fall into several parts.
  This is decided while the plan is built, from the input kinds and the other arguments, not row by row.
  `buffer`, for example, only looks at `distance`.
  A `distance` column, or one of at most 0, gives a `MultiPolygon` for every input.
  That includes a point, which never splits.
  Each expression documents what `"auto"` means for it.
- `promote_to_multi="always"` always outputs a `Multi*` (e.g. a `MultiPolygon` instead of a `Polygon`,
  even if the input types are only polygons). This is more flexible,
  but in many cases might make your data harder to work with.

An input that is already a `Multi*` (e.g. `MultiPolygon`) always gives a `Multi*`, whatever this argument says.
`intersection` is the exception, as its output kind depends on both inputs.
Its docstring gives the output kind for every pair of inputs.

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

Arguments that come together
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some arguments make another one necessary.

- An expression that can remove points takes `short_lines`,
  as dropping coordinates or parts can make rings and linestrings too short.
  One that takes `null_parts` therefore takes `short_lines` too.
- An expression that takes polygons (or multipolygons) and `short_lines`
  takes `empty_polygon_exterior` too,
  as removing points from a ring can otherwise invalidate a polygon.
- An expression that takes `new_ring_orientation` and can give polygons takes `new_holes` too.
- An expression that takes `promote_to_multi` takes `on_row_error` too,
  as `promote_to_multi="never"` makes a result in several parts an error.
- An expression that takes `early_coord_check` takes `on_row_error` too,
  as a GEOS kernel hands a row that fails the check to it.
- An expression that takes `empty_polygon_exterior` takes `on_row_error` too,
  as `empty_polygon_exterior="error"` hands the error to it.
- An expression that takes `on_out_of_bounds` takes `on_row_error` too,
  as `on_out_of_bounds="error"` hands the error to it.
  It also takes `crs_bounds`, to pick the bounds.

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
   * - `allow_non_polygons`, `allow_non_lines`
     - `False`
     - `area`, `length`
     - give a non-polygon (or a non-line) `0.0` rather than refusing it
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
