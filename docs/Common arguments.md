# Common Arguments

There are a few arguments that show up in many expressions.
Rather than repeating them everywhere (and risking doc drift), they are all described here.

The sections describe the arguments as the target API.
A *Legacy* note names the argument a section replaces (from the old/current design).
The table at the end lists arguments that only a few expressions share, as they are now.

## Errors

### `on_row_error: Literal["null", "drop", "raise"] = "null"`

Some kernels can error depending on the data in a row.
Something could be out of bounds, or a line might cross itself where an expression requires it not to.
With the `on_row_error` variable, you can set what to do if that happens.

- `on_row_error="null"` (default) instructs the kernel to simply mark the resulting row as null (invalid).
  The total rows coming out of the expression then does not change.
- `on_row_error="drop"` tells it to simply omit that row from the result.
  The amount of rows in the output can be the same if no errors occurred, or lower if they did.
- `on_row_error="raise"` stops processing, and returns the error that was raised.
  This is useful if you want to inspect the exact error that was raised.

> [!NOTE]
> Legacy: this replaces `ignore_per_row_errors: bool = False`.

### `early_geos_compat_check: bool = True`

GEOS supports neither empty points inside a geometry nor any other NaN or infinite x or y.
Operations involving such geometries would therefore always end up as an error.
By setting `early_geos_compat_check=True`,
the kernel will check for such values when creating the GEOS representation of a geometry,
allowing it to bypass other calculations as it is invalid anyway.
Setting this value to `early_geos_compat_check=False` forces it to be put into GEOS anyway.

TODO: why would we want to set it to false again?

> [!NOTE]
> Legacy: this replaces `early_finite_check: bool = False`.

## CRS

### `crs: str | None = None`

The CRS to label the output with, in any form PROJ accepts
(PROJJSON, WKT or an `AUTH:CODE` such as `"EPSG:4326"`).

For a constructor built from parts, it sets the CRS of parts that declare none,
and refuses parts that already declare another (`MetadataKwargs::apply` in `src/geoarrow/crs.rs`).
It only labels, and never reprojects (except for the kernel that actually projects, of course).

Related: `shortest_line_between(in_crs=None)` reprojects both sides to `in_crs` before measuring.
It is marked experimental.

### `allow_degree_distances: bool = False`

This function only really makes sense when used with coordinates that actually map to a distance.
A CRS that uses degrees would get distances in degrees, which is most of the time not what you want.
By default with `allow_degree_distances=False`,
these functions will raise a (plan-time) error when this is the case.
By setting `allow_degree_distances=True`, you override this behaviour.

## Comparing coordinates

### `abs_tol: float | None = None`, `rel_tol: float | None = None`

These are used to denote tolerances for equality/coincidence checks.
These are modelled after Polars' arguments of the same name.
Setting either only requires them to be close, as in `pl.Expr.is_close`.
By default (both `None`), they have to be *exactly* equal (`0.0`).
Values are compared *on every axis*, not as a vector magnitude.
Neither value can be negative or NaN. This will return an error at plan time.

### `compare_z: bool = True`, `compare_m: bool = False`

These flags control whether a certain axis is included in a comparison (e.g. for equality, or distance).
The x and y axes are always included, but sometimes it is better to leave the m and z values out.

z is compared by default, m is not (being a measure rather than a position).
Either flag is ignored for a geometry without that axis.

> [!NOTE]
> Legacy: this replaces `unique_z: bool = True`, `unique_m: bool = False`.

## Building and repairing geometries

### `close: CloseRings | None = None`

Closes open rings before they are checked, as `close_rings` does.
`CloseRings(snap_z=False, snap_m=False)` carries the flags of `close_rings`.
With `close=None`, an open ring is not a ring, so its polygon is null.

### `null_parts: Literal["drop", "null"] = "null"`

This argument is used in constructors,
and determines what happens when one of the parts or a coordinate is invalid.

- `null_parts="drop"` simply filters them. The other parts of the row are still added.
  What happens to linestrings and rings that are now too short is set by `line_too_short`.
- `null_parts="null"` makes the resulting row null.

(TODO: maybe add an 'empty' option here too? not a priority)

### `line_too_short: Literal["null", "keep", "empty"] = "null"`

Some operations remove coordinates. This can leave a linestring too short (single vertex),
or a ring that does not enclose any area (<4 vertices, due to needing to repeat the last vertex).
This argument says what should happen to such a thing.

- `line_too_short="null"` makes this entire geometry (row) null.
- `line_too_short="keep"` keeps the line as it is. This can cause invalid GeoArrow.
- `line_too_short="empty"` turns the line into an empty line.
  For a single linestring, this makes the outer geometry an empty geometry.
  For a polygon, this means the polygon will now have an empty ring.
  If there are still non-empty holes,
  what is done to an empty outer ring depends on the `empty_polygon_exterior` value.

> [!NOTE]
> Legacy: this replaces `too_short: Literal["null", "keep", "empty"] = "null"`.

### `empty_polygon_exterior: Literal["keep", "null", "raise"] = "null"`

GeoArrow allows you to have an empty polygon exterior but a non-empty hole.
Most of the time this is not a meaningful polygon though.

- `empty_polygon_exterior="null"` (the default) makes kernels check for these.
  If such a polygon is produced, it is turned into an error value `null` instead.
- `empty_polygon_exterior="raise"` checks too, but raises an error instead.
- `empty_polygon_exterior="keep"` allows a kernel to create these.
  This is valid GeoArrow, but not valid for OGC.
  Running a polygon like this through other expressions might give unexpected results.

## Output geometry

### `reorient_rings: Literal["no", "ccw", "cw"] = "no"`

Some functions create or modify rings, such as constructors or conversions from GEOS.
These might be in a different orientation from the rest of your geometry.
In these cases, you can ask what should be done with the newly added rings:

- `reorient_rings="no"` keeps the rings as-is. This is the cheapest option.
- `reorient_rings="ccw"` forces the rings to be oriented counter-clockwise.
- `reorient_rings="cw"` forces the rings to be oriented clockwise.

> [!NOTE]
> Legacy: this replaces `ccw_outer: bool = False`.

### `reverse_hole_direction: bool = True`

This complements the `reorient_rings` argument,
so it only affects newly transformed/built/converted rings.
If `reverse_hole_direction=True`, then holes are made to go the other direction from exterior rings.
I.e. if `reorient_rings="ccw"`, then the holes are made to go clockwise.

### `output_type: Literal["single", "multi"] = "single"`

Some operations can result in a shape falling apart into multiple.
This is mostly a concern when using GEOS kernels.
For example, taking the difference between two polygons can cause the polygon to be cut into two.
GeoPolars does not have an 'any geometry' datatype (yet),
so this argument decides what happens in such a case.

- `output_type="single"` expects every output to be a single geometry.
  Anything that results in multiple geometries is considered an error.
  How this error is then handled depends on the setting of the `on_row_error` argument.
- `output_type="multi"` always outputs a `Multi*` (e.g. a `MultiPolygon` instead of a `Polygon`,
  even if the input types are only polygons). This is more flexible,
  but in many cases might make your data harder to work with.

If the input geometry is already a `Multi*` (e.g. `MultiPolygon`), this argument has no effect.
These will always output a `Multi*`.

TODO: See if we can give this a better name that better reflects its use case?
Current name could imply that it gives a polygon as output for a multipolygon if `"single"`,
but that is not a thing.

> [!NOTE]
> Legacy: this replaces `allow_splitting: bool = False`.

## Affine transformations

### `origin: Origin | IntoExprColumn = (0.0, 0.0, 0.0)`

The point an affine transformation is done about:
one point for the whole column, as `(x, y)` or `(x, y, z)` (a `z` left out being 0),
or a `geoarrow.point` column with a point for every geometry.

### `unit: Literal["deg", "rad"] = "deg"`

The unit of an angle: degrees or radians.

> [!NOTE]
> Legacy: this replaces `unit: Literal["deg", "pi"] = "deg"`,
> where `"pi"` meant multiples of π radians rather than radians.

## Shared by two or three expressions

These are not (yet) common, but already share a name and meaning across expressions.
They are described as they are now.

| argument | default | taken by | meaning |
| - | - | - | - |
| `allow_non_polygons` | `False` | `area`, `area_planar`, `area_geodesic` | give a non-polygon `0.0` rather than refusing it |
| `allow_non_lines` | `False` | `length` | as `allow_non_polygons`, for non-lines |
| `method` | `None` | `distance`, `length` | `"planar"` or `"geodesic"`; picked by the CRS if not given |
| `ignore_z`, `ignore_m` | `False` | `filter_coordinates`, `hausdorff_line_between` | leave that axis out of the finite check |
| `action` | `"drop"` | `filter_coordinates`, `fix_out_of_bounds` (inside `OutOfBounds`) | what happens to a selected coordinate: `"drop"`, `"set_empty"`, `"set_null"`, and `"clamp"` for `fix_out_of_bounds` |
| `out_of_bounds` | `None` | `is_valid_geoarrow`, `is_valid_ogc`, and their `null_out_invalid_*` | check coordinates against the bounds `fix_out_of_bounds` would fix |
| `check_null_coordinates` | `True` | `is_valid_geoarrow`, `is_valid_ogc`, and their `null_out_invalid_*` | look for null values below the row |
| `quadrant_segments`, `join_style`, `mitre_limit` | `8`, `"round"`, `None` | `buffer`, `offset_curve` | the shape of the curve GEOS draws |
| `dtype` | required | `from_wkb`, `from_wkt` | the concrete geometry type to decode into |
