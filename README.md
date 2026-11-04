# GeoPolars

In this branch, we are rewriting large chunks of GeoPolars to work much more closely with the rest of Polars.
This is all very much still work-in-progress, and as such this branch is HIGHLY unstable.
Keep an eye on the tracking issues with the dev tag for where we are right now.
We'll need a little bit more time to make things nice, but we're making a lot of progress very quickly,
so stay tuned!

## Overview of Features

GeoPolars has Geo data types as first-class Polars citizens,
  represented using the Arrow Extension format as [GeoArrow](https://geoarrow.org/).
  This has several advantages:

- zero-cost conversion with native polars data types.
- Allows using regular polars operations directly on these new data types
- Full compatibility with streaming engine
- Mismatches, illegal operations etc. can be caught at plan time or expression creation time,
  rather than at runtime.

In addition, GeoPolars aims to be a modern GeoSpatial toolset, offering:

- CRS-aware by default, backed by PROJ
- Support for 3D (z) and measured quantities (m) for most operations
- (De)serialisation with GeoParquet

## Developer Documentation

If you are interested in the design paradigm and architecture used for this plugin,
See the `Docs` directory for more information about the internals of this plugin.

- [A high-level overview of the architecture of the plugin](./docs/Architecture.md)
- [A document that describes what happens when you call an expression, at plan and execution time](./docs/Typical expression flow.md).
  This is accompanied by [a diagram](./docs/expression-flow.dot).
- [What to consider when implementing new or editing existing expressions](./docs/Expression implementation guide.md)
- [The views kernels take, in `geopolars-core`](./geopolars-core/README.md)
- [What to consider when writing kernels, in `geopolars-compute`](./geopolars-compute/README.md)
- [An overview of which expression is implemented at which *tier*](./docs/Tier per expression.md)

## Quickstart

We present two equivalent APIs.
All functionality is is available through both.

```python
# functional
from geopolars import geo

geo.translate("route", (1.0, 2.0))

# expressions using a namespace
import geopolars as gpl

gpl.col("route").geo.translate((1.0, 2.0))
```

Both are fully type-checked, but with an asterisk.
Plain `pl.col("route").geo...` works at runtime,
but a checker cannot see namespaces that `register_expr_namespace`
patches onto `pl.Expr`.
To avoid that, prefer `gpl.col` to keep static typing.

For a list of all the geometry operations, look at the generated documentation!
They are written under the *functional* API, but hold exactly the same for the namespace API.

## Conventions

GeoPolars does not really care about winding order. It will take any,
and keep them in their orientation if that means less copying around is required.
This philosophy is extended to external tools.
If we don't need to transform any data, we'll just keep it in the same order.

## Validity

Every operation expects a *valid* geometry as its input.
If given one, they should (generally) not fail, and give sensible results.

But what does *valid* mean? GeoPolars knows *two* levels, which nest.

The first is validity according to the GeoArrow spec, checked by `geo.is_valid_geoarrow`.

- Only a whole geometry can be null.
  A part, a ring, or a coordinate nested in another geometry cannot be null.
- Every *ring* is closed. Its last vertex is the same as its first.

Checking GeoArrow validity is cheap. It looks at every coordinate once.
It can also check, in that same pass, that every coordinate is finite,
or that it lies inside the bounds of its CRS (`out_of_bounds`).
For example, a latitude has to be between -90° and 90°.
PROJ cannot transform a place that does not exist.
`geo.fix_out_of_bounds` drops or clamps the coordinates that do not.

The second is the OGC Simple Features standard, checked by `geo.is_valid_ogc`.
This is the same standard as followed by GEOS (and therefore shapely, and GeoPandas).
In GeoPolars, a geometry valid by OGC has to be valid by GeoArrow as well.

- Every coordinate is finite. A NaN or an infinity makes every area, length and distance NaN too.
  It also makes it impossible to tell if two shapes cross.
  The empty point is the exception, as a point or in a multipoint.
  GeoArrow writes it as NaN in x and y.
- A linestring has at least 2 different vertices (or none, making it empty).
- A ring has at least 4 different vertices
  (A ring with fewer vertices can only go back and forth, so it has no area).
- The rings of a polygon do not cross themselves or each other, and its holes lie *inside* its exterior.
- The parts of a multipolygon do not overlap.

The OGC standard has one more idea.
A geometry is **simple** when it does not cross or touch itself (`geo.is_simple`).
A linestring may cross itself and still be valid by OGC, but just not *simple*.
For example, a GPS track can pass the same crossing twice.

Checking OGC validity is much more expensive, as it has to compare pairs of segments, and pairs of parts.

OGC allows some geometries that might still surprise you!
An empty geometry is valid by OGC (find them with `geo.is_empty`),
and so is the same vertex twice in a row (`geo.has_repeats_in_succession`).

`geo.null_out_invalid_geoarrow` and `geo.null_out_invalid_ogc` replace invalid geometries with nulls.
See `is_valid_geoarrow` and `is_valid_ogc` for all the rules.

```python
df.select(geo.null_out_invalid_ogc("parcel"))
```

The table below lists every check, the geometries it applies to,
and the expressions that check or repair it.
`null_out_invalid_geoarrow` and `null_out_invalid_ogc` run the same checks as `is_valid_geoarrow`
and `is_valid_ogc`.
A flag next to an x turns that check off.
Without an x, the check only runs when that argument is given.

| Check | Geometry | Standalone | `is_valid_geoarrow` | `is_valid_ogc` | Repair |
| - | - | - | - | - | - |
| Nulls only exist at the outer level (no part, ring, coordinate or axis) | all (Point: only its axes) | | x (`check_null_coordinates`) | x (`check_null_coordinates`) | |
| Every ring is closed: its last vertex is its first, in x, y and z | Polygon, MultiPolygon | | x | x | `close_rings` |
| x and y lie inside the bounds of the CRS | all | | `out_of_bounds` | `out_of_bounds` | `fix_out_of_bounds` |
| There are no NaN/inf values in the coordinates (except 'empty', see geoarrow) | all | | `check_finite_coordinates=True` | x | `filter_coordinates` |
| There are no empty coordinates | all but Point, MultiPoint | | | only allowed in multipoints | `filter_coordinates` |
| A linestring's vertices do not all lie at one place in x and y (`[A, A]` is invalid) (exception: empty) | LineString, MultiLineString[^1] | | | x | |
| Every ring is empty, or every ring has at least 4 vertices that lie elsewhere in x and y than the vertex before (`[A, B, B, A]` is invalid) | Polygon, MultiPolygon[^1] | | | x | |
| Holes lie inside the exterior and outside each other, and rings touch only at points | Polygon, MultiPolygon[^1] | | | x (`check_polygon_self_overlap`) | |
| Rings that touch at points do not cut the area into separate pieces (e.g. a hole touching the exterior twice) | Polygon, MultiPolygon[^1] | | | only with every topology check on | |
| Parts share no area, and touch only at points | MultiPolygon | | | x (`check_multi_polygon_self_overlap`) | |
| Does not cross or touch itself (simple) | all (Point: trivially; LineString: its ends may meet; Polygon, MultiPolygon: every ring; MultiPoint: no two points at the same place; MultiLineString: every part, and parts only meet at their ends) | `is_simple` | | polygons and multipolygons only (`check_ring_self_intersect`) | |
| Is not empty on the outside: has at least one vertex | all (Point: not the empty point) | `is_empty` | | | |
| Is not empty on the inside: not every point is the empty point | all but Point | `is_empty(check_nested=True)` | | | |
| No two vertices in a row lie at the same place | all (Point: always; Polygon, MultiPolygon: within a ring; MultiLineString: within a part) | `has_repeats_in_succession` | | | `remove_repeats_in_succession` |

[^1]: Checked for every part.

## CRS awareness

### Geometries crossing the antimeridian

Things that cross the antimeridian can be represented as having points >180° or <-180°.

## Geometry Operations

## User-defined functions

Because GeoPolars has GeoArrow data types as first-class citizens,
you can use Polars' `map_elements` for a user-defined function (UDF)
like you would for other Polars data types.
There are some small extra things to consider though!

Inside a Python UDF, what the function receives depends on the geometry type.

| Type           | Value passed to the function         |
| -------------- | ------------------------------------ |
| PointXY        | `{'x': 1.0, 'y': 2.0}` (a dict)      |
| LineStringXY   | a Polars Series of `struct{x, y}`      |
| PolygonXY      | a Series of `list[struct{x, y}]`       |
| MultiPolygonXY | a Series of `list[list[struct{x, y}]]` |

Points arrive as plain Python, but everything else arrives as a Polars Series.
If you want to work with rings and coordinates, you'll have to call `.to_list()`.
After that, editing works, and returning nested lists of dicts rebuilds the geometry:

```python
pl.col("g").map_elements(
    lambda v: [[{**p, "x": p["x"] * 2} for p in ring] for ring in v.to_list()],
    return_dtype=df.schema["g"],
)
```

Two things in this snippet are easy to get wrong:

- `return_dtype` must be an instance. Passing the class like `return_dtype=PolygonXY`, raises a TypeError.
- The CRS can be lost. `PolygonXY()` produces a polygon column with no CRS. An existing CRS can be retained by using `df.schema["g"]`.

TODO: add helper function to set the CRS of the original column and describe it here.

TODO: setting for accepting/refusing trivial cases (e.g. 'remove duplicate points' for single point)?
or an envvar? or not at all?
