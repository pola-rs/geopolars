# Validity

Every operation expects a *valid* geometry as its input.
If given one, they should (generally) not fail, and give sensible results.

But what does *valid* mean? GeoPolars knows *two* levels, which nest.

The first is validity according to the GeoArrow spec, checked by `geo.is_valid_geoarrow`.

- Only a whole geometry can be null.
  A part, a ring, or a coordinate nested in another geometry cannot be null.
- Every *ring* is closed. Its last vertex is the same as its first.

Checking GeoArrow validity is cheap. It looks at every coordinate once.
It can also check, in that same pass, that every coordinate is finite,
or that it lies [inside the bounds of its CRS](crs.md#coordinates-outside-the-crs) (`out_of_bounds`).

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

!!! note

    OGC allows some geometries that might still surprise you!
    An empty geometry is valid by OGC (find them with `geo.is_empty`),
    and so is the same vertex twice in a row (`geo.has_repeats_in_succession`).

Below, we check some parcels against both levels.
The bowtie crosses itself, and the sliver goes back and forth, so neither is valid by OGC.
The empty polygon is valid by both.

{{code_block('user-guide/geo/concepts/validity','df',['DataFrame','geo.from_wkt','PolygonXY'])}}

{{code_block('user-guide/geo/concepts/validity','checks',['geo.is_valid_geoarrow','geo.is_valid_ogc','geo.is_empty','geo.has_repeats_in_succession'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/validity"
--8<-- "python/user-guide/geo/concepts/validity.py:df"
--8<-- "python/user-guide/geo/concepts/validity.py:checks"
```

`geo.null_out_invalid_geoarrow` and `geo.null_out_invalid_ogc` replace invalid geometries with nulls.
See `is_valid_geoarrow` and `is_valid_ogc` in the API reference for all the rules.

{{code_block('user-guide/geo/concepts/validity','null_out',['geo.null_out_invalid_ogc','geo.to_wkt'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/validity"
--8<-- "python/user-guide/geo/concepts/validity.py:null_out"
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
