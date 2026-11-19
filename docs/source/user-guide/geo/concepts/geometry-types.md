# Geometry types

GeoPolars stores geometries in the [GeoArrow](https://geoarrow.org/) format.
This is an extended standard to store geometries in Apache Arrow (the memory format that Polars uses).
Each kind of geometry has its own Polars data type,
just like integers and strings do.

## The six families of geometry

| Kind | What it represents | Data type |
| - | - | - |
| Point | a single position | `PointXY[ZM]` |
| LineString | a line through two or more vertices | `LineStringXY[ZM]` |
| Polygon | an area or surface, with an exterior ring and any number of holes | `PolygonXY[ZM]` |
| MultiPoint | a list of points | `MultiPointXY[ZM]` |
| MultiLineString | a semantically related group of linestrings | `MultiLineStringXY[ZM]` |
| MultiPolygon | a semantically related group of polygons | `MultiPolygonXY[ZM]` |

You will see that these names are also shown in the header of a dataframe when you print it,
together with the [CRS](crs.md) if the column has one:

{{code_block('user-guide/geo/concepts/geometry-types','df',['DataFrame','geo.from_wkt','PolygonXY'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/geometry-types"
--8<-- "python/user-guide/geo/concepts/geometry-types.py:df"
```

Under the hood, a geometry is plain Polars data.
A point is a struct of its coordinates, such as `{x, y}`.
A linestring is a list of those structs, and a polygon a list of rings,
each of which is a list of structs.
The multi kinds add one more list around their single kind.
(See [user-defined functions](../udfs.md) for where you might run into this.)

## One kind per column

A Polars column has a single data type, so every row in a geometry column is the same kind.
A column cannot hold a point in one row and a polygon in the next.

!!! note "No geometry collections"

```text
GeoPolars has no data type for a mix of kinds, such as a GeometryCollection, yet.
An operation whose result can fall apart into several pieces,
such as `intersection` or `difference`,
can give the multi kind instead (see `allow_splitting`).
```

This is an important limitation to be aware of when converting from and to other formats.
When you decode geometries, for example from WKT,
you need to say which data type you want.
One exception to this rule are multis:
A single geometry always fits into its multi kind,
so a mix of polygons and multipolygons can be decoded as `MultiPolygonXY`:

{{code_block('user-guide/geo/concepts/geometry-types','promote',['MultiPolygonXY','geo.to_wkt'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/geometry-types"
--8<-- "python/user-guide/geo/concepts/geometry-types.py:promote"
```

The same holds for the other kinds.
An empty single geometry becomes an empty multi geometry,
so `POINT EMPTY` decodes into `MULTIPOINT EMPTY`.

Anything else that does not fit is an error:

{{code_block('user-guide/geo/concepts/geometry-types','mismatch',[])}}

```python exec="on" result="text" session="user-guide/geo/concepts/geometry-types"
--8<-- "python/user-guide/geo/concepts/geometry-types.py:mismatch"
```

Because the kind is part of the data type, GeoPolars knows it before your query runs.
An operation that does not support a kind, such as `area` on points,
is refused while the query is built, rather than halfway through running it.

## Dimensions

The *kind* can say that something is a point, but that does not yet say anything about its *dimension*.
Every coordinate has an x and a y.
It can also have a z, an m, or both:

- **z** is the elevation, a third dimension of the position.
- **m** is a *measure*: any number you want to attach to a vertex,
  such as a timestamp, or the distance along a route.
  It is not really a part of the position,
  but also considered its own dimension as it is an extra value added to the point.

Each kind comes in four data types, one for each set of dimensions:
`PolygonXY`, `PolygonXYZ`, `PolygonXYM` and `PolygonXYZM`.
Though they support almost all the same operations,
a three-dimensional point is still a different data type from a two-dimensional one.
That way you cannot accidentally mix them up!
Every coordinate in a column has the same dimensions,
so decoding a 3D geometry into a 2D column is an error, too.

`x`, `y`, `z` and `m` read a coordinate out of a geometry.
`force_2d` and `without_m` drop the z and m,
while `force_3d` and `force_m` add them:

{{code_block('user-guide/geo/concepts/geometry-types','dimensions',['LineStringXYZ','geo.z','geo.force_2d'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/geometry-types"
--8<-- "python/user-guide/geo/concepts/geometry-types.py:dimensions"
```

As we use GEOS for many of the underlying operations,
there are some operations that do not support z and m values.
The API reference says, per operation, what it does with z and m.

## Boxes

A **box** is an axis-aligned rectangle.
This data type is mostly used to describe *bounds* as a quick way to filter over geometries, with
`xmin`, `ymin`, `xmax` and `ymax` (and the same for z and m, if it has them).
`bounds` gives one per geometry, and `to_polygon` turns it back into a polygon:

{{code_block('user-guide/geo/concepts/geometry-types','box',['geo.bounds','geo.to_polygon'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/geometry-types"
--8<-- "python/user-guide/geo/concepts/geometry-types.py:box"
```

A box with `xmin > xmax` [crosses the antimeridian](crs.md#crossing-the-antimeridian).

## WKT and WKB

Besides the native types above, GeoPolars has two *encoded* types.

- `Wkt`, for geometries written as well-known text (WKT), such as `POINT (1 2)`.
- `Wkb`, for geometries written as well-known binary (WKB).

Converting to and from this encoding is done using `to_wkt` and `to_wkb`.
These operations keep the CRS of their input:

{{code_block('user-guide/geo/concepts/geometry-types','encoded',['geo.to_wkt','geo.to_wkb'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/geometry-types"
--8<-- "python/user-guide/geo/concepts/geometry-types.py:encoded"
```

The encoded types are used for exchange, and not supported by most operations directly.
Consider them useful for reading and writing only.
To run an operation on them, decode them first, with `from_wkt` or `from_wkb`
(see [reading and writing](../io.md#wkt-and-wkb)).
