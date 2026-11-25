# Coming from GeoPandas

If you know [GeoPandas](https://geopandas.org/), most of what you know carries over.
The operations have similar names, and many of them run on the same GEOS library.
What differs is how you call them, and how they are stored under the hood.
This page goes over the differences you are most likely to run into.

## Geometries are columns, not a GeoSeries

In GeoPandas, a `GeoDataFrame` has one *active* geometry column,
and operations are methods on it, or on a `GeoSeries`.
In GeoPolars, a geometry column is a column like any other.
A DataFrame can have as many as you like,
and you name the one you want in each expression:

```python
# GeoPandas
gdf["area"] = gdf.area

# GeoPolars
df = df.with_columns(area=geo.area("geometry"))
```

As these are ordinary Polars expressions,
they also work in lazy queries, in `group_by`, and on the streaming engine.

## One kind of geometry per column

A GeoPandas column can hold a point in one row and a polygon in the next.
A GeoPolars column has a single [geometry type](../concepts/geometry-types.md),
such as `PolygonXY`, so every row is the same kind, with the same dimensions.
Columns that mix other kinds, such as points and linestrings, are not supported yet.
One exception is that of multis and their single variants;
polygons and multipolygons can go together by widening the Polygon into a `MultiPolygonXY` column.

## The CRS is part of the data type

In GeoPandas, you find the CRS with `gdf.crs`.
In GeoPolars, it is part of the data type of the column, so you find it in the schema:

```python
df.schema["geometry"]
```

GeoPolars compares CRSs by [how they are written](../concepts/crs.md#combining-columns).
GeoPandas writes them as PROJJSON,
so data from GeoPandas does not combine with a column you declared as `"EPSG:4326"`,
even though both are the same CRS.
Relabel one of them with `geo.set_crs(..., force=True)`.

## Longitude and latitude are measured in metres

In GeoPandas, the area of a polygon in longitude and latitude comes out in square degrees,
with a warning that it is probably wrong.
Measuring along the ellipsoid is a different operation entirely.
GeoPolars exposes both of these the same way, but makes the metric explicit.
This means you do not have to manually reproject first.
You specifically choose to measure the planar distance, or along the ellipsoid (projected).
`area`, `length`, `distance`, and other such expressions' output units depends on this setting.
See [geographic and projected](../concepts/crs.md#geographic-and-projected) for how GeoPolars decides how to measure.

## Nulls and empty geometries

GeoPandas has `None` for a missing geometry, and empty geometries such as `POLYGON EMPTY`.
GeoPolars has the same two, as nulls and empty geometries.
One difference is that GeoPolars only allows a *whole* geometry to be null.
For example, a point built from a null x is a null point.
See [nulls, NaN and empty geometries](../concepts/nulls-nan-and-empty.md).

## Converting between GeoPandas and GeoPolars

Both speak [GeoArrow](https://geoarrow.org/), so you can hand a whole table over at once.
GeoArrow can store coordinates *interleaved* (`x, y, x, y, …` in one array),
or *separated* (one array per axis).
GeoPolars only reads separated coordinates, so pass `interleaved=False`:

```python
import geopandas as gpd
import polars as pl

# GeoPandas to GeoPolars
df = pl.DataFrame(gdf.to_arrow(geometry_encoding="geoarrow", interleaved=False))

# GeoPolars to GeoPandas
gdf = gpd.GeoDataFrame.from_arrow(df)
```

The CRS comes along in both directions.
You can also convert through WKB, which is what GeoPandas gives by default.
Then you decode the `Wkb` column yourself with `geo.from_wkb`,
which lets you choose the type to decode into:

```python
from geopolars import MultiPolygonXY, geo

df = pl.DataFrame(gdf.to_arrow()).with_columns(
    geometry=geo.from_wkb("geometry", MultiPolygonXY)
)
```

Finally, both read and write [GeoParquet](../io.md#geoparquet).

## Operations Dictionary

Most GeoPandas operations have a GeoPolars counterpart in the `geo` module.
The table below lists the most common ones.
Each also works as a method in the `geo` namespace,
so `geo.area("geometry")` is the same as `gpl.col("geometry").geo.area()`.

| GeoPandas | GeoPolars |
| - | - |
| `gpd.read_parquet(path)` | `pl.scan_parquet(path)`, with [GeoParquet reading](../io.md#reading) switched on |
| `gdf.to_parquet(path)` | `df.write_parquet(path)`, with [GeoParquet writing](../io.md#writing) switched on |
| `gpd.points_from_xy(x, y, crs=...)` | `geo.point("x", "y", crs=...)` |
| `GeoSeries.from_wkt(...)`, `from_wkb(...)` | `geo.from_wkt(..., dtype)`, `geo.from_wkb(..., dtype)` |
| `.to_wkt()`, `.to_wkb()` | `geo.to_wkt`, `geo.to_wkb` |
| `.set_crs(...)`, `.to_crs(...)` | `geo.set_crs`, `geo.to_crs` |
| `.area`, `.length` | `geo.area`, `geo.length` |
| `.distance(other)` | `geo.distance` for points, `geo.length(geo.shortest_line_between(a, b))` for other kinds |
| `.bounds`, `.total_bounds` | `geo.bounds`, `geo.total_bounds` |
| `.centroid` | `geo.centroid` |
| `.x`, `.y`, `.z`, `.get_coordinates()` | `geo.x`, `geo.y`, `geo.z` |
| `.count_coordinates()` | `geo.count_coordinates` |
| `.exterior`, `.interiors`, `.boundary` | `geo.exterior`, `geo.interior`, `geo.boundary` |
| `.explode()` | `geo.parts`, followed by Polars' `explode` |
| `.is_valid` | `geo.is_valid_ogc` (see [validity](../concepts/validity.md)) |
| `.is_empty`, `.is_simple`, `.is_ring`, `.is_ccw` | `geo.is_empty`, `geo.is_simple`, `geo.is_ring`, `geo.is_ccw` |
| `.geom_equals_exact(...)`, `.geom_equals(...)` | `geo.is_equal`, `geo.is_topologically_equal` |
| `.relate(other)` | `geo.relate` |
| `.orient_polygons()` | `geo.to_ccw`, `geo.to_cw` (see [winding order](../concepts/winding-order.md)) |
| `.reverse()` | `geo.reverse` |
| `.remove_repeated_points()` | `geo.remove_repeats_in_succession` |
| `.force_2d()`, `.force_3d()` | `geo.force_2d`, `geo.force_3d` |
| `.buffer(...)`, `.offset_curve(...)` | `geo.buffer`, `geo.offset_curve` |
| `.simplify(...)` | `geo.simplify` |
| `.convex_hull` | `geo.convex_hull` |
| `.intersection(other)`, `.difference(other)` | `geo.intersection`, `geo.difference` |
| `.shortest_line(other)` | `geo.shortest_line_between` |
| `.interpolate(distance)` | `geo.line_interpolate_point` |
| `.affine_transform(...)` | `geo.affine_transform`, which takes the same matrix |
| `.translate`, `.rotate`, `.scale`, `.skew` | `geo.translate`, `geo.rotate`, `geo.scale`, `geo.skew` |
| `.apply(...)` | Polars' `map_elements` (see [user-defined functions](../udfs.md)) |

## Not available yet

Some things you might reach for in GeoPandas are not in GeoPolars yet:

- Spatial joins (`sjoin`, `sjoin_nearest`) and a spatial index (`sindex`).
- Predicates such as `intersects`, `contains` and `within`.
  For now, `geo.relate` gives the [DE-9IM](https://en.wikipedia.org/wiki/DE-9IM) matrix that each of these is defined by:
  a short code that describes how two geometries meet.
- `union_all`, `dissolve` and `make_valid`.
- Plotting. Convert to GeoPandas, and plot from there.
- Reading GeoJSON, shapefiles and other GDAL formats.
  Read them with GeoPandas, and convert.
