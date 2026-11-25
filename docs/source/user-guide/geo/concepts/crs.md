# Coordinate reference systems

Coordinates on their own are just numbers.
A **coordinate reference system (CRS)** says what they actually represent.
They could be degrees of longitude and latitude,
distance in metres away from some datum point on a national grid,
or feet on a state plane.

In GeoPolars, every geometry *column* carries its CRS in its data type.
This means that the CRS is the same for every row in that column.
This makes it possible for GeoPolars to check it,
and pick the right way to measure before your query runs.

## Declaring a CRS

Most constructors and decoders take a `crs`.
It can be anything [PROJ](https://proj.org/) accepts:
an `AUTH:CODE` such as `"EPSG:4326"`, WKT, or PROJJSON.
The CRS shows up in the data type of the column:

{{code_block('user-guide/geo/concepts/crs','declare',['geo.point'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/crs"
--8<-- "python/user-guide/geo/concepts/crs.py:declare"
```

A column that has no CRS yet can get one with `set_crs`:

{{code_block('user-guide/geo/concepts/crs','set_crs',['geo.set_crs'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/crs"
--8<-- "python/user-guide/geo/concepts/crs.py:set_crs"
```

Setting a CRS like this only applies a label.
It trusts that the data you give it is correct, and does not check or move any coordinate.
To this end, `set_crs` refuses a column that already declares a CRS,
as overwriting it would silently imply moving the geometry.
Use `force=True` if the label really is wrong.

So, what do you do if you *do* actually want to represent a location in another CRS?
That's where *reprojecting* comes in.

## Reprojecting

`to_crs` moves every coordinate from the column's CRS into another one, using PROJ.
The result declares the new CRS:

{{code_block('user-guide/geo/concepts/crs','to_crs',['geo.to_crs'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/crs"
--8<-- "python/user-guide/geo/concepts/crs.py:to_crs"
```

## Geographic and projected

A CRS is either *geographic* or *projected*.

- A **geographic** CRS, such as EPSG:4326, has longitude and latitude in degrees.
  Such CRSes can be identified with `is_geographic`.
- A **projected** CRS, such as EPSG:28992, lies flat on a plane, usually in metres.

The difference matters when you measure distances between two points.
Because the earth is (approximately) an ellipsoid,
a degree east is generally not as long as a degree north.
How far away that is depends on your location!
Because of this, measuring degrees as if they lie on a plane gives numbers that mean very little.
A geometry in a geographic CRS is therefore measured *geodesically* by default,
along the ellipsoid of its CRS, which gives metres.
A geometry in a projected CRS, or without one, is measured on the plane,
in whatever unit its coordinates have.
That is close for a CRS made for measuring, such as a UTM zone or a national grid,
but not for Web Mercator (EPSG:3857), which stretches distances away from the equator.
It's probably better to pass `distance_method="geodesic"` there!
`area` decides differently.
It measures on the plane only without a CRS, or in an equal-area CRS, whose plane keeps areas.
In every other CRS, it measures along the ellipsoid.

For example, these are the *planar* (in degrees) and *geodesic* (in metres, on the ellipsoid) distances
between Amsterdam and Utrecht:

{{code_block('user-guide/geo/concepts/crs','measure',['geo.distance'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/crs"
--8<-- "python/user-guide/geo/concepts/crs.py:measure"
```

Amsterdam and Utrecht are 34 kilometres apart, or 0.35 degrees planar.

Doing the same for these two cities that are very far north shows that this ratio changes.
TODO: Give an example comparing the distance in degrees and km somewhere very far up north.

In some cases, the planar area is still useful.
Measuring on the plane is also much cheaper than measuring along the ellipsoid.
That is why operations like `area`, `length`,
and `distance` allow you to choose which distance metric to use.

## Combining columns

An operation on two geometry columns, such as `distance` or `intersection`,
needs both to declare the same CRS, or neither to declare one.
Otherwise it is refused while the plan is built, before any data is read:

{{code_block('user-guide/geo/concepts/crs','mismatch',['geo.distance'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/crs"
--8<-- "python/user-guide/geo/concepts/crs.py:mismatch"
```

Reproject one of them with `to_crs` first.

!!! warning "CRSs are compared by how they are written"

    `"EPSG:4326"` and `"epsg:4326"` are the same CRS to PROJ,
    but GeoPolars compares the label as written, so they count as different.
    Write every CRS the same way, for example always as `"EPSG:<code>"`.

## CRS of expressions

Operations that return geometries keep the CRS of their input.
Results that are plain numbers, such as `area`, or `x` and `y`, have no CRS.
Their units might still be implied though!

Using [user-defined functions](../udfs.md), you must be careful to persist or adapt the CRS as required!

## Crossing the antimeridian

The antimeridian is where longitude jumps from 180° to -180°.
A geometry that crosses it can be written in two ways:

- with longitudes past the edge, such as a line from 178° to 182°, or
- with longitudes in range, such as a line from 178° to -178°.
  Many tools read this as a line that goes the long way round the globe.

`wrap_longitude` brings longitudes that went past the edge back into range.
It needs a geographic CRS.
Each vertex moves on its own, so a geometry that crosses the edge gets torn apart.
With `skip_crossing=True`, such a geometry is left as it is,
and only geometries that lie entirely past the edge are moved:

{{code_block('user-guide/geo/concepts/crs','antimeridian',['geo.wrap_longitude'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/crs"
--8<-- "python/user-guide/geo/concepts/crs.py:antimeridian"
```

!!! tip "Data from 0° to 360°"

    Some data, such as many climate and ocean grids, runs from 0° to 360° instead.
    EPSG:4326 covers both conventions, so the CRS cannot tell you which one data uses.
    Pass `start=0` to wrap into `[0, 360]` instead.

A bounding box that crosses the antimeridian has `xmin > xmax`.
`bounds` gives such a box for geometries that go past the edge:

{{code_block('user-guide/geo/concepts/crs','bounds',['geo.bounds'])}}

```python exec="on" result="text" session="user-guide/geo/concepts/crs"
--8<-- "python/user-guide/geo/concepts/crs.py:bounds"
```

## Coordinates outside the CRS

A CRS also has bounds: a latitude has to lie between -90° and 90°.
PROJ cannot transform a place that does not exist.

`is_valid_geoarrow` and `is_valid_ogc` check the bounds when you pass `out_of_bounds`
(see [validity](validity.md)).
`fix_out_of_bounds` drops or clamps the coordinates that lie outside them.
`area` raises on them by default when it measures geodesic (see `on_out_of_bounds`).

## GeoParquet

How a GeoParquet file stores the CRS of each geometry column,
and what that means when you write or read one,
is described under [CRS](../io.md#crs) on the reading and writing page.
