"""Building a geometry out of the parts it is made of."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import GeoPoint

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn


def _coord(value: IntoExprColumn, name: str) -> pl.Expr:
    """One coordinate column of a geometry's storage struct."""
    return _named(value, name).cast(pl.Float64)


def _named(value: IntoExprColumn, name: str) -> pl.Expr:
    """One argument, under the name of the coordinate it carries.

    The plugin reads the dimension back off these names, so they are the spec's
    (`x`, `y`, `z`, `m`) rather than whatever the column was called.
    """
    if isinstance(value, str):
        expr = pl.col(value)
    elif isinstance(value, pl.Series):
        expr = pl.lit(value)
    else:
        expr = value
    return expr.alias(name)


def _given(
    x: IntoExprColumn,
    y: IntoExprColumn | None,
    z: IntoExprColumn | None,
    m: IntoExprColumn | None,
) -> dict[str, IntoExprColumn]:
    """The coordinates that were passed, in the field order the spec fixes."""
    given = {"x": x, "y": y, "z": z, "m": m}
    return {name: value for name, value in given.items() if value is not None}


def _from_columns(
    function_name: str,
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None,
    m: IntoExprColumn | None,
) -> pl.Expr:
    """Zip one nested coordinate column per axis into one geometry per row."""
    # A plugin call rather than expressions: Polars can transpose a struct of
    # lists into a list of structs only by exploding and regrouping, where the
    # plugin reuses the offsets the columns already have.
    return register_plugin_function(
        plugin_path=LIB,
        args=[_named(value, name) for name, value in _given(x, y, z, m).items()],
        function_name=function_name,
        is_elementwise=True,
    )


def _gather(function_name: str, parts: IntoExprColumn) -> pl.Expr:
    """Gather a list column of a geometry's parts into one geometry per row."""
    # Unlike `point`, this is a plugin call: the output dimension is only
    # knowable from the input's dtype, which `output_type_func` gets to see and
    # Python does not.
    return register_plugin_function(
        plugin_path=LIB,
        args=[parts],
        function_name=function_name,
        is_elementwise=True,
    )


def validate(geometry: IntoExprColumn) -> pl.Expr:
    """Null out every geometry that is missing a part, or a coordinate of one.
    GeoArrow allows nulls only at the outermost level:
    A geometry is whole or it is null.
    <https://geoarrow.org/format.html#missing-values-null>

    Every constructor in this module already holds to that,
    so a geometry built by one never needs this.
    This function is for the routes into a geometry column that go
    around them and check nothing:

    ```python
    pl.scan_parquet("routes.parquet").select(geometry.validate("route"))
    ```

    `ext.to` is a relabelling and reads no data, so it cannot check either:

    ```python
    df.select(pl.col("route").ext.to(gpl.LineStringXY())).select(
        geometry.validate("route")
    )
    ```

    Operations over coordinates take the guarantee as given rather than paying
    for it on every call, so a column that arrived one of those ways is worth
    putting through this once. On a column that is already whole it finds
    nothing and copies nothing.
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[geometry],
        function_name="validate",
        is_elementwise=True,
    )


def point(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
) -> pl.Expr:
    """Build a `geoarrow.point` column from its coordinate columns.
    Whether you pass `z` and/or `m` decides the dimension, and so the dtype:

    | passed      | dtype       |
    |-------------|-------------|
    | -           | `PointXY`   |
    | `z`         | `PointXYZ`  |
    | `m`         | `PointXYM`  |
    | `z` and `m` | `PointXYZM` |

    `z` is an elevation; `m` is an arbitrary measure carried along with the
    vertex, such as a timestamp or a distance along a route.
    """
    given = _given(x, y, z, m)
    dimension = tuple(given)
    coords = [_coord(value, name) for name, value in given.items()]

    # No plugin call is needed:
    # the coordinates are never copied or crossed over FFI.
    #
    # Missing coordinate must invalidate the entire point.
    # https://geoarrow.org/format.html#missing-values-null
    # TODO: The spec can be interpreted as saying that 'm' may also not be null.
    # That really limits usability, no?
    complete = pl.all_horizontal([coord.is_not_null() for coord in coords])
    dtype = GeoPoint.of_dimension(dimension)
    return pl.when(complete).then(pl.struct(coords)).ext.to(dtype())


def linestring_from_vertices(vertices: IntoExprColumn) -> pl.Expr:
    """Build a `geoarrow.linestring` column out of lists of vertices.

    `vertices` is a list column holding one list per linestring, of either
    `geoarrow.point`s or the bare coordinate structs a point wraps.

    Vertices normally arrive grouped:

    ```python
    df.group_by("route").agg(
        geometry.linestring_from_vertices(
            geometry.point("lon", "lat").implode()
        ).alias("route")
    )
    ```
    """
    return _gather("linestring", vertices)


def linestring_from_columns(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
) -> pl.Expr:
    """Build a `geoarrow.linestring` column from one coordinate column per axis.
    Each is a `List[Float64]` holding one list of coordinates per linestring

    ```python
    df.select(geometry.linestring_from_columns("lon", "lat", z="elevation"))
    ```
    """
    return _from_columns("linestring_coords", x, y, z, m)


def linestring(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
) -> pl.Expr:
    """Build a `geoarrow.linestring` column, from vertices or from coordinates.
    Dispatches to either:
    - [`linestring_from_vertices`][geopolars.geometry.linestring_from_vertices]
    - [`linestring_from_columns`][geopolars.geometry.linestring_from_columns]
    """
    if y is None:
        if z is not None or m is not None:
            msg = (
                "linestring() got a z or m coordinate without a y coordinate; "
                "pass x and y as columns, or call linestring_from_vertices()"
            )
            raise TypeError(msg)
        return linestring_from_vertices(x)

    return linestring_from_columns(x, y, z, m)


def multipoint_from_points(points: IntoExprColumn) -> pl.Expr:
    """Build a `geoarrow.multipoint` column out of lists of points.

    ```python
    df.group_by("survey").agg(
        geometry.multipoint_from_points(
            geometry.point("lon", "lat").implode()
        ).alias("sightings")
    )
    ```
    """
    return _gather("multipoint", points)


def multipoint_from_columns(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multipoint` column from one coordinate column per axis.
    Each is a `List[Float64]` holding one list of coordinates per multipoint.

    ```python
    df.select(geometry.multipoint_from_columns("lon", "lat", m="seen_at"))
    ```
    """
    return _from_columns("multipoint_coords", x, y, z, m)


def multipoint(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multipoint` column, from points or from coordinates.
    Dispatches to either:
    - [`multipoint_from_points`][geopolars.geometry.multipoint_from_points]
    - [`multipoint_from_columns`][geopolars.geometry.multipoint_from_columns]
    """
    if y is None:
        if z is not None or m is not None:
            msg = (
                "multipoint() got a z or m coordinate without a y coordinate; "
                "pass x and y as columns, or call multipoint_from_points()"
            )
            raise TypeError(msg)
        return multipoint_from_points(x)

    return multipoint_from_columns(x, y, z, m)


def multilinestring_from_linestrings(linestrings: IntoExprColumn) -> pl.Expr:
    """Build a `geoarrow.multilinestring` column out of lists of linestrings.
    Takes either a list of linestrings, or bare vertex lists.

    ```python
    (
        df.group_by("river", "branch", maintain_order=True)
        .agg(
            geometry.linestring_from_vertices(
                geometry.point("lon", "lat").implode()
            ).alias("branch")
        )
        .group_by("river", maintain_order=True)
        .agg(
            geometry.multilinestring_from_linestrings(
                pl.col("branch").implode()
            ).alias("river")
        )
    )
    ```
    """
    return _gather("multilinestring", linestrings)


def multilinestring_from_columns(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multilinestring` column from one column per axis.
    Each is a `List[List[Float64]]`.

    ```python
    df.select(geometry.multilinestring_from_columns("lon", "lat"))
    ```
    """
    return _from_columns("multilinestring_coords", x, y, z, m)


def multilinestring(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multilinestring` column, from linestrings or coordinates.
    Dispatches to either:
    - [`multilinestring_from_linestrings`][geopolars.geometry.multilinestring_from_linestrings]
    - [`multilinestring_from_columns`][geopolars.geometry.multilinestring_from_columns]
    """
    if y is None:
        if z is not None or m is not None:
            msg = (
                "multilinestring() got a z or m coordinate without a y "
                "coordinate; pass x and y as columns, or call "
                "multilinestring_from_linestrings()"
            )
            raise TypeError(msg)
        return multilinestring_from_linestrings(x)

    return multilinestring_from_columns(x, y, z, m)


def polygon_from_rings(rings: IntoExprColumn) -> pl.Expr:
    """Build a `geoarrow.polygon` column out of lists of rings.

    `rings` is a list column holding one list per polygon, of either
    `geoarrow.linestring`s or the bare vertex lists a linestring wraps. The
    first ring of a polygon is its exterior ring; the rest are its holes.

    A ring is a closed linestring: its last vertex has to repeat its first.
    That is the caller's to get right -- this does not check it, and does not
    close a ring for you.

    Rings normally arrive grouped, one linestring at a time:

    ```python
    (
        df.group_by("plot", "ring", maintain_order=True)
        .agg(
            geometry.linestring_from_vertices(
                geometry.point("lon", "lat").implode()
            ).alias("ring")
        )
        .group_by("plot", maintain_order=True)
        .agg(geometry.polygon_from_rings(pl.col("ring").implode()).alias("plot"))
    )
    ```
    """
    return _gather("polygon", rings)


def polygon_from_columns(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
) -> pl.Expr:
    """Build a `geoarrow.polygon` column from one coordinate column per axis.
    Each is a `List[List[Float64]]`.

    ```python
    df.select(geometry.polygon_from_columns("lon", "lat"))
    ```

    The first ring of a polygon is its exterior ring,
    and the rest are its holes.
    """
    return _from_columns("polygon_coords", x, y, z, m)


def polygon(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
) -> pl.Expr:
    """Build a `geoarrow.polygon` column, from rings or from coordinates.
    Dispatches to either:
    - [`polygon_from_rings`][geopolars.geometry.polygon_from_rings]
    - [`polygon_from_columns`][geopolars.geometry.polygon_from_columns]
    """
    if y is None:
        if z is not None or m is not None:
            msg = (
                "polygon() got a z or m coordinate without a y coordinate; "
                "pass x and y as columns, or call polygon_from_rings()"
            )
            raise TypeError(msg)
        return polygon_from_rings(x)

    return polygon_from_columns(x, y, z, m)
