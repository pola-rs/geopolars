"""Building points and multipoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from geopolars._docs import doc
from geopolars.datatypes import PointType
from geopolars.geo.construct._common import (
    _coord,
    _from_columns,
    _gather,
    _given,
    _metadata,
)

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn


@doc()
def point(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.point` column from its coordinate columns.
    Whether you pass `z` and/or `m` decides the dimension, and so the dtype:

    | passed      | dtype       |
    |-------------|-------------|
    | -           | `PointXY`   |
    | `z`         | `PointXYZ`  |
    | `m`         | `PointXYM`  |
    | `z` and `m` | `PointXYZM` |

    `z` is an elevation.
    `m` is an arbitrary measure carried along with the vertex,
    such as a timestamp or a distance along a route.

    ```python
    df.select(geo.point("lon", "lat", crs="EPSG:4326"))
    ```

    {null_inside_row}

    {constructed_geoarrow_valid}
    """
    given = _given(x, y, z, m)
    dimension = tuple(given)
    coords = [_coord(value, name) for name, value in given.items()]

    # No plugin call is needed:
    # the coordinates are never copied or crossed over FFI.
    #
    # Null coordinate must null the entire point.
    # https://geoarrow.org/format.html#missing-values-null
    # TODO: The spec can be interpreted as saying that 'm' may also not be null.
    # That really limits usability, no?
    complete = pl.all_horizontal([coord.is_not_null() for coord in coords])
    dtype = PointType.of_dimension(dimension)
    return pl.when(complete).then(pl.struct(coords)).ext.to(dtype(crs=crs))


@doc()
def multi_point_from_points(
    points: IntoExprColumn, *, crs: str | None = None
) -> pl.Expr:
    """Build a `geoarrow.multipoint` column out of lists of points.

    ```python
    df.group_by("survey").agg(
        geo.multi_point_from_points(
            geo.point("lon", "lat").implode()
        ).alias("sightings")
    )
    ```

    {null_inside_row}

    {constructed_geoarrow_valid}
    """
    return _gather("multipoint", points, _metadata(crs))


@doc()
def multi_point_from_columns(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multipoint` column from one coordinate column per axis.
    Each is a `List[Float64]` holding one list of coordinates per multipoint.

    ```python
    df.select(geo.multi_point_from_columns("lon", "lat", m="seen_at"))
    ```

    {null_inside_row}

    {constructed_geoarrow_valid}
    """
    return _from_columns("multipoint_coords", x, y, z, m, _metadata(crs))


def multi_point(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multipoint` column, from points or from coordinates.
    Dispatches to either:
    - [`multi_point_from_points`][geopolars.geo.multi_point_from_points]
    - [`multi_point_from_columns`][geopolars.geo.multi_point_from_columns]
    """
    if y is None:
        if z is not None or m is not None:
            msg = (
                "multi_point() got a z or m coordinate without a y coordinate; "
                "pass x and y as columns, or call multi_point_from_points()"
            )
            raise TypeError(msg)
        return multi_point_from_points(x, crs=crs)

    return multi_point_from_columns(x, y, z, m, crs=crs)
