"""Averaging over coordinates.
Note that this is different from an area centroid!
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from geopolars.datatypes import GeoPoint
from geopolars.geometry._dispatch import by_geometry

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _opened(rings: pl.Expr) -> pl.Expr:
    """drops the final coordinate from a ring.
    This is needed because a ring repeats the final coordinate.
    """
    # `list.len()` is unsigned, so a ring with nothing in it would wrap around.
    length = (rings.list.len().cast(pl.Int64) - 1).clip(lower_bound=0)
    return rings.list.slice(0, length)


def _flattened(parts: pl.Expr, layers: int, rings: bool) -> pl.Expr:
    """The coordinates under `layers` `List` layers of parts, all in one list."""
    if layers == 0:
        return parts
    if layers == 1 and rings:
        parts = _opened(parts)
    return _flattened(parts.explode(empty_as_null=False), layers - 1, rings)


def _mean(geometry: type[GeoArrowType], column: pl.Expr) -> pl.Expr:
    """The mean of one geometry's coordinates, per row, as a point."""
    storage = column.ext.storage()
    # Wrap coordinates in a list as a unifying step
    # With this we can use the same function for everything that has coordinates.
    vertices = storage.repeat_by(1) if geometry._nesting == 0 else storage

    coords = _flattened(pl.element(), max(geometry._nesting - 1, 0), geometry._rings)
    # `m` is also averaged and included here, though not actually an axis.
    axes = coords.struct.field("*")

    # If there is a single null, then the centroid is null
    # https://geoarrow.org/format.html#missing-values-null
    complete = (coords.len() > 0) & pl.all_horizontal(axes.is_not_null()).all()

    # One list layer is left, holding the single coordinate just computed.
    centroid = vertices.list.eval(pl.when(complete).then(pl.struct(axes.mean())))
    return centroid.list.first().ext.to(GeoPoint.of_dimension(geometry._dimension)())


def coordinate_centroid(geometry: IntoExprColumn) -> pl.Expr:
    """The mean of the coordinates a geometry is made of, as a `geoarrow.point`.

    Note that this is *not* the centre of mass (area centroid).
    Every coordinate counts once and counts the same,
    so this is the centre of a geometry's vertices rather than of the space it covers.
    For a polygon those differ:
    this one moves with where the vertices are dense.

    | in                     | out          |
    |------------------------|--------------|
    | `PointXY`              | `PointXY`    |
    | `LineStringXYZ`        | `PointXYZ`   |
    | `PolygonXYZM`          | `PointXYZM`  |

    The measure `m` is also averaged.
    A geometry with no coordinates at all or an invalid coordinate has no centroid.

    ```python
    df.select(geometry.coordinate_centroid("route"))
    ```
    """
    return by_geometry(geometry, _mean)
