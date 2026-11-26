"""The mean coordinate of a geometry: every coordinate weighted the same.
Note that this is not a centroid, which weighs by length or area!
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import polars as pl

from geopolars.datatypes import PointType
from geopolars.geo._dispatch import on_geometry
from geopolars.geo.coordinates import _axis

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

# A reduction over the coordinates of one innermost part of a geometry.
Reduce = Callable[[pl.Expr, bool], pl.Expr]


def _total(part: pl.Expr, rings: bool) -> pl.Expr:
    """The sum of one axis over one part's coordinates."""
    if not rings:
        return part.list.sum()
    # A ring's last coordinate repeats its first, so take it back out.
    # An empty ring has no last coordinate, and no sum to correct.
    return part.list.sum() - part.list.last().fill_null(0.0)


def _count(part: pl.Expr, rings: bool) -> pl.Expr:
    """How many coordinates one part holds."""
    if not rings:
        return part.list.len()
    # ...minus the one that closes it, where there is a ring to close.
    return part.list.len() - (part.list.len() > 0)


def _per_geometry(values: pl.Expr, layers: int, reduce: Reduce, rings: bool) -> pl.Expr:
    """Reduce a geometry's innermost parts, and add up what that gives."""

    # The reduction lands on the innermost lists by recursion
    # (e.g. a polygon's rings, a linestring's vertices).
    # This way we can use native list kernels.
    if layers == 1:
        return reduce(values, rings)
    inner = _per_geometry(pl.element(), layers - 1, reduce, rings)
    return values.list.eval(inner).list.sum()


def _flat(geometry: GeoArrowType) -> bool:
    """
    A polygon nests one layer deeper and repeats a coordinate
    per ring, so it has to go the long way round.
    """
    return geometry._nesting == 1 and not geometry._rings


def _mean(geometry: GeoArrowType, column: pl.Expr, axis: str) -> pl.Expr:
    """The mean of one axis over a geometry's coordinates, per row."""
    storage = column.ext.storage()

    if _flat(geometry):
        return _axis(storage, 1, axis).list.mean()

    nesting, rings = geometry._nesting, geometry._rings
    total = _per_geometry(_axis(storage, nesting, axis), nesting, _total, rings)
    count = _per_geometry(storage, nesting, _count, rings)
    return total / count


def _coordinates(geometry: GeoArrowType, column: pl.Expr) -> pl.Expr:
    """How many coordinates a geometry holds, over all of its parts."""
    storage = column.ext.storage()
    return _per_geometry(storage, geometry._nesting, _count, geometry._rings)


def _mean_coordinate(
    column: pl.Expr,
    geometry: GeoArrowType,
    empty: Literal["null", "empty_point"] = "null",
) -> pl.Expr:
    """The mean coordinate of one geometry type, per row."""
    # A point is its own mean coordinate.
    if geometry._nesting == 0:
        return column

    # `m` is a measure rather than an axis, but it averages like one.
    dimension = geometry._dimension
    means = [_mean(geometry, column, axis).alias(axis) for axis in dimension]
    # Geometries don't have interior nulls (see spec), so only check whether
    # it has a coordinate at all. A null geometry counts null, which `when`
    # reads as false.
    coordinates = _coordinates(geometry, column)
    mean = pl.when(coordinates > 0).then(pl.struct(means))
    if empty == "empty_point":
        nan = [pl.lit(math.nan).alias(axis) for axis in dimension]
        mean = mean.when(coordinates == 0).then(pl.struct(nan))
    return mean.ext.to(PointType.of_dimension(dimension)._with_metadata_of(geometry))


def mean_coordinate(geometry: IntoExprColumn) -> pl.Expr:
    """The mean of the coordinates a geometry is made of, as a `geoarrow.point`.

    Note that this is *not* the centroid (centre of mass).
    Every coordinate counts once and counts the same,
    so this is the centre of a geometry's vertices rather than of the space it covers.
    For a linestring or a polygon those differ:
    this one moves with where the vertices are dense.

    =======================  =============
    in                       out
    =======================  =============
    `PointXY[ZM]`            `PointXY[ZM]`
    `LineStringXY[ZM]`       `PointXY[ZM]`
    `PolygonXY[ZM]`          `PointXY[ZM]`
    `MultiPointXY[ZM]`       `PointXY[ZM]`
    `MultiLineStringXY[ZM]`  `PointXY[ZM]`
    `MultiPolygonXY[ZM]`     `PointXY[ZM]`
    =======================  =============

    The measure `m` is also averaged.
    A null geometry, and a geometry with no coordinates at all,
    has no mean coordinate.

    .. code-block:: python

        df.select(geo.mean_coordinate("route"))
    """
    return on_geometry(geometry, _mean_coordinate)
