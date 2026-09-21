"""Averaging over coordinates.
Note that this is different from an area centroid!
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl

from geopolars.datatypes import GEOMETRIES, GeoPoint
from geopolars.datatypes.dimension import ALL as DIMENSIONS
from geopolars.datatypes.dimension import XYZM
from geopolars.geometry._dispatch import by_geometry

if TYPE_CHECKING:
    from collections.abc import Iterator

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType
    from geopolars.geometry._dispatch import Where

# A reduction over the coordinates of one innermost part of a geometry.
Reduce = Callable[[pl.Expr, bool], pl.Expr]


def _axis(values: pl.Expr, layers: int, axis: str) -> pl.Expr:
    """One axis of the coordinates."""
    # should keep this element-wise,
    # so not all geometries are walked at the same time.
    if layers == 0:
        return values.struct.field(axis)
    return values.list.eval(_axis(pl.element(), layers - 1, axis))


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


def _flat(geometry: type[GeoArrowType]) -> bool:
    """
    A polygon nests one layer deeper and repeats a coordinate
    per ring, so it has to go the long way round.
    """
    return geometry._nesting == 1 and not geometry._rings


def _mean(geometry: type[GeoArrowType], column: pl.Expr, axis: str) -> pl.Expr:
    """The mean of one axis over a geometry's coordinates, per row."""
    storage = column.ext.storage()

    if _flat(geometry):
        return _axis(storage, 1, axis).list.mean()

    nesting, rings = geometry._nesting, geometry._rings
    total = _per_geometry(_axis(storage, nesting, axis), nesting, _total, rings)
    count = _per_geometry(storage, nesting, _count, rings)
    return total / count


def _coordinates(geometry: type[GeoArrowType], column: pl.Expr) -> pl.Expr:
    """How many coordinates a geometry holds, over all of its parts."""
    storage = column.ext.storage()
    return _per_geometry(storage, geometry._nesting, _count, geometry._rings)


def _branches(where: Where) -> Iterator[pl.Expr]:
    """One centroid per geometry the column could turn out to hold."""
    nested = [geometry for geometry in GEOMETRIES if geometry._nesting > 0]

    # A point is its own centroid. Its storage is already the coordinate struct
    # a point wraps and its dtype is already the one to hand back.
    for dimension in DIMENSIONS:
        column = where(GeoPoint.of_dimension(dimension))
        if column is not None:
            yield column

    # Geometries don't have interior nulls (see spec).
    # Only need to check if the geometry has a coordinate at all.
    # `list.len` reads that off the offsets,
    counted = []
    for geometry in nested:
        for dimension in DIMENSIONS:
            column = where(geometry.of_dimension(dimension))
            if column is not None:
                counted.append(_coordinates(geometry, column) > 0)
    # The literal is what keeps a coalesce from being empty,
    # and says "no centroid" for a column that is none of these.
    filled = pl.coalesce([*counted, pl.lit(False)])

    # `m` is a measure rather than an axis, but it averages like one.
    coordinates: dict[str, pl.Expr] = {}
    for axis in XYZM:
        means = []
        for geometry in nested:
            carries = [t for t in geometry.dimensions() if axis in t._dimension]
            column = where(*carries)
            if column is not None:
                means.append(_mean(geometry, column, axis))
        coordinates[axis] = pl.coalesce([*means, pl.lit(None, dtype=pl.Float64)])

    for dimension in DIMENSIONS:
        column = where(*(geometry.of_dimension(dimension) for geometry in nested))
        if column is None:
            continue
        yield (
            pl.when(column.is_not_null() & filled)
            .then(pl.struct([coordinates[axis].alias(axis) for axis in dimension]))
            .ext.to(GeoPoint.of_dimension(dimension)())
        )


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
    A null geometry, and a geometry with no coordinates at all, has no centroid.

    ```python
    df.select(geometry.coordinate_centroid("route"))
    ```
    """
    return by_geometry(geometry, _branches)
