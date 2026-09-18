"""Averaging over coordinates.
Note that this is different from an area centroid!
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl

from geopolars.datatypes import GEOMETRIES, GeoPoint
from geopolars.datatypes.dimension import ALL as DIMENSIONS
from geopolars.datatypes.dimension import XYZM
from geopolars.geometry._dispatch import by_geometry

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType
    from geopolars.datatypes.dimension import Dimension
    from geopolars.geometry._dispatch import Where

# A reduction over the coordinates of one innermost part of a geometry.
Reduce = Callable[[pl.Expr, bool], pl.Expr]


def _all(conditions: Iterable[pl.Expr]) -> pl.Expr:
    """Every one of these holds."""
    # Not `pl.all_horizontal`: a dispatched expression vanishes when the column
    # is not its geometry, and that collapses to `True` instead of vanishing
    # with it. `&` is a method on the expression, so it goes along quietly.
    return functools.reduce(operator.and_, conditions)


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


def _missing_coordinates(part: pl.Expr, rings: bool) -> pl.Expr:
    # GeoArrow allows nulls only at the outermost level.
    # Anything counted here is a geometry that should not exist.
    return part.list.count_matches(None)


def _missing_parts(parts: pl.Expr, rings: bool) -> pl.Expr:
    # `count_matches` only compares values,
    # Ask each part whether it is there.
    return parts.list.eval(pl.element().is_null()).list.sum()


def _per_geometry(values: pl.Expr, layers: int, reduce: Reduce, rings: bool) -> pl.Expr:
    """Reduce a geometry's innermost parts, and add up what that gives."""

    # The reduction lands on the innermost lists by recursion
    # (e.g. a polygon's rings, a linestring's vertices).
    # This way we can use native list kernels.
    if layers == 1:
        return reduce(values, rings)
    inner = _per_geometry(pl.element(), layers - 1, reduce, rings)
    return values.list.eval(inner).list.sum()


def _mean(geometry: type[GeoArrowType], column: pl.Expr, axis: str) -> pl.Expr:
    """The mean of one axis over a geometry's coordinates, per row."""
    storage = column.ext.storage()
    values = _axis(storage, geometry._nesting, axis)
    total = _per_geometry(values, geometry._nesting, _total, geometry._rings)
    count = _per_geometry(storage, geometry._nesting, _count, geometry._rings)
    return total / count


def _defined(
    geometry: type[GeoArrowType], column: pl.Expr, dimension: Dimension
) -> pl.Expr:
    """can this type and instance of geometry have a centroid at all?"""
    # Cases to check:
    # - polygon with a null ring
    # - null geometry
    # https://geoarrow.org/format.html#missing-values-null
    storage = column.ext.storage()
    nesting, rings = geometry._nesting, geometry._rings

    whole = [
        _per_geometry(
            _axis(storage, nesting, axis), nesting, _missing_coordinates, rings
        )
        == 0
        for axis in dimension
    ]
    whole += [
        _per_geometry(storage, layer, _missing_parts, rings=False) == 0
        for layer in range(1, nesting)
    ]
    return _all([_per_geometry(storage, nesting, _count, rings) > 0, *whole])


def _branches(where: Where) -> Iterator[pl.Expr]:
    """One centroid per geometry the column could turn out to hold."""
    nested = [geometry for geometry in GEOMETRIES if geometry._nesting > 0]

    # A point is its own centroid, and its storage is already the coordinate
    # struct a point wraps: nothing to average, nothing to build.
    for dimension in DIMENSIONS:
        point = GeoPoint.of_dimension(dimension)
        column = where(point)
        if column is not None:
            storage = column.ext.storage()
            present = _all(storage.struct.field(a).is_not_null() for a in dimension)
            yield pl.when(present).then(storage).ext.to(point())

    defined = []
    for geometry in nested:
        for dimension in DIMENSIONS:
            column = where(geometry.of_dimension(dimension))
            if column is not None:
                defined.append(_defined(geometry, column, dimension))
    # The literal is what keeps a coalesce from being empty,
    # and says "no centroid" for a column that is none of these.
    whole = pl.coalesce([*defined, pl.lit(False)])

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
            pl.when(column.is_not_null() & whole)
            .then(pl.struct(**{axis: coordinates[axis] for axis in dimension}))
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
    A geometry with no coordinates at all or an invalid coordinate has no centroid.

    ```python
    df.select(geometry.coordinate_centroid("route"))
    ```
    """
    return by_geometry(geometry, _branches)
