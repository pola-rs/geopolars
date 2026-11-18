"""The boundary of a geometry: a polygon's rings, or a linestring's endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from geopolars.datatypes import (
    LineStringType,
    MultiLineStringType,
    MultiPointType,
    MultiPolygonType,
    PolygonType,
)
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars._typing import PolarsDataType

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _endpoints(
    line: pl.Expr, points_storage: PolarsDataType, axes: Sequence[str]
) -> pl.Expr:
    """The first and last vertex of one linestring's storage,
    or none if the line is closed (they are equal on `axes`)."""
    first, last = line.list.first(), line.list.last()
    no_points = pl.lit([], dtype=points_storage)
    # `concat_list` would turn a null line into two null points.
    return (
        pl.when(line.is_null())
        .then(None)
        .when(line.list.len() == 0)
        .then(no_points)
        .when(
            pl.all_horizontal(
                first.struct.field(axis) == last.struct.field(axis) for axis in axes
            )
        )
        .then(no_points)
        .otherwise(pl.concat_list(first, last))
    )


def _boundary(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    storage = column.ext.storage()
    if isinstance(geometry, (PolygonType, MultiPolygonType)):
        rings_type = MultiLineStringType.of_dimension(geometry._dimension)
        rings = rings_type._with_metadata_of(geometry)
        if isinstance(geometry, MultiPolygonType):
            # Polars can't cast to a list of an extension type, so relabel inside
            # `list.eval`. Its body only changes the type, so this costs nothing.
            return storage.list.eval(pl.element().ext.to(rings))
        # A polygon's storage already is a list of rings: only the type changes.
        return storage.ext.to(rings)
    if isinstance(geometry, (LineStringType, MultiLineStringType)):
        ends_type = MultiPointType.of_dimension(geometry._dimension)
        ends = ends_type._with_metadata_of(geometry)
        # m is a measure along the line, not a position: it doesn't close a line.
        axes = [axis for axis in geometry._dimension if axis != "m"]
        if isinstance(geometry, MultiLineStringType):
            # One pair of endpoints per part. A null multilinestring stays null.
            return storage.list.eval(
                _endpoints(pl.element(), ends_type._geo_storage, axes).ext.to(ends)
            )
        return _endpoints(storage, ends_type._geo_storage, axes).ext.to(ends)
    msg = (
        "boundary expects a (multi)linestring or (multi)polygon column, "
        f"got: {geometry!r}"
    )
    raise TypeError(msg)


def boundary(geometry: IntoExprColumn) -> pl.Expr:
    """The boundary of a geometry, with its dimension and CRS.
    A polygon gives its rings, the exterior first and then its holes.
    A linestring gives its first and last vertex, or none if it is closed.
    A multi-geometry gives a list, one boundary per part.

    =======================  =============================
    in                       out
    =======================  =============================
    `PointXY[ZM]`            refused
    `LineStringXY[ZM]`       `MultiPointXY[ZM]`
    `PolygonXY[ZM]`          `MultiLineStringXY[ZM]`
    `MultiPointXY[ZM]`       refused
    `MultiLineStringXY[ZM]`  `list[MultiPointXY[ZM]]`
    `MultiPolygonXY[ZM]`     `list[MultiLineStringXY[ZM]]`
    =======================  =============================

    The rings stay closed, as they are stored.
    A linestring is closed when its first and last vertex lie at exactly the same place
    in x, y and z.
    m is a measure along the line rather than a position, so it is not compared.
    A closed linestring, or one with a single vertex, has an empty boundary.

    A polygon without rings, or a linestring without vertices, gives an empty boundary.
    A null geometry gives a null boundary.
    Points and multipoints have no boundary, and are refused.

    .. code-block:: python

        df.select(geo.boundary("parcel"))
    """
    return on_geometry(geometry, _boundary)
