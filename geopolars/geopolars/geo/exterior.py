"""The exterior (outer) ring of a polygon, as a linestring.

A native kernel (A-tier) copies the rings out in bulk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import MultiPolygonType, PolygonType
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _exterior(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    if not isinstance(geometry, (PolygonType, MultiPolygonType)):
        msg = f"exterior expects a polygon or multipolygon column, got: {geometry!r}"
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="exterior",
        is_elementwise=True,
    )


def exterior(geometry: IntoExprColumn) -> pl.Expr:
    """The exterior (outer) ring of a polygon (holes left out),
    as a closed linestring with the polygon's dimension and CRS.
    A multipolygon gives a list of exteriors instead, one per polygon.

    =======================  ========================
    in                       out
    =======================  ========================
    `PointXY[ZM]`            refused
    `LineStringXY[ZM]`       refused
    `PolygonXY[ZM]`          `LineStringXY[ZM]`
    `MultiPointXY[ZM]`       refused
    `MultiLineStringXY[ZM]`  refused
    `MultiPolygonXY[ZM]`     `list[LineStringXY[ZM]]`
    =======================  ========================

    A polygon without rings gives an empty linestring.
    A null polygon gives a null linestring.

    .. code-block:: python

        df.select(geo.exterior("parcel"))
    """
    return on_geometry(geometry, _exterior)
