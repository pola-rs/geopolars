"""The interior rings (holes) of a polygon, as a multilinestring.

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


def _interior(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    if not isinstance(geometry, (PolygonType, MultiPolygonType)):
        msg = f"interior expects a polygon or multipolygon column, got: {geometry!r}"
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="interior",
        is_elementwise=True,
    )


def interior(geometry: IntoExprColumn) -> pl.Expr:
    """The interior rings (holes) of a polygon, as a multilinestring
    with the polygon's dimension and CRS. The exterior is left out.
    A multipolygon gives a list instead, one multilinestring per polygon.

    =======================  =============================
    in                       out
    =======================  =============================
    `PointXY[ZM]`            refused
    `LineStringXY[ZM]`       refused
    `PolygonXY[ZM]`          `MultiLineStringXY[ZM]`
    `MultiPointXY[ZM]`       refused
    `MultiLineStringXY[ZM]`  refused
    `MultiPolygonXY[ZM]`     `list[MultiLineStringXY[ZM]]`
    =======================  =============================

    The rings stay closed, as they are stored.
    A polygon without holes, or without any rings, gives an empty multilinestring.
    A null polygon gives a null multilinestring.

    .. code-block:: python

        df.select(geo.interior("parcel"))
    """
    return on_geometry(geometry, _interior)
