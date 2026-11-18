"""How many coordinates a geometry is made of."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _count_coordinates(
    column: pl.Expr, geometry: GeoArrowType, *, include_empty: bool
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="count_coordinates",
        kwargs={"include_empty": include_empty},
        is_elementwise=True,
    )


@doc()
def count_coordinates(
    geometry: IntoExprColumn, *, include_empty: bool = False
) -> pl.Expr:
    """How many coordinates a geometry is made of, over all of its parts.

    This gives one `UInt32` count per geometry, however deeply it is nested.

    =======================  =====================================================
    in                       counts
    =======================  =====================================================
    `PointXY[ZM]`            `1`, or `0` when it is empty
    `LineStringXY[ZM]`       its non-empty vertices
    `PolygonXY[ZM]`          the non-empty vertices of every ring
    `MultiPointXY[ZM]`       its non-empty points
    `MultiLineStringXY[ZM]`  the non-empty vertices of every linestring
    `MultiPolygonXY[ZM]`     the non-empty vertices of every ring of every polygon
    =======================  =====================================================

    With `include_empty=True`, an empty point counts like any other.
    {empty_point}
    This skips a ring's last coordinate, as this repeats its first.
    A geometry without coordinates counts `0`; a null geometry gives null.

    .. code-block:: python

        df.select(geo.count_coordinates("parcel"))
    """
    return on_geometry(
        geometry, partial(_count_coordinates, include_empty=include_empty)
    )
