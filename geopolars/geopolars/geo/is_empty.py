"""Whether a geometry holds anything at all."""

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


def _is_empty(
    column: pl.Expr, geometry: GeoArrowType, *, check_nested: bool
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="is_empty",
        kwargs={"check_nested": check_nested},
        is_elementwise=True,
    )


@doc()
def is_empty(geometry: IntoExprColumn, *, check_nested: bool = False) -> pl.Expr:
    """Whether a geometry is empty.

    =======================  =====================================================
    in                       empty when
    =======================  =====================================================
    `PointXY[ZM]`            it is the empty point (see below)
    `LineStringXY[ZM]`       it has no vertices
    `PolygonXY[ZM]`          it has no rings, or only rings without vertices
    `MultiPointXY[ZM]`       it has no points
    `MultiLineStringXY[ZM]`  it has no parts, or only parts without vertices
    `MultiPolygonXY[ZM]`     it has no polygons, or only polygons without vertices
    =======================  =====================================================

    By default, this only checks for outer emptiness.
    A linestring with only empty points would still not be considered empty.
    A single non-empty point makes it non-empty.
    A null geometry is neither empty nor not: it gives null.
    With `check_nested=True`, any geometry but a point is also empty
    when all of its points are empty.
    This would catch a case like a linestring with only empty points,
    but as it reads every coordinate, it is much slower.

    {empty_point}

    .. code-block:: python

        df.filter(~geo.is_empty("parcel"))
    """
    return on_geometry(geometry, partial(_is_empty, check_nested=check_nested))
