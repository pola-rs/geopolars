"""Whether a geometry runs counter-clockwise, with a polygon's holes clockwise."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import (
    LineStringType,
    MultiLineStringType,
    MultiPolygonType,
    PolygonType,
)
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _is_ccw(column: pl.Expr, geometry: GeoArrowType, *, holes_cw: bool) -> pl.Expr:
    if not isinstance(
        geometry,
        (LineStringType, PolygonType, MultiLineStringType, MultiPolygonType),
    ):
        msg = (
            "is_ccw expects a linestring, polygon, multilinestring"
            f" or multipolygon column, got: {geometry!r}"
        )
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="is_ccw",
        kwargs={"holes_cw": holes_cw},
        is_elementwise=True,
    )


def is_ccw(geometry: IntoExprColumn, *, holes_cw: bool = True) -> pl.Expr:
    """Whether a geometry runs counter-clockwise.

    A linestring has to run counter-clockwise.
    A polygon has to have its exterior counter-clockwise and its holes clockwise.
    With `holes_cw=False`, its holes have to run counter-clockwise too.
    The default is how GeoParquet and OGC wind a polygon.
    A multilinestring or multipolygon has to have every part run that way.

    =======================  =======
    in                       out
    =======================  =======
    `PointXY[ZM]`            refused
    `LineStringXY[ZM]`       `bool`
    `PolygonXY[ZM]`          `bool`
    `MultiPointXY[ZM]`       refused
    `MultiLineStringXY[ZM]`  `bool`
    `MultiPolygonXY[ZM]`     `bool`
    =======================  =======

    The orientation is the sign of the area the ring encloses in x and y; z and m are ignored.
    An empty linestring, ring, polygon or multi runs either way, so it gives `True`.
    Any other ring that encloses nothing (too few vertices, all on a line,
    or with a NaN vertex) runs neither way, so it gives `False`.

    Whether a line is closed is not checked: an open line is taken as closed back to its start.
    Combine with `is_ring` or `is_valid_geoarrow` to check that too.

    .. code-block:: python

        df.select(geo.is_ccw("parcel"))
    """
    return on_geometry(geometry, partial(_is_ccw, holes_cw=holes_cw))
