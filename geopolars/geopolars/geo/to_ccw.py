"""Making a geometry run counter-clockwise (or clockwise)."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.datatypes import (
    LineStringType,
    MultiLineStringType,
    MultiPolygonType,
    PolygonType,
)
from geopolars.geo._dispatch import on_geometry
from geopolars.geo.reverse import reverse

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _to_ccw(column: pl.Expr, geometry: GeoArrowType, *, holes_cw: bool) -> pl.Expr:
    if not isinstance(
        geometry,
        (LineStringType, PolygonType, MultiLineStringType, MultiPolygonType),
    ):
        msg = (
            "to_ccw expects a linestring, polygon, multilinestring"
            f" or multipolygon column, got: {geometry!r}"
        )
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="to_ccw",
        kwargs={"holes_cw": holes_cw},
        is_elementwise=True,
    )


@doc()
def to_ccw(geometry: IntoExprColumn, *, holes_cw: bool = True) -> pl.Expr:
    """The geometry, with every ring that runs the wrong way for `is_ccw` reversed.

    Takes the same geometries as `is_ccw`.
    {keeps_type_dimension_and_crs}
    A polygon gets its exterior counter-clockwise and its holes clockwise.
    With `holes_cw=False`, its holes run counter-clockwise too.
    A multilinestring or multipolygon has every part made so, and keeps the order of its parts.

    A ring that runs neither way (see `is_ccw`) is left as it is.
    {ignores_empty_and_invalid}

    .. code-block:: python

        df.select(geo.to_ccw("parcel"))
    """
    return on_geometry(geometry, partial(_to_ccw, holes_cw=holes_cw))


def to_cw(geometry: IntoExprColumn, *, holes_ccw: bool = True) -> pl.Expr:
    """`to_ccw`, but with every ring the other way around.

    A polygon gets its exterior clockwise and its holes counter-clockwise.
    With `holes_ccw=False`, its holes run clockwise too.

    This is `to_ccw`, then `reverse` of every ring.

    .. code-block:: python

        df.select(geo.to_cw("parcel"))
    """
    return reverse(
        to_ccw(geometry, holes_cw=holes_ccw), reverse_holes=True, nested=False
    )
