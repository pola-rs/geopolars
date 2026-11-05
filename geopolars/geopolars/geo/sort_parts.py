"""Sorting the parts of a geometry into a fixed order."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.datatypes import LineStringType, PointType
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _sort_parts(
    column: pl.Expr, geometry: GeoArrowType, *, descending: bool
) -> pl.Expr:
    if isinstance(geometry, (PointType, LineStringType)):
        return column
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="sort_parts",
        kwargs={"descending": descending},
        is_elementwise=True,
    )


@doc()
def sort_parts(geometry: IntoExprColumn, *, descending: bool = False) -> pl.Expr:
    """The geometry with its parts sorted.
    {keeps_type_dimension_and_crs}

    | in | sorted |
    | - | - |
    | `PointXY[ZM]` | nothing, returned as it is |
    | `LineStringXY[ZM]` | nothing, returned as it is |
    | `PolygonXY[ZM]` | its holes (the exterior stays first) |
    | `MultiPointXY[ZM]` | its points |
    | `MultiLineStringXY[ZM]` | its linestrings |
    | `MultiPolygonXY[ZM]` | the holes of every polygon, then its polygons |

    Two geometries that only differ in the order of their parts
    come out the same, so they can be compared part by part.
    The vertices of a part keep their order.

    Parts are ordered as Polars sorts a geometry column.
    They are compared vertex by vertex,
    and each vertex on every axis it has (x, then y, z, and m).
    A part that runs out first, being equal up to there, is the smaller one.
    Polygons are compared ring by ring, with their holes already sorted.
    NaN is greater than any other value, and `-0.0` is equal to `0.0`.

    With `descending=True`, the parts are sorted from greatest to smallest.
    Equal parts keep their order either way.

    A null geometry stays null, and an empty one stays empty.

    ```python
    df.select(geo.sort_parts("islands"))
    ```
    """
    return on_geometry(geometry, partial(_sort_parts, descending=descending))
