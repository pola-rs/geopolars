"""Reversing the order of a geometry's vertices."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.datatypes import PointType
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _reverse(
    column: pl.Expr, geometry: GeoArrowType, *, reverse_holes: bool, nested: bool
) -> pl.Expr:
    if isinstance(geometry, PointType):
        msg = f"reverse does not accept a point column, got: {geometry!r}"
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="reverse",
        kwargs={"reverse_holes": reverse_holes, "nested": nested},
        is_elementwise=True,
    )


@doc()
def reverse(
    geometry: IntoExprColumn, *, reverse_holes: bool = False, nested: bool = True
) -> pl.Expr:
    """The geometry with the order of its vertices reversed.
    {keeps_type_dimension_and_crs}

    | in | reversed |
    | - | - |
    | `PointXY[ZM]` | refused |
    | `LineStringXY[ZM]` | its vertices |
    | `PolygonXY[ZM]` | its exterior |
    | `MultiPointXY[ZM]` | the order of its points |
    | `MultiLineStringXY[ZM]` | every linestring |
    | `MultiPolygonXY[ZM]` | the exterior of every polygon |

    With `reverse_holes=True`, the holes of a (multi)polygon are reversed too.
    The order of a polygon's rings is kept.

    With `nested=True`, the order of the parts of a multilinestring or multipolygon
    is reversed as well.
    For a multilinestring, this reverses it as a whole, which is the cheaper option.
    For a multipolygon, it is the more expensive one.
    With `nested=False`, every part keeps its place.

    {ignores_empty_and_invalid}

    ```python
    df.select(geo.reverse("route"))
    ```
    """
    return on_geometry(
        geometry, partial(_reverse, reverse_holes=reverse_holes, nested=nested)
    )
