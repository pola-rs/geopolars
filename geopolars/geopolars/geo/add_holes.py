"""Adding holes to polygons, from another column.

A native kernel (A-tier) copies the polygons and their new holes into new buffers.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.datatypes import MultiLineStringType, PolygonType
from geopolars.geo._dispatch import _to_expr, on_geometry
from geopolars.geo.parts import _parts

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType
    from geopolars.geo.close_rings import CloseRings


def _as_list_of_rings(holes: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    if isinstance(dtype, MultiLineStringType):
        return _parts(holes, dtype)
    return holes


def _add_holes(
    column: pl.Expr,
    geometry: GeoArrowType,
    holes: IntoExprColumn,
    close: CloseRings | None,
) -> pl.Expr:
    if not isinstance(geometry, PolygonType):
        msg = f"add_holes expects a polygon column, got: {geometry!r}"
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column, _to_expr(holes).pipe_with_dtype(_as_list_of_rings)],
        function_name="add_holes",
        is_elementwise=True,
        kwargs={"close": None if close is None else asdict(close)},
    )


@doc()
def add_holes(
    polygon: IntoExprColumn,
    holes: IntoExprColumn,
    *,
    close: CloseRings | None = None,
) -> pl.Expr:
    """Add the rings of `holes` to each polygon as extra holes,
    after the holes it already has.
    `holes` holds one linestring per polygon, a list of them, or a multilinestring.

    | polygon | holes | out |
    | - | - | - |
    | `PolygonXY[ZM]` | `LineStringXY[ZM]` | `PolygonXY[ZM]` |
    | `PolygonXY[ZM]` | `list[LineStringXY[ZM]]` | `PolygonXY[ZM]` |
    | `PolygonXY[ZM]` | `MultiLineStringXY[ZM]` | `PolygonXY[ZM]` |
    | anything else | | refused |

    The result keeps the polygon's dimension and CRS.
    Holes of another dimension, or holes that declare another CRS, are refused.

    A null polygon gives a null polygon.
    A null `holes` value, or an empty list, leaves the polygon as it is.
    A null ring among the holes makes the polygon null.
    An empty polygon (one without rings) that gets holes is null,
    as it has no exterior to put them in.

    Every ring has to be a ring (see [`is_ring`][geopolars.geo.is_ring]),
    or the polygon is null.
    With `close=CloseRings(...)`, an open ring is first closed as `close_rings` does.

    {closing_rings}

    ```python
    df.select(geo.add_holes("parcel", "ponds"))
    ```
    """
    return on_geometry(
        polygon,
        lambda column, geometry: _add_holes(column, geometry, holes, close),
    )
