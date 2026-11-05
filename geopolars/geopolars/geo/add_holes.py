"""Adding holes to polygons, from another column.

A native kernel (A-tier) copies the polygons and their new holes into new buffers.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.datatypes import PolygonType
from geopolars.geo._dispatch import _to_expr, on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType
    from geopolars.geo.close_rings import CloseRings


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
        args=[column, _to_expr(holes)],
        function_name="add_holes",
        is_elementwise=True,
        kwargs={"close": None if close is None else asdict(close)},
    )


@doc(parts="holes", geometry="polygon")
def add_holes(
    polygon: IntoExprColumn,
    holes: IntoExprColumn,
    *,
    close: CloseRings | None = None,
) -> pl.Expr:
    """Add the rings of `holes` to each polygon as extra holes,
    after the holes it already has.
    `holes` holds one linestring per polygon (or its bare vertex list),
    a list of them, or a multilinestring.

    | polygon | holes | out |
    | - | - | - |
    | `PolygonXY[ZM]` | `LineStringXY[ZM]` | `PolygonXY[ZM]` |
    | `PolygonXY[ZM]` | `list[LineStringXY[ZM]]` | `PolygonXY[ZM]` |
    | `PolygonXY[ZM]` | `MultiLineStringXY[ZM]` | `PolygonXY[ZM]` |
    | anything else | | refused |


    {paired_rows}
    With a single linestring as `holes`, every polygon gets that same hole.
    With a single polygon, the result has one copy of it per row of `holes`,
    each with the holes of that row.

    The result keeps the polygon's dimension and CRS.
    Holes of another dimension are refused.
    {same_crs}

    A null polygon gives a null polygon.
    A null `holes` value, or an empty list adds no holes.
    The polygon's own rings are still checked, as described below.
    A null ring or a null vertex among the holes makes the polygon null.
    An empty polygon that gets holes with vertices is null,
    as it has no exterior to put them in.
    Empty holes can still be added to it, and it stays an empty polygon.

    Every ring of the result has to be a ring (see [`is_ring`][geopolars.geo.is_ring]),
    the polygon's own existing rings included,
    or the polygon is null.
    A polygon whose rings are all empty is the exception, and stays as it is.
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
