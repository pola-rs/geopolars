"""Adding holes to polygons, from another column.

A native kernel (A-tier) copies the polygons and their new holes into new buffers.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._typing import EmptyPolygonExterior, NullParts, OnRowError
from geopolars._utils import LIB, validate_one_of
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
    null_parts: NullParts,
    empty_polygon_exterior: EmptyPolygonExterior,
    on_row_error: OnRowError,
) -> pl.Expr:
    if not isinstance(geometry, PolygonType):
        msg = f"add_holes expects a polygon column, got: {geometry!r}"
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column, _to_expr(holes)],
        function_name="add_holes",
        is_elementwise=True,
        kwargs={
            "close": None if close is None else asdict(close),
            "null_parts": null_parts,
            "empty_polygon_exterior": empty_polygon_exterior,
            "on_row_error": on_row_error,
        },
    )


@doc(parts="holes", part="hole", geometry="polygon")
def add_holes(
    polygon: IntoExprColumn,
    holes: IntoExprColumn,
    *,
    close: CloseRings | None = None,
    null_parts: NullParts = "null",
    empty_polygon_exterior: EmptyPolygonExterior = "null",
    on_row_error: OnRowError = "raise",
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

    A linestring can also be given as its vertices, as `list[PointXY]` or `MultiPointXY`.
    Bare coordinates (the storage of points) work in place of the points.
    A list of linestrings is then a list of those.

    {paired_rows}
    With a single linestring as `holes`, every polygon gets that same hole.
    With a single polygon, the result has one copy of it per row of `holes`,
    each with the holes of that row.

    The result keeps the polygon's dimension and CRS.
    Holes of another dimension are refused.
    {same_crs}

    A null polygon gives a null polygon.
    {null_parts}

    An empty list adds no holes.
    The polygon's own rings are still checked, as described below.
    {empty_polygon_exterior}
    With `"empty"`, such a polygon stays as it is: none of the holes of its row are added.
    With `"keep"`, a polygon stored without rings first gets an empty exterior,
    so that its new rings stay holes.
    A polygon with an empty exterior that only gets empty holes takes them,
    and stays an empty polygon.

    A polygon is null if any of its rings is not a ring (see [`is_ring`][geopolars.geo.is_ring]),
    its existing rings included.
    A polygon whose rings are all empty is the exception, and stays as it is.
    With `empty_polygon_exterior="keep"`, so does one with an empty exterior
    whose holes are all rings or empty.
    Passing `close=CloseRings(...)` first closes every open ring, as `close_rings` does.

    {closing_rings}

    {on_row_error}

    ```python
    df.select(geo.add_holes("parcel", "ponds"))
    ```
    """
    validate_one_of("null_parts", null_parts, NullParts)
    validate_one_of(
        "empty_polygon_exterior", empty_polygon_exterior, EmptyPolygonExterior
    )
    validate_one_of("on_row_error", on_row_error, OnRowError)
    return on_geometry(
        polygon,
        lambda column, geometry: _add_holes(
            column,
            geometry,
            holes,
            close,
            null_parts,
            empty_polygon_exterior,
            on_row_error,
        ),
    )
