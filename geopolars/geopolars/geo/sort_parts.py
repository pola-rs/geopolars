"""Sorting the parts of a geometry into a fixed order."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal, get_args

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.datatypes import LineStringType, PointType
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

Rings = Literal["keep", "rotate"]


def _sort_parts(
    column: pl.Expr,
    geometry: GeoArrowType,
    *,
    descending: bool,
    rings: Rings,
    ignore_per_row_errors: bool,
) -> pl.Expr:
    if isinstance(geometry, PointType) or (
        isinstance(geometry, LineStringType) and rings == "keep"
    ):
        return column
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="sort_parts",
        kwargs={
            "descending": descending,
            "rings": rings,
            "ignore_per_row_errors": ignore_per_row_errors,
        },
        is_elementwise=True,
    )


@doc(output="geometry")
def sort_parts(
    geometry: IntoExprColumn,
    *,
    descending: bool = False,
    rings: Rings = "keep",
    ignore_per_row_errors: bool = False,
) -> pl.Expr:
    """The geometry with its parts sorted.
    {keeps_type_dimension_and_crs}

    | in | sorted | rotated with `rings="rotate"` |
    | - | - | - |
    | `PointXY[ZM]` | nothing, returned as it is | nothing |
    | `LineStringXY[ZM]` | nothing, returned as it is | the linestring |
    | `PolygonXY[ZM]` | its holes (the exterior stays first) | every ring |
    | `MultiPointXY[ZM]` | its points | nothing |
    | `MultiLineStringXY[ZM]` | its linestrings | every linestring |
    | `MultiPolygonXY[ZM]` | the holes of every polygon, then its polygons | every ring |

    Two geometries that only differ in the order of their parts
    come out the same, so they can be compared part by part.
    The vertices of a part keep their order, unless rings are rotated.

    Parts are ordered as Polars sorts a geometry column.
    They are compared vertex by vertex,
    and each vertex on every axis it has (x, then y, z, and m).
    A part that runs out first, being equal up to there, is the smaller one.
    Polygons are compared ring by ring, with their holes already sorted.
    NaN is greater than any other value, and `-0.0` is equal to `0.0`.

    With `descending=True`, the parts are sorted from greatest to smallest.
    Equal parts keep their order either way.

    With `rings="rotate"`, every ring first starts at the vertex
    from which its vertices are smallest in the order above, keeping its direction.
    The closing vertex is left out of this comparison.
    Of starts that make it equally small, the earliest is kept.
    The parts are then compared as they are once rotated.
    A ring that already starts there is left as it is.
    Any other ring is closed again by its new first vertex,
    which replaces the old closing vertex (and so its m).
    The start does not depend on `descending`.

    With `rings="rotate"`, every ring has to be closed (see `close_rings`), or empty.
    This includes the linestrings that the table says are rotated.
    Otherwise this raises, or with `ignore_per_row_errors=True`,
    makes its row null.
    A ring of fewer than four vertices is fine, unlike for `is_ring`.
    A ring whose first vertex has NaN in x, y, or z never counts as closed,
    as NaN does not equal itself.

    {empty_and_null}

    ```python
    df.select(geo.sort_parts("islands", rings="rotate"))
    ```
    """
    if rings not in get_args(Rings):
        msg = f"`rings` must be one of {list(get_args(Rings))}, got {rings!r}"
        raise ValueError(msg)
    return on_geometry(
        geometry,
        partial(
            _sort_parts,
            descending=descending,
            rings=rings,
            ignore_per_row_errors=ignore_per_row_errors,
        ),
    )
