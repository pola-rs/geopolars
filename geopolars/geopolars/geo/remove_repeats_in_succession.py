"""Removing coinciding points in succession (A-tier)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB, validate_coincidence_tolerances, validate_one_of
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

CollapsedLines = Literal["keep_ends", "single_vertex"]
TooShort = Literal["null", "keep", "empty"]


@doc()
def remove_repeats_in_succession(
    geometry: IntoExprColumn,
    *,
    abs_tol: float | None = None,
    rel_tol: float | None = None,
    unique_z: bool = True,
    unique_m: bool = False,
    collapsed_lines: CollapsedLines = "keep_ends",
    too_short: TooShort = "null",
) -> pl.Expr:
    """Removes coincident points in direct succession, leaving only one.
    If `n` repeating points are all at the same place, all but the first are removed,
    except at the end of a linestring or ring.
    Does not remove duplicates that are not directly after each other (see XXXXX)

    The last vertex of a linestring or ring is always kept, so a ring stays closed.
    In such a case any repeating point before that is removed rather than the very last point
    to ensure exact coincidence (a prerequisite for GeoArrow validity),
    unless that one is the first vertex (see `collapsed_lines` below).

    | geometry | removed |
    | - | - |
    | `PointXY[ZM]` | nothing, returns the input |
    | `LineStringXY[ZM]`, `MultiPointXY[ZM]` | coincidence in direct succeccesion |
    | `PolygonXY[ZM]` | the same, per ring |
    | `MultiLineStringXY[ZM]`, `MultiPolygonXY[ZM]` | the same, per part |

    Each vertex is compared with the first vertex in a coinciding set, not the one directly in front of it.
    This means that a group of vertices that slowly drifts will at some point be cut off.

    {coincidence_axes}
    {coincidence_tolerance}
    {tolerance_range}

    `collapsed_lines` says what a linestring keeps when all its vertices lie at one place.
    With `"keep_ends"`, it keeps its first and last vertex, as GEOS does.
    With `"single_vertex"`, it keeps only its first.
    This argument is not considered for polygon rings.
    A ring always keeps both ends in order to stay closed (and therefore valid GeoArrow).

    {too_short}

    For a linestring `A A A`, `"keep_ends"` gives `[A, A]`, which is not too short.
    `"single_vertex"` gives `[A]`, which is.

    {ignores_empty_and_invalid}

    ```python
    df.select(geo.remove_repeats_in_succession("route", abs_tol=0.01))
    ```
    """

    validate_one_of("collapsed_lines", collapsed_lines, CollapsedLines)
    validate_one_of("too_short", too_short, TooShort)
    validate_coincidence_tolerances(abs_tol, rel_tol)

    def build(column: pl.Expr, _geometry: GeoArrowType) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name="remove_repeats_in_succession",
            is_elementwise=True,
            kwargs={
                "abs_tol": abs_tol,
                "rel_tol": rel_tol,
                "unique_z": unique_z,
                "unique_m": unique_m,
                "collapsed_lines": collapsed_lines,
                "too_short": too_short,
            },
        )

    return on_geometry(geometry, build)
