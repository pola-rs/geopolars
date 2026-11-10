"""The difference of two geometries, forwarded to GEOS (B-tier)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.geo._dispatch import ensure_same_crs, on_geometry_pair

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


@doc(output="geometry", paired_with="paired with")
def difference(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    allow_splitting: bool = False,
    ccw_outer: bool = False,
    ignore_per_row_errors: bool = False,
    early_finite_check: bool = False,
) -> pl.Expr:
    """The geometry you get when you subtract `b` from `a`.

    Taking the difference is only really meaningful if the two geometries have the same
    *dimensionality* (as in a line is 1d, a polygon 2d, not whether or not it has z/m).
    A polygon minus a line or a point covers the same area as before,
    although GEOS may add a vertex where the line crosses its boundary.
    Likewise, a line minus a point covers the same length as before.

    The result is the kind of `a`, and of `a` alone.
    For a LineString or Polygon, the result can split.
    {overlay_splitting}

    | a | output kind |
    | - | - |
    | Point | Point |
    | MultiPoint | MultiPoint |
    | LineString | LineString, or MultiLineString with `allow_splitting` |
    | MultiLineString | MultiLineString |
    | Polygon | Polygon, or MultiPolygon with `allow_splitting` |
    | MultiPolygon | MultiPolygon |

    {broadcast_single_geometry}

    There are some gotcha's with linestrings:
    A line minus a line that crosses it only gains a vertex where the two cross.
    GEOS also cuts a line where it crosses itself, and a MultiLineString where its parts cross.
    A LineString that crosses itself therefore splits whenever it goes through GEOS,
    even if `b` does not touch it.
    Only an empty `b` leaves it whole (see below).

    {computed_by_geos_keeping_z}
    {overlay_z}

    {null_on_either_side}
    An empty `a` gives an empty {output}, without going through GEOS.
    An empty `b` takes nothing away, and gives `a` back as it is, also without going through GEOS.
    The one exception is a polygon part of `a` whose rings have no vertices.
    It is stored without rings, like every other empty polygon.
    A `b` that covers all of `a` takes everything away, so it gives an empty {output} too.

    {geos_errors}
    The same goes for a result in several parts without `allow_splitting`.

    {geos_non_finite}
    {early_finite_check}
    A row where `b` is empty does not go to GEOS,
    but raises on such a vertex in `a` whatever `early_finite_check` is
    (or gives a null {output} with `ignore_per_row_errors=True`).
    GEOS itself does not raise on every such vertex,
    so this row can be stricter than one that GEOS computes.

    {ccw_outer}
    It only affects a Polygon or MultiPolygon result.
    {ccw_outer_skipping_geos}
    Here, those are the rows where `b` is empty.

    {same_crs_pair}
    {overlay_crs}

    ```python
    df.select(geo.difference("parcel", "road_reserve"))
    df.select(geo.difference("trail", "private_land", allow_splitting=True))
    ```
    """
    kwargs = {
        "allow_splitting": allow_splitting,
        "ccw_outer": ccw_outer,
        "ignore_per_row_errors": ignore_per_row_errors,
        "early_finite_check": early_finite_check,
    }

    def build(
        a: pl.Expr, a_dtype: GeoArrowType, b: pl.Expr, b_dtype: GeoArrowType
    ) -> pl.Expr:
        ensure_same_crs(a_dtype, b_dtype)
        return register_plugin_function(
            plugin_path=LIB,
            args=[a, b],
            function_name="difference",
            kwargs=kwargs,
            is_elementwise=True,
        )

    return on_geometry_pair(a, b, build)
