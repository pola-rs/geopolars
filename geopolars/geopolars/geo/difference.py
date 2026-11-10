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


@doc(output="geometry")
def difference(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    allow_splitting: bool = False,
    ccw_outer: bool = False,
    ignore_per_row_errors: bool = False,
    early_finite_check: bool = False,
) -> pl.Expr:
    """What is left of `a` once `b` is taken away, row by row.
    A single geometry on either side is paired with every row of the other.

    The result is always of the dimension of `a`, whatever `b` is.
    Only an area can take something away from a polygon:
    a polygon minus a line or a point covers the same area as before,
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

    A line minus a line that crosses it only gains a vertex where they cross.
    A line that crosses itself, or a MultiLineString whose parts cross, is cut there, though.
    A LineString therefore splits even if a non-empty `b` does not touch it.

    {computed_by_geos_keeping_z}
    {overlay_z}

    A null geometry on either side gives a null {output}, and an empty `a` an empty {output}.
    So does taking everything away.
    A Point result is then the empty point (NaN on every axis).
    An empty `b` takes nothing away, so its row gives `a` as it is, without computing a difference.
    It keeps its vertices and ring orientation, and never splits.
    If only `b` has z, the z of such a row is NaN.

    {geos_errors}
    The same goes for a result in several parts without `allow_splitting`.

    {geos_non_finite}
    {early_finite_check}
    A row where `b` is empty fails on such a vertex in `a` too,
    although it is not computed by GEOS.

    {ccw_outer}
    It only affects a Polygon or MultiPolygon result,
    and not the rows where `b` is empty.

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
