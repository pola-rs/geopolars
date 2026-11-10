"""The DE-9IM matrix of two geometries, forwarded to GEOS (B-tier)."""

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


@doc(output="matrix")
def relate(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    ignore_per_row_errors: bool = False,
    early_finite_check: bool = False,
) -> pl.Expr:
    """How two geometries intersect, row by row, as the text of their DE-9IM matrix.
    A single geometry on either side is related to every row of the other.

    The matrix compares the interior, boundary, and exterior of `a` (its rows)
    against those of `b` (its columns).
    Each cell is the dimension of where they intersect:
    `0` for points, `1` for lines, `2` for an area, and `F` where they do not intersect.
    Its 9 cells are written row after row,
    so a polygon `a` that lies inside a larger polygon `b` gives `"2FF1FF212"`.
    Swapping `a` and `b` transposes the matrix.
    The kinds and dimensions do not have to match, and only x and y are compared.
    A geometry without parts relates as an empty point,
    such as `"FFFFFF212"` for an empty `a` against a polygon `b`.

    Computed by GEOS, without a tolerance.

    {geos_errors}

    {geos_non_finite}
    GEOS then gives no meaningful answer.
    {early_finite_check}

    {same_crs_pair}

    ```python
    df.select(geo.relate("parcel", "zone"))
    ```
    """
    kwargs = {
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
            function_name="relate",
            kwargs=kwargs,
            is_elementwise=True,
        )

    return on_geometry_pair(a, b, build)
