"""The DE-9IM matrix of two geometries, forwarded to GEOS (B-tier)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._typing import EarlyCoordCheck, OnRowError
from geopolars._utils import LIB, validate_one_of
from geopolars.geo._dispatch import ensure_same_crs, on_geometry_pair

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


@doc(output="result", paired_with="related to")
def relate(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    on_row_error: OnRowError = "raise",
    early_coord_check: EarlyCoordCheck = "finite_and_empty",
) -> pl.Expr:
    """Gets the DE-9IM matrix in a text format. This tells you how two geometries intersect.

    {broadcast_single_geometry}

    OGC Simple Features defines the matrix
    (<https://docs.ogc.org/is/06-103r4/06-103r4.pdf>, section 6.1.15.2).
    In short, its rows stand for the interior, boundary, and exterior of `a`,
    and its columns for those of `b`.
    Swapping `a` and `b` therefore transposes the matrix.
    Each cell holds the highest dimension of where they intersect (`0`, `1`, or `2`),
    or `F` if they do not.
    The text lists the 9 cells row by row.
    For example, a polygon `a` that lies inside a polygon `b`, without touching its boundary,
    gives `"2FF1FF212"`.

    `a` and `b` can be of any two kinds, and only x and y are compared.
    A geometry without parts relates as an empty point,
    such as `"FFFFFF212"` for an empty `a` against a polygon `b`.

    Computed by GEOS, without a tolerance.

    {geos_errors}
    {on_row_error}

    {geos_non_finite}
    GEOS then gives no meaningful answer.
    {early_coord_check}

    {same_crs_pair}

    ```python
    df.select(geo.relate("parcel", "zone"))
    ```
    """
    validate_one_of("on_row_error", on_row_error, OnRowError)
    validate_one_of("early_coord_check", early_coord_check, EarlyCoordCheck)
    kwargs = {
        "on_row_error": on_row_error,
        "early_coord_check": early_coord_check,
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
