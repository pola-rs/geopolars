"""Whether two geometries intersect, forwarded to GEOS (B-tier)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from geopolars._docs import doc
from geopolars.geo._predicate import predicate

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn


@doc(output="result", paired_with="tested against")
def intersects(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    ignore_per_row_errors: bool = False,
    early_finite_check: bool = False,
) -> pl.Expr:
    """Whether two geometries share at least one point, row by row, as a `bool`.

    {broadcast_single_geometry}

    Geometries that only touch, such as two polygons sharing an edge, also intersect.
    This is quicker than finding where they intersect with `intersection`,
    or getting the full DE-9IM matrix with `relate`.
    It is quickest when you pass a single geometry on either side.

    `a` and `b` can be of any two kinds, and only x and y are compared.
    A geometry without parts intersects nothing, so it gives `False`.
    A geometry that is not valid by `is_valid_ogc`, such as a line of zero length,
    gives no meaningful answer.
    It can then even differ between passing a single geometry and a column of them.

    Computed by GEOS, without a tolerance.

    {geos_errors}

    {geos_non_finite}
    GEOS then gives no meaningful answer.
    {early_finite_check}

    {same_crs_pair}

    ```python
    df.filter(geo.intersects("parcel", "zone"))
    ```
    """
    return predicate(
        a,
        b,
        "intersects",
        ignore_per_row_errors=ignore_per_row_errors,
        early_finite_check=early_finite_check,
    )
