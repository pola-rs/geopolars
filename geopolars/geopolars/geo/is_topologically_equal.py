"""Whether two geometries are topologically equal, forwarded to GEOS (B-tier)."""

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


@doc(output="result", paired_with="compared against")
def is_topologically_equal(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    ignore_per_row_errors: bool = False,
    early_finite_check: bool = False,
) -> pl.Expr:
    """Whether two geometries cover the same points in x and y, row by row, as a `bool`.
    {broadcast_single_geometry}

    Unlike `is_equal`, it does not matter how the coordinates are stored.
    A ring that starts at another vertex or runs the other way is equal.
    So is a line with an extra vertex along one of its segments,
    and a multi-geometry with its parts in another order.
    The kinds and dimensions do not have to match either.
    A polygon is equal to a multipolygon with only that polygon as a part,
    and any two empty geometries are equal.

    Computed by GEOS, without a tolerance.

    {geos_errors}

    {geos_non_finite}
    GEOS then gives no meaningful answer.
    Depending on where the vertex is, it raises or gives `True` or `False`,
    even for two of the same geometry.
    {early_finite_check}

    {same_crs_pair}

    ```python
    df.filter(geo.is_topologically_equal("parcel", "surveyed"))
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
            function_name="is_topologically_equal",
            kwargs=kwargs,
            is_elementwise=True,
        )

    return on_geometry_pair(a, b, build)
