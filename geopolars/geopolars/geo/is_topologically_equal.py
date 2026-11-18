"""Whether two geometries are topologically equal, forwarded to GEOS (B-tier)."""

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


@doc(output="result", paired_with="compared against")
def is_topologically_equal(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    on_row_error: OnRowError = "raise",
    early_coord_check: EarlyCoordCheck = "finite_and_empty",
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
    {on_row_error}

    {geos_non_finite}
    GEOS then gives no meaningful answer.
    Depending on where the vertex is, it raises or gives `True` or `False`,
    even for two of the same geometry.
    {early_coord_check}

    {same_crs_pair}

    ```python
    df.filter(geo.is_topologically_equal("parcel", "surveyed"))
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
            function_name="is_topologically_equal",
            kwargs=kwargs,
            is_elementwise=True,
        )

    return on_geometry_pair(a, b, build)
