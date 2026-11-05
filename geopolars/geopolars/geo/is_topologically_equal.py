"""Whether two geometries are topologically equal, forwarded to GEOS (B-tier)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.geo._dispatch import on_geometry_pair

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


@doc(output="result")
def is_topologically_equal(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    ignore_per_row_errors: bool = False,
    early_finite_check: bool = False,
) -> pl.Expr:
    """Whether two geometries cover the same points in x and y, row by row, as a `bool`.

    `a` and `b` can be any two kinds and dimensions.
    Both have to declare the same CRS, or neither one.
    Anything else raises at plan time.

    Equal lengths are compared row by row.
    A single geometry on either side is compared against every row of the other.
    A null geometry on either side gives a null result.

    Unlike `is_equal`, it does not matter how the coordinates are stored.
    A ring that starts at another vertex or runs the other way is equal,
    as is a line with an extra vertex along one of its segments,
    or a multi-geometry with its parts in another order.
    A polygon is equal to a multipolygon with only that polygon as a part.
    Any two empty geometries are equal, whatever their kinds.

    Computed by GEOS, in x and y only.
    It takes no tolerance.

    {geos_errors}

    {geos_non_finite}
    GEOS then gives no meaningful answer: depending on where the vertex is, it raises,
    or gives `True` or `False`, even for two of the same geometry.
    A point (not a multipoint) is the exception.
    GEOS compares its coordinates as they are, with NaN equal to NaN and an infinity equal to itself.
    {early_finite_check}

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
        a_crs, b_crs = a_dtype._crs(), b_dtype._crs()
        if a_crs != b_crs:
            msg = (
                f"the geometries declare different CRSs, {a_crs} and {b_crs}; "
                "use `to_crs` to bring one onto the other first"
            )
            raise ValueError(msg)
        return register_plugin_function(
            plugin_path=LIB,
            args=[a, b],
            function_name="is_topologically_equal",
            kwargs=kwargs,
            is_elementwise=True,
        )

    return on_geometry_pair(a, b, build)
