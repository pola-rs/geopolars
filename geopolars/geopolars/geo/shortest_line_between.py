"""The shortest line between two geometries, forwarded to GEOS (B-tier)."""

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


@doc(output="line")
def shortest_line_between(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    ignore_per_row_errors: bool = False,
    early_finite_check: bool = False,
) -> pl.Expr:
    """The shortest line between two geometries, row by row, as a `LineStringXY`.
    It runs from the point on `a` closest to `b`, to the point on `b` closest to `a`.
    If the geometries touch or overlap, both vertices are the same point.
    The length of the line is the distance between the two geometries.
    A single geometry on either side is measured against every row of the other.

    Any kind and dimension can be on either side.
    Computed by GEOS in x and y only, so the line has no z or m.
    It is measured in the plane of the coordinates,
    so for a geographic CRS it is not the shortest line on the ellipsoid.

    If either geometry is empty, so is the line.

    {geos_errors}

    {geos_non_finite}
    GEOS then gives no meaningful answer.
    Depending on where the vertex is, it raises or might give a line that is not the shortest.
    {early_finite_check}

    `a` and `b` have to declare the same CRS, or neither one.
    If they are not the same (or only one has a CRS, and the other does not), an error is raised at plan time.

    ```python
    df.with_columns(geo.shortest_line_between("house", "road").alias("driveway"))
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
            function_name="shortest_line_between",
            kwargs=kwargs,
            is_elementwise=True,
        )

    return on_geometry_pair(a, b, build)
