"""Whether two geometries are equal, coordinate by coordinate.

A native kernel (A-tier) compares the coordinates in place,
stopping at the first that is not equal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB, validate_coincidence_tolerances
from geopolars.geo._dispatch import ensure_same_crs, on_geometry_pair

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


@doc()
def is_equal(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    abs_tol: float | None = None,
    rel_tol: float | None = None,
    nans_equal: bool = False,
    unique_z: bool = True,
    unique_m: bool = False,
) -> pl.Expr:
    """Whether two geometries are equal, row by row, as a `bool`.
    A single geometry on either side is compared against every row of the other.

    Every coordinate has to equal the one at the same position in the other,
    so a ring that starts at another vertex, or runs the other way, is not equal.
    The same goes for the parts of a multi-geometry, even though they have no order of their own.
    See `is_topologically_equal` to compare geometries however they are stored.

    {coincidence_axes}
    {coincidence_tolerance}
    {tolerance_range}

    A NaN is equal to nothing, as in `pl.Expr.is_close`.
    So the empty point (NaN in x and y) is not even equal to itself,
    while two empty geometries of any other kind are equal.
    `nans_equal=True` makes a NaN equal to a NaN, as in `==` in Polars.

    `a` and `b` have to be the same kind and dimension.
    Both have to declare the same CRS, or neither one.
    If they are not the same (or only one has a CRS, and the other does not), an error is raised at plan time.

    ```python
    df.filter(geo.is_equal("parcel", "surveyed", abs_tol=0.01))
    ```
    """
    validate_coincidence_tolerances(abs_tol, rel_tol)
    kwargs = {
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "nans_equal": nans_equal,
        "unique_z": unique_z,
        "unique_m": unique_m,
    }

    def build(
        a: pl.Expr, a_dtype: GeoArrowType, b: pl.Expr, b_dtype: GeoArrowType
    ) -> pl.Expr:
        if type(a_dtype) is not type(b_dtype):
            msg = (
                "is_equal expects two geometries of the same type,"
                f" got: {a_dtype!r} and {b_dtype!r}"
            )
            raise TypeError(msg)
        ensure_same_crs(a_dtype, b_dtype)
        return register_plugin_function(
            plugin_path=LIB,
            args=[a, b],
            function_name="is_equal",
            kwargs=kwargs,
            is_elementwise=True,
        )

    return on_geometry_pair(a, b, build)
