"""Whether two geometries are almost the same.

A native kernel (A-tier) compares the coordinates in place, stopping at the first that is not close.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB, validate_coincidence_tolerances
from geopolars.geo._dispatch import on_geometry_pair

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


@doc()
def is_close(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    abs_tol: float = 0.0,
    rel_tol: float = 1e-09,
    nans_equal: bool = False,
    unique_z: bool = True,
    unique_m: bool = False,
) -> pl.Expr:
    """Whether two geometries are almost coincident or the same.
    Every coordinate has to be close to the one at the same position in the other, on every compared axis.
    This operation runs row by row, returning a `bool`.
    This is meant to complement pl.Expr.is_close` with `abs_tol`, `rel_tol` and `nans_equal`.

    | `a` and `b` | out |
    | - | - |
    | `PointXY[ZM]` | `bool` |
    | `LineStringXY[ZM]` | `bool` |
    | `PolygonXY[ZM]` | `bool` |
    | `MultiPointXY[ZM]` | `bool` |
    | `MultiLineStringXY[ZM]` | `bool` |
    | `MultiPolygonXY[ZM]` | `bool` |

    Comparisons between different types, different dimensions, or different CRSes raise at plan time.
    The coordinates are compared in the order they are stored,
    so a ring that starts at another vertex, or runs the other way, is not close.
    The same goes for the parts of a multi-geometry, even though they have no order of their own.

    {coincidence_axes}
    A tolerance has to be finite and non-negative.
    The empty point (NaN in x and y) is only close to another one with `nans_equal=True`.

    Equal lengths are compared row by row.
    A single geometry on either side is compared against every row of the other.
    A null geometry on either side gives a null result.

    ```python
    df.filter(geo.is_close("parcel", "surveyed", abs_tol=0.01))
    ```
    """
    validate_coincidence_tolerances(abs_tol, rel_tol)

    def build(
        a: pl.Expr, a_dtype: GeoArrowType, b: pl.Expr, b_dtype: GeoArrowType
    ) -> pl.Expr:
        if type(a_dtype) is not type(b_dtype):
            msg = (
                "is_close expects two geometries of the same type,"
                f" got: {a_dtype!r} and {b_dtype!r}"
            )
            raise TypeError(msg)
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
            function_name="is_close",
            kwargs={
                "abs_tol": abs_tol,
                "rel_tol": rel_tol,
                "nans_equal": nans_equal,
                "unique_z": unique_z,
                "unique_m": unique_m,
            },
            is_elementwise=True,
        )

    return on_geometry_pair(a, b, build)
