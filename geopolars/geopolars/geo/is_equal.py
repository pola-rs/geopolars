"""Whether two geometries are equal.

With `mode="identical"`, a native kernel (A-tier) compares the coordinates in place,
stopping at the first that is not equal.
With `mode="topological"`, the comparison is forwarded to GEOS (B-tier).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB, validate_coincidence_tolerances
from geopolars.geo._dispatch import on_geometry_pair
from geopolars.geo.buffer import _check_one_of

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

_MODES = ("identical", "topological")


@doc(output="result")
def is_equal(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    mode: Literal["identical", "topological"] = "identical",
    abs_tol: float | None = None,
    rel_tol: float | None = None,
    nans_equal: bool = True,
    unique_z: bool = True,
    unique_m: bool = False,
    ignore_per_row_errors: bool = False,
    early_finite_check: bool = False,
) -> pl.Expr:
    """Whether two geometries are equal, row by row, as a `bool`.

    `mode` says what counts as equal:

    | `mode` | equal when | `a` and `b` |
    | - | - | - |
    | `"identical"` (default) | every coordinate is equal (or close) to the one at the same position in the other | the same kind and dimension |
    | `"topological"` | they cover the same points in x and y | any two kinds and dimensions |

    Both have to declare the same CRS, or neither one.
    Anything else raises at plan time.

    Equal lengths are compared row by row.
    A single geometry on either side is compared against every row of the other.
    A null geometry on either side gives a null result.

    **Identical**

    The coordinates are compared in the order they are stored,
    so a ring that starts at another vertex, or runs the other way, is not equal.
    The same goes for the parts of a multi-geometry, even though they have no order of their own.

    {coincidence_axes}
    {coincidence_tolerance}
    {tolerance_range}
    With `nans_equal=True`, the default, a NaN is equal to a NaN, as in `==` in Polars.
    So the empty point (NaN in x and y) is equal to another one.
    With `nans_equal=False`, a NaN is equal to nothing, as in `pl.Expr.is_close` by default.

    **Topological**

    Two geometries are equal when they cover the same points,
    however their coordinates are stored.
    A ring that starts at another vertex or runs the other way is equal,
    as is a line with an extra vertex along one of its segments,
    or a multi-geometry with its parts in another order.
    A polygon is equal to a multipolygon with only that polygon as a part.
    Any two empty geometries are equal, whatever their kinds.

    Computed by GEOS, in x and y only.
    It takes no tolerance, and `nans_equal`, `unique_z`, and `unique_m` have to keep their defaults.

    {geos_errors}

    {geos_non_finite}
    GEOS then gives no meaningful answer: depending on where the vertex is, it raises,
    or gives `True` or `False`, even for two of the same geometry.
    A point (not a multipoint) is the exception.
    GEOS compares its coordinates as they are, with NaN equal to NaN and an infinity equal to itself.
    {early_finite_check}

    ```python
    df.filter(geo.is_equal("parcel", "surveyed", abs_tol=0.01))
    df.filter(geo.is_equal("parcel", "surveyed", mode="topological"))
    ```
    """
    _check_one_of("is_equal", "mode", mode, _MODES)
    validate_coincidence_tolerances(abs_tol, rel_tol)
    if mode == "topological":
        _check_topological_options(abs_tol, rel_tol, nans_equal, unique_z, unique_m)
        function_name = "is_equal_topological"
        kwargs: dict[str, float | bool | None] = {
            "ignore_per_row_errors": ignore_per_row_errors,
            "early_finite_check": early_finite_check,
        }
    else:
        _check_identical_options(ignore_per_row_errors, early_finite_check)
        function_name = "is_equal_identical"
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
        if mode == "identical" and type(a_dtype) is not type(b_dtype):
            msg = (
                'is_equal with mode="identical" expects two geometries of the same type,'
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
            function_name=function_name,
            kwargs=kwargs,
            is_elementwise=True,
        )

    return on_geometry_pair(a, b, build)


def _check_topological_options(
    abs_tol: float | None,
    rel_tol: float | None,
    nans_equal: bool,
    unique_z: bool,
    unique_m: bool,
) -> None:
    if abs_tol is not None or rel_tol is not None:
        msg = (
            'is_equal with mode="topological" takes no `abs_tol` or `rel_tol`:'
            " GEOS's topological equality does not support a tolerance"
        )
        raise ValueError(msg)
    changed = {
        "nans_equal=False": (not nans_equal, "GEOS has no option for how NaNs compare"),
        "unique_z=False": (not unique_z, "GEOS compares x and y only"),
        "unique_m=True": (unique_m, "GEOS compares x and y only"),
    }
    for option, (is_given, reason) in changed.items():
        if is_given:
            msg = f'is_equal with mode="topological" takes no `{option}`: {reason}'
            raise ValueError(msg)


def _check_identical_options(
    ignore_per_row_errors: bool, early_finite_check: bool
) -> None:
    given = {
        "ignore_per_row_errors": ignore_per_row_errors,
        "early_finite_check": early_finite_check,
    }
    for name, value in given.items():
        if value:
            msg = (
                f'is_equal with mode="identical" takes no `{name}`:'
                ' it only applies to mode="topological", which goes through GEOS'
            )
            raise ValueError(msg)
