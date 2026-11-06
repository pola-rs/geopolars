"""Choosing an expression by the dtype of the column it is given.

Which geometry a column holds is in its dtype,
and every operation has to be written differently per geometry.
Using `Expr.pipe_with_dtype` we can switch dynamically based on the exact dtype
to alter the plan that polars finally generates.
`build` is called once, with the one concrete geometry type,
and gives back a plain expression for just that geometry.
It works the same for a column, an expression or a Series.

This allows us to raise errors on a type level at plan time,
(which includes calling these functions on 'unknown' geometries or other polars data types)
rather than at execution time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl

from geopolars.datatypes import GEOMETRY_DTYPES, BoxType, GeoArrowType

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn

# analog of `Kind::ALL` x `Dimension::ALL` on the Rust side.
GEOMETRY_TYPES: tuple[type[GeoArrowType], ...] = tuple(
    concrete for geometry in GEOMETRY_DTYPES for concrete in geometry.dimensions()
)


def _names() -> str:
    """The extension names, for an error message. Mirrors `Kind::names`."""
    names = [f"`{geometry._extension_name}`" for geometry in GEOMETRY_DTYPES]
    return f"{', '.join(names[:-1])} or {names[-1]}"


UNSUPPORTED = f"expected a {_names()} column with separated x/y[/z][/m] coordinates"


def _geometry_of(dtype: pl.DataType) -> GeoArrowType:
    """`dtype` as the concrete geometry it is, metadata and all."""
    if type(dtype) not in GEOMETRY_TYPES:
        msg = f"{UNSUPPORTED}, got: {dtype!r}"
        raise TypeError(msg)
    return dtype  # type: ignore[return-value]


def _box_of(dtype: pl.DataType) -> BoxType:
    """`dtype` as the concrete box"""
    if not isinstance(dtype, BoxType):
        msg = f"expected a `{BoxType._extension_name}` column, got: {dtype!r}"
        raise TypeError(msg)
    return dtype


def _to_expr(value: IntoExprColumn) -> pl.Expr:
    if isinstance(value, str):
        return pl.col(value)
    if isinstance(value, pl.Series):
        return pl.lit(value)
    return value


def _named_after(built: pl.Expr, source: pl.Expr) -> pl.Expr:
    """With this, we can pass a List[T] where we would otherwise use a T,
    and get a List[result] back.
    """
    name = source.meta.output_name(raise_if_undetermined=False)
    return built if name is None else built.alias(name)


def on_geometry(
    value: IntoExprColumn,
    build: Callable[[pl.Expr, GeoArrowType], pl.Expr],
) -> pl.Expr:
    """Wrapper for callbacks for pipe_with_dtype.

    `build` gets the dtype itself rather than its class, metadata included.
    A geometry it gives back has to carry that metadata through
    (see `GeoArrowType._with_metadata_of`), or the CRS is lost.
    """
    expr = _to_expr(value)

    def resolved(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
        return _named_after(build(expr, _geometry_of(dtype)), expr)

    return expr.pipe_with_dtype(resolved)


def on_geometry_pair(
    first: IntoExprColumn,
    second: IntoExprColumn,
    build: Callable[[pl.Expr, GeoArrowType, pl.Expr, GeoArrowType], pl.Expr],
) -> pl.Expr:
    """`on_geometry` for an operation between two geometries.
    The result is named after `first`, like a binary operator in Polars.
    """

    def resolved_first(first: pl.Expr, first_dtype: pl.DataType) -> pl.Expr:
        first_geometry = _geometry_of(first_dtype)

        def resolved_second(second: pl.Expr, second_dtype: pl.DataType) -> pl.Expr:
            built = build(first, first_geometry, second, _geometry_of(second_dtype))
            return _named_after(built, first)

        return _to_expr(second).pipe_with_dtype(resolved_second)

    return _to_expr(first).pipe_with_dtype(resolved_first)


def ensure_same_crs(
    a_dtype: GeoArrowType, b_dtype: GeoArrowType, *, described_as: str = "geometries"
) -> None:
    """Raises unless `a_dtype` and `b_dtype` declare the same CRS, or neither declares one."""
    a_crs, b_crs = a_dtype._crs(), b_dtype._crs()
    if (a_crs is None) != (b_crs is None):
        msg = f"only one of the {described_as} declares a CRS; declare the same one on both"
        raise ValueError(msg)
    if a_crs != b_crs:
        msg = (
            f"the {described_as} declare different CRSs, {a_crs} and {b_crs}; "
            "use `to_crs` to bring one onto the other first"
        )
        raise ValueError(msg)


def on_box(
    value: IntoExprColumn,
    build: Callable[[pl.Expr, BoxType], pl.Expr],
) -> pl.Expr:
    """`on_geometry` for an operation on a `geoarrow.box`,
    which is not one of the geometries `on_geometry` accepts.
    """
    expr = _to_expr(value)

    def resolved(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
        return _named_after(build(expr, _box_of(dtype)), expr)

    return expr.pipe_with_dtype(resolved)
