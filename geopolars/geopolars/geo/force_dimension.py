"""Adding or dropping the `z` and `m` coordinates of a geometry."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl

from geopolars._docs import doc
from geopolars.datatypes.dimension import XYZM
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType
    from geopolars.datatypes.dimension import Dimension


def _filled(values: pl.Expr, layers: int, added: Dimension) -> pl.Expr:
    """Every coordinate with the `added` axes set to `0.0`,
    or to NaN for an empty point (to keep it as empty)."""
    if layers == 0:
        empty = pl.field("x").is_nan() & pl.field("y").is_nan()
        value = pl.when(empty).then(math.nan).otherwise(0.0)
        return values.struct.with_fields(value.alias(axis) for axis in added)
    return values.list.eval(_filled(pl.element(), layers - 1, added))


def _with_axes(
    z: bool | None, m: bool | None
) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    """`z=None` or `m=None` keeps that axis as it is."""
    wanted = {"z": z, "m": m}

    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        current = geometry._dimension

        def is_kept(axis: str) -> bool:
            keep = wanted.get(axis, True)
            return axis in current if keep is None else keep

        dimension = tuple(axis for axis in XYZM if is_kept(axis))
        if dimension == current:
            return column

        target = type(geometry).of_dimension(dimension)._with_metadata_of(geometry)
        # Matches the fields by name: a dropped axis goes, an added one is null.
        storage = column.ext.storage().cast(target._geo_storage, strict=False)
        added = tuple(axis for axis in dimension if axis not in current)
        if added:
            storage = _filled(storage, geometry._nesting, added)
        return storage.ext.to(target)

    return build


@doc()
def force_2d(geometry: IntoExprColumn) -> pl.Expr:
    """The geometry without its `z` coordinate.

    Only `z` is dropped: an `m` is kept.
    The geometry kind, its nesting and its CRS stay the same.

    =======  ======
    in       out
    =======  ======
    `…XY`    `…XY`
    `…XYZ`   `…XY`
    `…XYM`   `…XYM`
    `…XYZM`  `…XYM`
    =======  ======

    A geometry without a `z` is returned as it is.
    {ignores_empty_and_invalid}

    .. code-block:: python

        df.select(geo.force_2d("route"))
    """
    return on_geometry(geometry, _with_axes(z=False, m=None))


def force_3d(geometry: IntoExprColumn) -> pl.Expr:
    """The geometry with a `z` coordinate.

    Works like `force_2d`, but adds a `z` of `0.0` where there was none.
    An empty point (NaN in `x` and `y`) gets a NaN `z` instead, so that it stays empty.
    A geometry that already has a `z` is returned as it is.

    .. code-block:: python

        df.select(geo.force_3d("route"))
    """
    return on_geometry(geometry, _with_axes(z=True, m=None))


def without_m(geometry: IntoExprColumn) -> pl.Expr:
    """The geometry without its `m` (measure) value.

    Works like `force_2d`, but drops `m` instead of `z`.

    .. code-block:: python

        df.select(geo.without_m("route"))
    """
    return on_geometry(geometry, _with_axes(z=None, m=False))


def force_m(geometry: IntoExprColumn) -> pl.Expr:
    """The geometry with an `m` (measure) value.

    Works like `force_3d`, but adds `m` instead of `z`.

    .. code-block:: python

        df.select(geo.force_m("route"))
    """
    return on_geometry(geometry, _with_axes(z=None, m=True))
