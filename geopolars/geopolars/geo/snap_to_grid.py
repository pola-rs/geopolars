"""Snapping every coordinate onto a grid."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, get_args

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.geo._dispatch import on_geometry
from geopolars.geo.affine import Origin, _origin

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

Spacing = tuple[float, float] | tuple[float, float, float]
RoundMode = Literal["half_to_even", "half_away_from_zero", "to_zero"]


def _spacing(spacing: Spacing) -> tuple[float, float, float]:
    """`spacing` as `(x, y, z)`, a `z` left out being 0."""
    if len(spacing) not in (2, 3) or not all(
        math.isfinite(s) and s >= 0.0 for s in spacing
    ):
        msg = (
            "`spacing` has to be two or three finite numbers of at least 0, "
            f"got: {spacing!r}"
        )
        raise ValueError(msg)
    x, y, *z = spacing
    return x, y, z[0] if z else 0.0


def _snap_to_grid(
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    mode: RoundMode,
) -> Callable[[pl.Expr, GeoArrowType], pl.Expr]:
    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        # Without this, snapping an XY column along z would quietly do nothing.
        if spacing[2] != 0.0 and "z" not in geometry._dimension:
            msg = f"cannot snap z to a grid: {geometry!r} has no z coordinate"
            raise TypeError(msg)
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name="snap_to_grid",
            is_elementwise=True,
            kwargs={
                "spacing": list(spacing),
                "origin": list(origin),
                "mode": mode,
            },
        )

    return build


@doc()
def snap_to_grid(
    geometry: IntoExprColumn,
    spacing: Spacing,
    origin: Origin = (0.0, 0.0, 0.0),
    *,
    mode: RoundMode = "half_to_even",
) -> pl.Expr:
    """Move every position of a geometry onto the nearest point of a grid.

    `spacing` is how far apart the grid lines are, as `(x, y)` or `(x, y, z)`.
    `origin` is a point the grid runs through, as `(x, y)` or `(x, y, z)`,
    by default `(0, 0, 0)`. A `z` left out is 0.
    Each axis is snapped on its own:

    ```text
    x' = origin_x + spacing_x·round((x - origin_x) / spacing_x)
    ```

    and the same for `y` and `z`.
    A spacing of 0 leaves that axis as it is.
    A `z` spacing other than 0 is refused for a geometry with no `z`,
    rather than silently ignored.
    `m`, being a measure, is always carried through untouched.

    `mode` says which grid line a position goes to, as in [Polars `pl.Expr.round`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.round.html),
    but counted in spacings from `origin`:

    - `"half_to_even"`, the default: the nearest one.
      A tie goes to the one an even number of spacings from `origin`, as in PostGIS's `ST_SnapToGrid`.
      Snapping many positions then does not pull them one way on average.
    - `"half_away_from_zero"`: the nearest one. A tie goes to the one further from `origin`.
    - `"to_zero"`: the one between the position and `origin`.

    A NaN stays NaN. {empty_point}
    A ring's ends snap alike, so a closed ring stays closed.
    This does not delete now-overlapping vertices and collapsed geometries (unlike what PostGIS does).
    To get such an effect, you can use it with `remove_repeats_in_succession`
    or TODO<expr to remove all points that coincide>

    Snapping still functions with `f64` accuracy,
    so `0.3` on a grid of `0.1` comes out as `0.30000000000000004`.

    Coordinates are snapped as they are, in the units of their CRS.
    {keeps_type_dimension_and_crs}

    ```python
    df.select(geo.snap_to_grid("shape", (0.5, 0.5)))
    df.select(geo.snap_to_grid("shape", (10.0, 10.0, 1.0), origin=(5.0, 5.0)))
    ```
    """
    if mode not in get_args(RoundMode):
        msg = f"`mode` must be one of {list(get_args(RoundMode))}, got {mode!r}"
        raise ValueError(msg)
    return on_geometry(
        geometry, _snap_to_grid(_spacing(spacing), _origin(origin), mode)
    )
