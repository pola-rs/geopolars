"""The bounding box of a geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def _bounds(
    column: pl.Expr,
    geometry: GeoArrowType,
    margin_x: float,
    margin_y: float,
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="bounds",
        kwargs={
            "margin_x": margin_x,
            "margin_y": margin_y,
            "turn": geometry._longitude_turn(),
        },
        is_elementwise=True,
    )


@doc()
def bounds(
    geometry: IntoExprColumn, *, margin_x: float = 0.0, margin_y: float = 0.0
) -> pl.Expr:
    """The bounding box of a geometry, as a `geoarrow.box` with its dimension and CRS.
    A multi-geometry gets one box around all of its parts.

    A point's box repeats its own coordinates: `xmin == xmax`.
    `margin_x` and `margin_y` widen the box by that much on both sides
    (a negative margin shrinks it); z and m get no margin.
    The margin is added before `x` is brought back around the globe.

    {box_of_coordinates}
    {box_around_the_globe}
    {empty_box}
    A null geometry gives a null box.

    .. code-block:: python

        df.select(geo.bounds("parcel", margin_x=10.0, margin_y=10.0))
    """
    return on_geometry(
        geometry,
        lambda column, dtype: _bounds(column, dtype, margin_x, margin_y),
    )


@doc()
def total_bounds(geometry: IntoExprColumn, *, ignore_nulls: bool = False) -> pl.Expr:
    """The bounding box of a whole column, as one `geoarrow.box`.
    The one box surrounds every coordinate of every row,
    with all parts of a multi-geometry together.
    For a box per row, see `bounds`.

    {box_of_coordinates}
    {box_around_the_globe}
    The column is brought back around the globe as a whole, not row by row.
    {empty_box}

    A null geometry makes the box null,
    unless `ignore_nulls=True`: then it is skipped.
    A column without rows, or with only null ones, gives a null box.

    .. code-block:: python

        df.group_by("region").agg(geo.total_bounds("parcel"))
    """

    # This function does not stream. Plugins cannot really make streaming things yet.
    # We'd also have to make the kernel itself streaming if we do.

    def build(column: pl.Expr, dtype: GeoArrowType) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name="total_bounds",
            kwargs={"turn": dtype._longitude_turn(), "ignore_nulls": ignore_nulls},
            changes_length=True,
            returns_scalar=True,
        )

    return on_geometry(geometry, build)
