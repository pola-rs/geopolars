"""The rectangle a box describes, as a polygon."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from geopolars.datatypes import PolygonType
from geopolars.geo._dispatch import on_box

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import BoxType


def _to_polygon(column: pl.Expr, box: BoxType) -> pl.Expr:
    bounds = column.ext.storage()
    dimension = box._dimension
    low = {axis: bounds.struct.field(f"{axis}min") for axis in dimension}
    high = {axis: bounds.struct.field(f"{axis}max") for axis in dimension}

    # The spec writes an empty range as `inf` to `-inf`, in any dimension.
    # `xmin > xmax` is otherwise a box crossing the antimeridian;
    # the exception does not apply to the other dimensions.
    crosses_antimeridian = low["x"] > high["x"]
    empty_x = crosses_antimeridian & (
        (low["x"] == float("inf")) | (high["x"] == float("-inf"))
    )
    empty = pl.any_horizontal(
        empty_x, *(low[axis] > high[axis] for axis in dimension if axis != "x")
    )

    turn = box._longitude_turn()
    if turn is not None:
        # Continue east past the antimeridian (to 190 rather than -170),
        # so the polygon covers the side the box means.
        high["x"] = (
            pl.when(crosses_antimeridian).then(high["x"] + turn).otherwise(high["x"])
        )

    def vertex(x: dict[str, pl.Expr], y: dict[str, pl.Expr]) -> pl.Expr:
        # z and m follow x: the vertices at xmin take every other minimum,
        # those at xmax every other maximum.
        # Both bounds survive, and the ring stays flat (a tilted plane).
        return pl.struct(
            **{axis: (y if axis == "y" else x)[axis] for axis in dimension}
        )

    # The exterior ring, counter-clockwise and closed.
    corners = [(low, low), (high, low), (high, high), (low, high), (low, low)]
    ring = pl.concat_arr([vertex(x, y) for x, y in corners])
    # Polars has no elementwise way to wrap a list in another list,
    # so a single ring is reshaped into a list of one ring.
    polygon_type = PolygonType.of_dimension(dimension)
    storage = polygon_type._geo_storage
    rings = ring.reshape((-1, 1, len(corners))).cast(storage)

    polygon = (
        pl.when(bounds.is_null()).then(None).when(empty).then(pl.lit([], dtype=storage))
    )
    if turn is None:
        # Without a longitude there is no way around: no polygon describes it.
        polygon = polygon.when(crosses_antimeridian).then(None)
    polygon = polygon.otherwise(rings)
    return polygon.ext.to(polygon_type._with_metadata_of(box))


def to_polygon(box: IntoExprColumn) -> pl.Expr:
    """The rectangle a box describes, as a polygon with the box's dimension and CRS:
    `BoxXY` gives a `PolygonXY`, `BoxXYZ` a `PolygonXYZ`, and so on.

    The polygon has a single, counter-clockwise exterior ring of five vertices,
    starting and ending at `(xmin, ymin)`.
    z and m follow x: the two vertices at `xmin` take `zmin` and `mmin`,
    the two at `xmax` take `zmax` and `mmax`.
    Both ends of every range survive, and the ring stays flat:
    a plane tilting from the `xmin` side up to the `xmax` side.

    =====================================  ===============================
    box                                    polygon
    =====================================  ===============================
    null                                   null
    an empty range in any dimension        empty (no rings)
    `xmin > xmax`, geographic CRS          continues past the antimeridian
    `xmin > xmax`, any other CRS, or none  null
    =====================================  ===============================

    A box from 170 to -170 crosses the antimeridian.
    With a geographic CRS, its polygon runs from 170 to 190 instead:
    one full turn (360 degrees, or 400 grads) is added to `xmax`,
    so the polygon covers the same side of the globe as the box.
    Subtract the turn again to bring a vertex back into range.
    Planar operations on such a polygon only see it as running to 190,
    so a point at -175 is not inside it to them.
    Without a geographic CRS there is no way around, and no polygon describes the box.

    An empty range is `inf` to `-inf`, or any other reversed range outside of x.

    .. code-block:: python

        df.select(geo.to_polygon("extent"))
    """
    return on_box(box, _to_polygon)
