"""Building boxes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from geopolars.datatypes import BoxType
from geopolars.geo.construct._common import _coord, _given

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn


def box(
    xmin: IntoExprColumn,
    ymin: IntoExprColumn,
    xmax: IntoExprColumn,
    ymax: IntoExprColumn,
    *,
    zmin: IntoExprColumn | None = None,
    zmax: IntoExprColumn | None = None,
    mmin: IntoExprColumn | None = None,
    mmax: IntoExprColumn | None = None,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.box` column from its bounds.
    Whether you pass the `z` and/or `m` bounds decides the dimension, and so the dtype:

    | passed                         | dtype     |
    |--------------------------------|-----------|
    | -                              | `BoxXY`   |
    | `zmin`, `zmax`                 | `BoxXYZ`  |
    | `mmin`, `mmax`                 | `BoxXYM`  |
    | `zmin`, `zmax`, `mmin`, `mmax` | `BoxXYZM` |

    The bounds are taken as given: `xmin > xmax` is a box that crosses the antimeridian,
    and a range from `inf` to `-inf` is empty.
    A null bound makes the whole box null.

    ```python
    df.select(geo.box("west", "south", "east", "north", crs="EPSG:4326"))
    ```
    """
    given = _given(xmin, ymin, zmin, mmin)
    for axis, low, high in (("z", zmin, zmax), ("m", mmin, mmax)):
        if (low is None) != (high is None):
            msg = f"pass both {axis}min and {axis}max, or neither"
            raise ValueError(msg)
    dimension = tuple(given)
    maxima = _given(xmax, ymax, zmax, mmax)
    bounds = [_coord(value, f"{axis}min") for axis, value in given.items()] + [
        _coord(value, f"{axis}max") for axis, value in maxima.items()
    ]

    # A null bound nulls the entire box.
    complete = pl.all_horizontal([bound.is_not_null() for bound in bounds])
    dtype = BoxType.of_dimension(dimension)
    return pl.when(complete).then(pl.struct(bounds)).ext.to(dtype(crs=crs))
