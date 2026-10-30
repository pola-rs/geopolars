"""The parts of a multi-geometry, as a list.

A multi-geometry's storage already is a list of its parts' storage,
so this is C-tier: only the type changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from geopolars.datatypes import (
    LineStringType,
    MultiLineStringType,
    MultiPointType,
    MultiPolygonType,
    PointType,
    PolygonType,
)
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

_PART_OF: dict[type[GeoArrowType], type[GeoArrowType]] = {
    MultiPointType: PointType,
    MultiLineStringType: LineStringType,
    MultiPolygonType: PolygonType,
}


def _parts(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
    part_type = next(
        (part for multi, part in _PART_OF.items() if isinstance(geometry, multi)),
        None,
    )
    if part_type is None:
        msg = (
            "parts expects a multipoint, multilinestring or multipolygon column, "
            f"got: {geometry!r}"
        )
        raise TypeError(msg)
    part = part_type.of_dimension(geometry._dimension)._with_metadata_of(geometry)
    # Polars can't cast to a list of an extension type, so relabel inside
    # `list.eval`. Its body only changes the type, so this costs nothing.
    return column.ext.storage().list.eval(pl.element().ext.to(part))


def parts(geometry: IntoExprColumn) -> pl.Expr:
    """Breaks down a multi-geometry into its parts, as a list in stored order.
    The parts keep the multi-geometry's dimension and CRS.

    | in | out |
    | - | - |
    | `PointXY[ZM]` | refused |
    | `LineStringXY[ZM]` | refused |
    | `PolygonXY[ZM]` | refused |
    | `MultiPointXY[ZM]` | `list[PointXY[ZM]]` |
    | `MultiLineStringXY[ZM]` | `list[LineStringXY[ZM]]` |
    | `MultiPolygonXY[ZM]` | `list[PolygonXY[ZM]]` |

    An empty multi-geometry gives an empty list.
    A null multi-geometry gives a null list.

    ```python
    df.select(geo.parts("river"))
    ```
    """
    return on_geometry(geometry, _parts)
