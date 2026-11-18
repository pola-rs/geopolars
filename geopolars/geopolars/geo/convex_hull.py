"""The convex hull of a linestring or polygon, or their multi-variants.

Forwarded to GEOS (B-tier).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._typing import (
    EarlyCoordCheck,
    NewHoles,
    NewRingOrientation,
    OnRowError,
)
from geopolars._utils import LIB, validate_one_of
from geopolars.datatypes import (
    LineStringType,
    MultiLineStringType,
    MultiPointType,
    MultiPolygonType,
    PolygonType,
)
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

_ACCEPTED = (
    LineStringType,
    MultiPointType,
    MultiLineStringType,
    PolygonType,
    MultiPolygonType,
)


@doc(output="polygon")
def convex_hull(
    geometry: IntoExprColumn,
    *,
    on_row_error: OnRowError = "raise",
    new_ring_orientation: NewRingOrientation = "keep",
    new_holes: NewHoles = "opposite",
    early_coord_check: EarlyCoordCheck = "finite_and_empty",
) -> pl.Expr:
    """The smallest convex polygon that contains every vertex of a linestring or polygon,
    with its CRS.
    A multipoint, multilinestring or multipolygon gives a single polygon:
    the hull of the geometry as a whole, not one per part.
    Only the exterior of a polygon counts, so its holes are ignored.
    An XY or XYM geometry gives a `PolygonXY`, an XYZ or XYZM one a `PolygonXYZ`:

    | in | out |
    | - | - |
    | `PointXY[ZM]` | refused |
    | `LineStringXY[M]` | `PolygonXY` |
    | `LineStringXYZ[M]` | `PolygonXYZ` |
    | `PolygonXY[M]` | `PolygonXY` |
    | `PolygonXYZ[M]` | `PolygonXYZ` |
    | `MultiPointXY[M]` | `PolygonXY` |
    | `MultiPointXYZ[M]` | `PolygonXYZ` |
    | `MultiLineStringXY[M]` | `PolygonXY` |
    | `MultiLineStringXYZ[M]` | `PolygonXYZ` |
    | `MultiPolygonXY[M]` | `PolygonXY` |
    | `MultiPolygonXYZ[M]` | `PolygonXYZ` |

    {computed_by_geos_keeping_z}
    {empty_and_null}
    One that encloses nothing (all its vertices on one point or one line,
    such as a straight linestring, or a multipoint of one or two points)
    has a point or a line for a hull,
    so it gives a null polygon.

    CRS is ignored:
    for longitude/latitude the hull's edges are straight in degrees, not along the globe.
    A geometry crossing the antimeridian is hulled the long way round.

    {geos_errors}
    {on_row_error}

    {geos_non_finite}
    Depending on where such a vertex sits, it is left out of the hull,
    the polygon is null, or GEOS raises.
    {early_coord_check}
    A geometry with a single coordinate encloses nothing,
    so it gives a null polygon. (This bypasses finiteness chcking)

    {new_ring_orientation}
    A hull has no holes, so `new_holes` changes nothing.

    ```python
    df.select(geo.convex_hull("parcel"))
    ```
    """
    validate_one_of("on_row_error", on_row_error, OnRowError)
    validate_one_of("early_coord_check", early_coord_check, EarlyCoordCheck)
    validate_one_of("new_ring_orientation", new_ring_orientation, NewRingOrientation)
    validate_one_of("new_holes", new_holes, NewHoles)

    def build(column: pl.Expr, dtype: GeoArrowType) -> pl.Expr:
        if not isinstance(dtype, _ACCEPTED):
            msg = (
                "convex_hull expects a linestring, polygon, multipoint, multilinestring"
                f" or multipolygon column, got: {dtype!r}"
            )
            raise TypeError(msg)
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name="convex_hull",
            kwargs={
                "on_row_error": on_row_error,
                "new_ring_orientation": new_ring_orientation,
                "new_holes": new_holes,
                "early_coord_check": early_coord_check,
            },
            is_elementwise=True,
        )

    return on_geometry(geometry, build)
