"""Whether a geometry is simple: it does not cross or touch itself.

Points and multipoints are A-tier, the others go to GEOS (B-tier).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def is_simple(geometry: IntoExprColumn) -> pl.Expr:
    """Whether each geometry is simple by OGC Simple Features, as GEOS's (and shapely's) `is_simple` has it.
    One `bool` per row, a multi-geometry included. A null geometry gives a null result.

    A simple geometry does not cross or touch itself
    (https://docs.ogc.org/is/06-103r4/06-103r4.pdf, sections 6.1.4 to 6.1.8).
    The geometries have to be valid by `is_valid_geoarrow`, but not necessarily by OGC (see `is_valid_ogc`).

    =======================  ============================================================================
    geometry                 simple when
    =======================  ============================================================================
    `PointXY[ZM]`            always
    `MultiPointXY[ZM]`       no two points lie at exactly the same place in x and y
    `LineStringXY[ZM]`       it does not pass through the same place twice, except that its ends may meet
    `MultiLineStringXY[ZM]`  every part is simple, and two parts only meet where both have an end
    `PolygonXY[ZM]`          every ring is simple, as a linestring with its ends meeting
    `MultiPolygonXY[ZM]`     every ring of every part is simple
    =======================  ============================================================================

    Going back and forth between two points counts as passing through the same place twice.
    Repeated vertices do not count, as long as they are exactly equal.
    An empty geometry is simple, and so is a single segment.
    A linestring of 1 vertex is not a line, so it is not simple.
    Neither is a ring of 1 to 3 vertices, which can only go back and forth.

    A closed part of a multilinestring has no ends, so no other part may touch it.

    How the rings of a polygon, or the parts of a multipolygon, fit together is not looked at.
    See `is_valid_ogc` for that.

    This looks at a projected plane.
    Only x and y are looked at. z and m are ignored.
    For anything but a point or multipoint, a NaN or infinite x or y makes the geometry not simple.
    In a multipoint, a NaN never lies at the same place as anything.

    .. code-block:: python

        df.filter(geo.is_simple("route"))
    """

    def build(column: pl.Expr, _geometry: GeoArrowType) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name="is_simple",
            is_elementwise=True,
        )

    return on_geometry(geometry, build)
