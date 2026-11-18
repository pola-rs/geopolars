"""The `geoarrow.polygon` extension type.

Per the `GeoArrow spec <https://geoarrow.org/format.html>`_,
An array of polygons is `List<List<Coordinate>>`:
one list of rings per polygon, and one list of vertices per ring.
The first ring is the exterior one.
Any that follow are interior rings, the holes.
Every ring is closed, meaning its last vertex repeats its first.
"""

from __future__ import annotations

from typing import ClassVar

from geopolars._docs import doc
from geopolars.datatypes.base import GeoArrowType
from geopolars.datatypes.dimension import XY, XYM, XYZ, XYZM, Dimension


class PolygonType(GeoArrowType):
    """Base class for `geoarrow.polygon`."""

    _extension_name: ClassVar[str] = "geoarrow.polygon"
    _display: ClassVar[str] = "polygon"
    _nesting: ClassVar[int] = 2
    # A polygon's inner lists are its rings, and a ring is closed.
    _rings: ClassVar[bool] = True


class PolygonXY(PolygonType):
    """A 2D polygon: `geoarrow.polygon` over `List<List<Struct<x, y: f64>>>`.

    One list of rings per polygon. That list may be empty: like a linestring,
    and unlike a point, an empty polygon is a geometry this layout can hold.
    """

    _dimension: ClassVar[Dimension] = XY


class PolygonXYZ(PolygonType):
    """A 3D polygon: `geoarrow.polygon` over `List<List<Struct<x, y, z: f64>>>`.

    `z` is the elevation of a vertex.
    """

    _dimension: ClassVar[Dimension] = XYZ


@doc()
class PolygonXYM(PolygonType):
    """A 2D polygon with a measure: `List<List<Struct<x, y, m: f64>>>`.

    {polygon_measure}
    """

    _dimension: ClassVar[Dimension] = XYM


@doc()
class PolygonXYZM(PolygonType):
    """A 3D polygon with a measure: `List<List<Struct<x, y, z, m: f64>>>`.

    {polygon_measure}
    """

    _dimension: ClassVar[Dimension] = XYZM
