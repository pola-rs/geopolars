"""The `geoarrow.multipolygon` extension type.

Per the `GeoArrow spec <https://geoarrow.org/format.html>`_,
an array of multipolygons is `List<List<List<Coordinate>>>`.
"""

from __future__ import annotations

from typing import ClassVar

from geopolars.datatypes.base import GeoArrowType
from geopolars.datatypes.dimension import XY, XYM, XYZ, XYZM, Dimension


class MultiPolygonType(GeoArrowType):
    """Base class for `geoarrow.multipolygon`."""

    _extension_name: ClassVar[str] = "geoarrow.multipolygon"
    _display: ClassVar[str] = "multipolygon"
    _nesting: ClassVar[int] = 3
    # A multipolygon's innermost lists are its polygons' rings, and a ring is closed.
    _rings: ClassVar[bool] = True


class MultiPolygonXY(MultiPolygonType):
    """A 2D multipolygon: `List<List<List<Struct<x: f64, y: f64>>>>`."""

    _dimension: ClassVar[Dimension] = XY


class MultiPolygonXYZ(MultiPolygonType):
    """A 3D multipolygon: `List<List<List<Struct<x, y, z: f64>>>>`."""

    _dimension: ClassVar[Dimension] = XYZ


class MultiPolygonXYM(MultiPolygonType):
    """A 2D multipolygon with a measure: `List<List<List<Struct<x, y, m: f64>>>>`."""

    _dimension: ClassVar[Dimension] = XYM


class MultiPolygonXYZM(MultiPolygonType):
    """A 3D multipolygon with a measure: `List<List<List<Struct<x, y, z, m: f64>>>>`."""

    _dimension: ClassVar[Dimension] = XYZM
