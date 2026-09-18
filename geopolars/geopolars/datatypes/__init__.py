"""GeoArrow extension data types.

GeoArrow uses one extension name per geometry, covering every dimension. XY vs
XYZ is a property of the coordinates it stores, so one registered name and one
dispatching class covers them all.
"""

from geopolars.datatypes.base import GeoArrowType
from geopolars.datatypes.linestring import (
    GeoLineString,
    LineStringXY,
    LineStringXYM,
    LineStringXYZ,
    LineStringXYZM,
)
from geopolars.datatypes.point import (
    GeoPoint,
    PointXY,
    PointXYM,
    PointXYZ,
    PointXYZM,
)
from geopolars.datatypes.polygon import (
    GeoPolygon,
    PolygonXY,
    PolygonXYM,
    PolygonXYZ,
    PolygonXYZM,
)
from geopolars.datatypes.registry import GEOMETRIES

__all__ = [
    "GEOMETRIES",
    "GeoArrowType",
    "GeoLineString",
    "GeoPoint",
    "GeoPolygon",
    "LineStringXY",
    "LineStringXYM",
    "LineStringXYZ",
    "LineStringXYZM",
    "PointXY",
    "PointXYM",
    "PointXYZ",
    "PointXYZM",
    "PolygonXY",
    "PolygonXYM",
    "PolygonXYZ",
    "PolygonXYZM",
]
