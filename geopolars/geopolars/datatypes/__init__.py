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
from geopolars.datatypes.multilinestring import (
    GeoMultiLineString,
    MultiLineStringXY,
    MultiLineStringXYM,
    MultiLineStringXYZ,
    MultiLineStringXYZM,
)
from geopolars.datatypes.multipoint import (
    GeoMultiPoint,
    MultiPointXY,
    MultiPointXYM,
    MultiPointXYZ,
    MultiPointXYZM,
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
    "GeoMultiLineString",
    "GeoMultiPoint",
    "GeoPoint",
    "GeoPolygon",
    "LineStringXY",
    "LineStringXYM",
    "LineStringXYZ",
    "LineStringXYZM",
    "MultiLineStringXY",
    "MultiLineStringXYM",
    "MultiLineStringXYZ",
    "MultiLineStringXYZM",
    "MultiPointXY",
    "MultiPointXYM",
    "MultiPointXYZ",
    "MultiPointXYZM",
    "PointXY",
    "PointXYM",
    "PointXYZ",
    "PointXYZM",
    "PolygonXY",
    "PolygonXYM",
    "PolygonXYZ",
    "PolygonXYZM",
]
