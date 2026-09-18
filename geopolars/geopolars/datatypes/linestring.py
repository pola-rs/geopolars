"""The `geoarrow.linestring` extension type.

Per the [GeoArrow spec](https://geoarrow.org/format.html), an array of
linestrings is `List<Coordinate>`: the same coordinates a point stores, one list
of them per linestring.
"""

from __future__ import annotations

from typing import ClassVar

from geopolars.datatypes.base import GeoArrowType
from geopolars.datatypes.dimension import XY, XYM, XYZ, XYZM, Dimension


class GeoLineString(GeoArrowType):
    """Base class for `geoarrow.linestring`."""

    _extension_name: ClassVar[str] = "geoarrow.linestring"
    _display: ClassVar[str] = "linestring"
    _nesting: ClassVar[int] = 1


class LineStringXY(GeoLineString):
    """A 2D linestring: `geoarrow.linestring` over `List<Struct<x: f64, y: f64>>`.

    One list of vertices per linestring. That list may be empty: unlike a point,
    an empty linestring is a geometry this layout can hold.
    """

    _dimension: ClassVar[Dimension] = XY


class LineStringXYZ(GeoLineString):
    """A 3D linestring: `geoarrow.linestring` over `List<Struct<x, y, z: f64>>`.

    `z` is the elevation of a vertex.
    """

    _dimension: ClassVar[Dimension] = XYZ


class LineStringXYM(GeoLineString):
    """A 2D linestring with a measure: `List<Struct<x, y, m: f64>>`.

    `m` stands for 'measure', an arbitrary value carried per *vertex* rather
    than per linestring. Typical uses:
    - a timestamp, making the linestring a trajectory
    - a distance along a route
    - an airspeed measurement
    """

    _dimension: ClassVar[Dimension] = XYM


class LineStringXYZM(GeoLineString):
    """A 3D linestring with a measure: `List<Struct<x, y, z, m: f64>>`.

    `m` stands for 'measure', an arbitrary value carried per *vertex* rather
    than per linestring. Typical uses:
    - a timestamp, making the linestring a trajectory
    - a distance along a route
    - an airspeed measurement
    """

    _dimension: ClassVar[Dimension] = XYZM
