"""The `geoarrow.point` extension type.

Per the `GeoArrow spec <https://geoarrow.org/format.html>`_:
an array of points is an array of coordinates.
In the 'separated' coordinates representation that we use,
that is `Struct<x: double, y: double, ...>`.
"""

from __future__ import annotations

from typing import ClassVar

from geopolars._docs import doc
from geopolars.datatypes.base import GeoArrowType
from geopolars.datatypes.dimension import XY, XYM, XYZ, XYZM, Dimension


class PointType(GeoArrowType):
    """Base class for `geoarrow.point`.

    Point != Coordinate: a Coordinate is simply the 'encoding'.
    A series of coordinates can be 'a line' or 'a collection of points',
    which is why this distinction is made.
    """

    _extension_name: ClassVar[str] = "geoarrow.point"
    _display: ClassVar[str] = "point"
    _nesting: ClassVar[int] = 0


class PointXY(PointType):
    """A 2D point: `geoarrow.point` over `Struct<x: f64, y: f64>`."""

    _dimension: ClassVar[Dimension] = XY


class PointXYZ(PointType):
    """A 3D point: `geoarrow.point` over `Struct<x: f64, y: f64, z: f64>`.

    `z` is the elevation.
    """

    _dimension: ClassVar[Dimension] = XYZ


@doc()
class PointXYM(PointType):
    """A 2D point with a measure: `Struct<x: f64, y: f64, m: f64>`.

    {measure}
    """

    _dimension: ClassVar[Dimension] = XYM


@doc()
class PointXYZM(PointType):
    """A 3D point with a measure: `Struct<x: f64, y: f64, z: f64, m: f64>`.

    {measure}
    """

    _dimension: ClassVar[Dimension] = XYZM
