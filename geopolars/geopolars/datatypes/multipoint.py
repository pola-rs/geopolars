"""The `geoarrow.multipoint` extension type."""

# Same Layout as a geoarrow.linestring.
# The only thing that differentiates them is their dtype,
# TODO: and that the child name of the outer list should be "points"

from __future__ import annotations

from typing import ClassVar

from geopolars.datatypes.base import GeoArrowType
from geopolars.datatypes.dimension import XY, XYM, XYZ, XYZM, Dimension


class GeoMultiPoint(GeoArrowType):
    """Base class for `geoarrow.multipoint`."""

    _extension_name: ClassVar[str] = "geoarrow.multipoint"
    _display: ClassVar[str] = "multipoint"
    _nesting: ClassVar[int] = 1


class MultiPointXY(GeoMultiPoint):
    """A 2D multipoint: `geoarrow.multipoint` over `List<Struct<x: f64, y: f64>>`."""

    _dimension: ClassVar[Dimension] = XY


class MultiPointXYZ(GeoMultiPoint):
    """A 3D multipoint: `geoarrow.multipoint` over `List<Struct<x, y, z: f64>>`."""

    _dimension: ClassVar[Dimension] = XYZ


class MultiPointXYM(GeoMultiPoint):
    """A 2D multipoint with a measure: `List<Struct<x, y, m: f64>>`."""

    _dimension: ClassVar[Dimension] = XYM


class MultiPointXYZM(GeoMultiPoint):
    """A 3D multipoint with a measure: `List<Struct<x, y, z, m: f64>>`."""

    _dimension: ClassVar[Dimension] = XYZM
