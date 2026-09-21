"""Registering the GeoArrow types, on both sides of the boundary.
Importing this module registers the types in *both* the Polars registry and the
cdylib. They *have to* agree on names.
"""

from __future__ import annotations

import polars as pl

# Importing the compiled module runs `PyInit_geopolars`,
# which is what registers these types on the Rust side.
from geopolars import geopolars as _rust  # noqa: F401
from geopolars.datatypes.base import GeoArrowType
from geopolars.datatypes.linestring import GeoLineString
from geopolars.datatypes.multipoint import GeoMultiPoint
from geopolars.datatypes.point import GeoPoint
from geopolars.datatypes.polygon import GeoPolygon

# Mirrors `Kind::ALL`
GEOMETRIES: tuple[type[GeoArrowType], ...] = (
    GeoPoint,
    GeoLineString,
    GeoPolygon,
    GeoMultiPoint,
)

for _geometry in GEOMETRIES:
    pl.register_extension_type(_geometry._extension_name, _geometry)
