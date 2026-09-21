"""Polars plugin expressions, with a statically-typed namespace accessor.

Two equivalent APIs:

    from geopolars import geometry
    geometry.translate("route", dx=1.0, dy=2.0)      # functional

    import geopolars as gp
    gpl.col("route").geometry.translate(1.0, 2.0)     # namespace

Both are fully type-checked. Plain `pl.col("route").geometry...` also works at
runtime, but a checker cannot see namespaces that `register_expr_namespace`
patches onto `pl.Expr`, so prefer `gpl.col` to keep static typing.

In code organisation, the layering is one-directional:
`expr` (namespaces) -> the functional modules -> `datatypes`.
There should be no links the other direction.
"""

from geopolars import datatypes, date_util, dist, geometry, language, panic
from geopolars.datatypes import (
    GeoLineString,
    GeoMultiPoint,
    GeoPoint,
    GeoPolygon,
    LineStringXY,
    LineStringXYM,
    LineStringXYZ,
    LineStringXYZM,
    MultiPointXY,
    MultiPointXYM,
    MultiPointXYZ,
    MultiPointXYZM,
    PointXY,
    PointXYM,
    PointXYZ,
    PointXYZM,
    PolygonXY,
    PolygonXYM,
    PolygonXYZ,
    PolygonXYZM,
)

# Importing this registers the namespaces on pl.Expr as a side effect.
from geopolars.expr import (
    DateUtil,
    Distance,
    Geometry,
    Language,
    Panic,
    PluginExpr,
    as_plugin,
    col,
)

__all__ = [
    "DateUtil",
    "Distance",
    "GeoLineString",
    "GeoMultiPoint",
    "GeoPoint",
    "GeoPolygon",
    "Geometry",
    "Language",
    "LineStringXY",
    "LineStringXYM",
    "LineStringXYZ",
    "LineStringXYZM",
    "MultiPointXY",
    "MultiPointXYM",
    "MultiPointXYZ",
    "MultiPointXYZM",
    "Panic",
    "PluginExpr",
    "PointXY",
    "PointXYM",
    "PointXYZ",
    "PointXYZM",
    "PolygonXY",
    "PolygonXYM",
    "PolygonXYZ",
    "PolygonXYZM",
    "as_plugin",
    "col",
    "datatypes",
    "date_util",
    "dist",
    "geometry",
    "language",
    "panic",
]
