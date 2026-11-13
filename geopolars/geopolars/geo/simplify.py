"""The simplification of a geometry: fewer vertices, within a tolerance.

Forwarded to GEOS (B-tier).
A (multi)point has nothing to leave out, so it is returned as it is (C-tier).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.datatypes import MultiPointType, PointType
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


@doc(output="geometry")
def simplify(
    geometry: IntoExprColumn,
    tolerance: float,
    *,
    preserve_topology: bool = False,
    ignore_per_row_errors: bool = False,
    early_finite_check: bool = False,
) -> pl.Expr:
    """Every geometry with fewer vertices,
    none of which strays more than `tolerance` from the original.
    `tolerance` is in the units of the coordinates
    (degrees for longitude/latitude).


    A linestring or polygon (or their multi-variants) keeps its kind,
    and an XY or XYM geometry gives XY, an XYZ or XYZM one XYZ:

    | in | out |
    | - | - |
    | `PointXY[ZM]` | `PointXY[ZM]` |
    | `LineStringXY[M]` | `LineStringXY` |
    | `LineStringXYZ[M]` | `LineStringXYZ` |
    | `PolygonXY[M]` | `PolygonXY` |
    | `PolygonXYZ[M]` | `PolygonXYZ` |
    | `MultiPointXY[ZM]` | `MultiPointXY[ZM]` |
    | `MultiLineStringXY[M]` | `MultiLineStringXY` |
    | `MultiLineStringXYZ[M]` | `MultiLineStringXYZ` |
    | `MultiPolygonXY[M]` | `MultiPolygonXY` |
    | `MultiPolygonXYZ[M]` | `MultiPolygonXYZ` |

    {computed_by_geos_keeping_z}
    A point or multipoint has nothing to leave out,
    so it is returned as it is, m included.
    {ignores_empty_and_invalid}
    A linestring always keeps its endpoints.
    An empty part of a multilinestring is dropped.

    By default this uses Douglas-Peucker simplification,
    which might break topology.

    If `preserve_topology=False`:
    - a polygon smaller than `tolerance` becomes empty;
    - a hole, or a part of a multipolygon, smaller than `tolerance` is dropped;
    - a polygon whose simplified rings cross is repaired,
      which can split it into more polygons.
      For a multipolygon those are more parts,
      but a polygon can't hold them, so that polygon fails
      (`preserve_topology=True` never splits it).

    With `preserve_topology=True`, GEOS's topology-preserving simplifier is used instead.
    This is slower, but never lets rings cross each other or polygons collapse.
    No linestring or ring then crosses itself or another, and every part and ring is kept.

    {geos_errors}
    The same goes for a polygon it splits.

    {geos_non_finite}
    With `preserve_topology=True`, GEOS then raises.
    With `preserve_topology=False`, such a vertex is silently left out,
    unless it starts a ring, where GEOS raises.
    {early_finite_check}

    ```python
    df.select(geo.simplify("parcel", 0.5))
    ```
    """
    if not tolerance >= 0:
        msg = f"simplify expects a tolerance of at least 0, got: {tolerance!r}"
        raise ValueError(msg)

    def build(column: pl.Expr, dtype: GeoArrowType) -> pl.Expr:
        if isinstance(dtype, PointType | MultiPointType):
            return column
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name="simplify",
            kwargs={
                "tolerance": float(tolerance),
                "preserve_topology": preserve_topology,
                "ignore_per_row_errors": ignore_per_row_errors,
                "early_finite_check": early_finite_check,
            },
            is_elementwise=True,
        )

    return on_geometry(geometry, build)
