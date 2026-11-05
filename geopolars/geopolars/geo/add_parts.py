"""Adding parts to geometries, from another column.

A native kernel (A-tier) copies the geometries and their new parts into new buffers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB, validate_one_of
from geopolars.geo._dispatch import _to_expr, on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

NullParts = Literal["drop", "null"]
EmptyParts = Literal["keep", "drop"]


def _add_parts(
    column: pl.Expr,
    geometry: GeoArrowType,
    parts: IntoExprColumn,
    null_parts: NullParts,
    empty_parts: EmptyParts,
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[column, _to_expr(parts)],
        function_name="add_parts",
        is_elementwise=True,
        kwargs={"null_parts": null_parts, "empty_parts": empty_parts},
    )


@doc(parts="parts", part="part", geometry="geometry")
def add_parts(
    geometry: IntoExprColumn,
    parts: IntoExprColumn,
    *,
    null_parts: NullParts = "drop",
    empty_parts: EmptyParts = "keep",
) -> pl.Expr:
    """Add the parts of `parts` to each geometry, after the parts it already has.
    The result is always a multi-geometry.

    | geometry | parts | out |
    | - | - | - |
    | `PointXY[ZM]`, `MultiPointXY[ZM]` | `PointXY[ZM]`, `list[PointXY[ZM]]`, `MultiPointXY[ZM]` | `MultiPointXY[ZM]` |
    | `LineStringXY[ZM]`, `MultiLineStringXY[ZM]` | `LineStringXY[ZM]`, `list[LineStringXY[ZM]]`, `MultiLineStringXY[ZM]` | `MultiLineStringXY[ZM]` |
    | `PolygonXY[ZM]`, `MultiPolygonXY[ZM]` | `PolygonXY[ZM]`, `list[PolygonXY[ZM]]`, `MultiPolygonXY[ZM]` | `MultiPolygonXY[ZM]` |
    | anything else | | refused |

    A part can also be given as its vertices, as points nested in lists in its shape:
    a linestring as `list[PointXY]` or `MultiPointXY`, a polygon as `list[list[PointXY]]`.
    A polygon can also be given as its rings,
    as `list[LineStringXY]`, `list[MultiPointXY]` or `MultiLineStringXY`.
    Bare coordinates (the storage of points) work in place of the points.
    A list of parts is then a list of those.

    {paired_rows}

    {keeps_dimension_and_crs}
    Parts of another dimension are refused.
    {same_crs}

    The parts are not checked for validity: they are copied as they are.

    A null geometry gives a null result.
    {null_parts}

    `"drop"` suits a list of parts aggregated from rows of which some are null.

    What an empty part (a point with every axis NaN, or a part without vertices)
    does is set by `empty_parts`.
    It only applies to the parts that are added: those the geometry has are kept.

    | `empty_parts` | an empty part |
    | - | - |
    | `"keep"` | is added like any other part |
    | `"drop"` | is left out |

    ```python
    df.select(geo.add_parts("roads", "new_roads"))

    # Every road of a region added to that region's highway.
    roads_per_region = roads.group_by("region").agg("road")
    highways.join(roads_per_region, on="region").select(geo.add_parts("highway", "road"))
    ```
    """
    validate_one_of("null_parts", null_parts, NullParts)
    validate_one_of("empty_parts", empty_parts, EmptyParts)
    return on_geometry(
        geometry,
        lambda column, dtype: _add_parts(column, dtype, parts, null_parts, empty_parts),
    )
