"""The intersection of two geometries, forwarded to GEOS (B-tier)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB, validate_one_of
from geopolars.datatypes import (
    LineStringType,
    MultiLineStringType,
    MultiPolygonType,
    PolygonType,
)
from geopolars.geo._dispatch import ensure_same_crs, on_geometry_pair

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

# Ordered by dimension, from points up to polygons.
Keep = Literal["points", "lines", "polygons"]
_KEEP: tuple[Keep, ...] = get_args(Keep)


def _topological_dimension(dtype: GeoArrowType) -> int:
    if isinstance(dtype, PolygonType | MultiPolygonType):
        return 2
    if isinstance(dtype, LineStringType | MultiLineStringType):
        return 1
    return 0


@doc(output="geometry")
def intersection(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    keep: Keep | None = None,
    allow_splitting: bool = False,
    ccw_outer: bool = False,
    ignore_per_row_errors: bool = False,
    early_finite_check: bool = False,
) -> pl.Expr:
    """The part two geometries have in common, row by row.
    A single geometry on either side is intersected with every row of the other.

    The result has one kind for the whole column,
    but the intersection of one row can have parts of several dimensions.
    Two polygons can overlap in an area, but also touch along an edge or at a point,
    and two lines can overlap along a stretch, but also cross at a point.
    So only the parts of one dimension are kept, and the others are dropped.
    By default, that is the dimension of the lower-dimensional input.
    Set `keep` to `"points"`, `"lines"` or `"polygons"` to keep another,
    such as `keep="points"` for where two lines cross.
    It cannot be higher than the default, as two lines never intersect in a polygon.

    The result is a single kind if the lower-dimensional input is one
    (or if either input is one, when they share a dimension).
    A row whose result has several parts then raises.
    With `allow_splitting=True`, the result is the multi kind instead, which holds every part.
    A point cannot split, so a point input always gives a point.
    The table below gives the output kind as "without `allow_splitting` / with it" where the two differ.
    It is symmetric, so only its upper half is filled.

    | a \\ b | Point | MultiPoint | LineString | MultiLineString | Polygon | MultiPolygon |
    | - | - | - | - | - | - | - |
    | Point | Point | Point | Point | Point | Point | Point |
    | MultiPoint | | MultiPoint | MultiPoint | MultiPoint | MultiPoint | MultiPoint |
    | LineString | | | LineString / MultiLineString | LineString / MultiLineString | LineString / MultiLineString | LineString / MultiLineString |
    | MultiLineString | | | | MultiLineString | MultiLineString | MultiLineString |
    | Polygon | | | | | Polygon / MultiPolygon | Polygon / MultiPolygon |
    | MultiPolygon | | | | | | MultiPolygon |

    `keep` only picks the dimension.
    Whether the result is a single or a multi kind still follows the table above.
    Two linestrings with `keep="points"` therefore give a Point,
    and a row where they cross twice raises without `allow_splitting`.

    | `keep` | allowed when | single / multi kind | what is dropped |
    | - | - | - | - |
    | `None` | always | as the table above | parts of a lower dimension than the lower-dimensional input |
    | `"points"` | always | Point / MultiPoint | every line and polygon part |
    | `"lines"` | neither is a point or multipoint | LineString / MultiLineString | every point and polygon part |
    | `"polygons"` | both are polygons or multipolygons | Polygon / MultiPolygon | every point and line part |

    Points here are parts of the result, not vertices of its lines or polygons.
    A dropped part does not count towards a split,
    so two polygons that overlap in one area and touch at a point give one polygon.
    A line that crosses itself is cut where it does,
    so it splits even if it lies entirely inside a polygon.

    {computed_by_geos_keeping_z}
    The output has z as soon as either input has it.
    Where only one input has z, GEOS fills it in for the other.
    Where a line crosses or runs along an edge with z, its z is interpolated along that edge.
    Any other vertex, and any point, only gets a rough estimate:
    the average z of the vertices with z in the same cell of a 3 by 3 grid over both inputs.
    A point on a line with z can therefore be far off from the z interpolated there.
    {empty_and_null}
    Two geometries that do not meet, or only meet in parts that are dropped,
    give an empty {output} too.
    A Point result is then the empty point (NaN on every axis).

    {geos_errors}
    The same goes for a result in several parts without `allow_splitting`.

    {geos_non_finite}
    {early_finite_check}

    {ccw_outer}
    It only affects a Polygon or MultiPolygon result.

    `a` and `b` have to declare the same CRS, or neither one.
    The result has the CRS of `a`.
    A CRS that does not match, or a `keep` that cannot be kept, raises at plan time.

    ```python
    df.select(geo.intersection("parcel", "flood_zone"))
    df.select(geo.intersection("road", "river", keep="points", allow_splitting=True))
    ```
    """
    if keep is not None:
        validate_one_of("keep", keep, Keep)
    kwargs = {
        "keep": keep,
        "allow_splitting": allow_splitting,
        "ccw_outer": ccw_outer,
        "ignore_per_row_errors": ignore_per_row_errors,
        "early_finite_check": early_finite_check,
    }

    def build(
        a: pl.Expr, a_dtype: GeoArrowType, b: pl.Expr, b_dtype: GeoArrowType
    ) -> pl.Expr:
        ensure_same_crs(a_dtype, b_dtype)
        lowest = min(_topological_dimension(a_dtype), _topological_dimension(b_dtype))
        if keep is not None and _KEEP.index(keep) > lowest:
            msg = (
                f"the intersection of {a_dtype!r} and {b_dtype!r} has no {keep};"
                f" it can keep at most {_KEEP[lowest]}"
            )
            raise ValueError(msg)
        return register_plugin_function(
            plugin_path=LIB,
            args=[a, b],
            function_name="intersection",
            kwargs=kwargs,
            is_elementwise=True,
        )

    return on_geometry_pair(a, b, build)
