"""The intersection of two geometries, forwarded to GEOS (B-tier)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._typing import (
    EarlyCoordCheck,
    NewHoles,
    NewRingOrientation,
    OnRowError,
    PromoteToMulti,
)
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


@doc(output="geometry", paired_with="intersected with")
def intersection(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    keep: Keep | None = None,
    promote_to_multi: PromoteToMulti = "auto",
    new_ring_orientation: NewRingOrientation = "keep",
    new_holes: NewHoles = "opposite",
    on_row_error: OnRowError = "raise",
    early_coord_check: EarlyCoordCheck = "finite_and_empty",
) -> pl.Expr:
    """The part two geometries have in common, row by row.
    {broadcast_single_geometry}

    The result has one kind for the whole column,
    but the intersection of one row can have parts of several dimensions.
    Two polygons can overlap in an area, but also touch along an edge or at a point,
    and two lines can overlap along a stretch, but also cross at a point.
    Only the parts of one dimension are therefore kept, and the others are dropped.
    By default, that is the dimension of the lower-dimensional input.
    Set `keep` to `"points"`, `"lines"` or `"polygons"` to keep another,
    such as `keep="points"` for where two lines cross.
    It cannot be higher than the default, as two lines never intersect in a polygon.

    {overlay_splitting}
    A point cannot split, so `"auto"` gives a point for a point input.
    With `"never"`, the result is a single kind if the lower-dimensional input is one
    (or if either input is one, when they share a dimension).
    The table below gives the output kind as "`"never"` / `"auto"`" where the two differ.
    `"always"` gives the multi kind of every cell.
    The table is symmetric, so only its upper half is filled.

    ===============  =====  ==========  ============================  ============================  ============================  ============================
    ``a \\ b``        Point  MultiPoint  LineString                    MultiLineString               Polygon                       MultiPolygon
    ===============  =====  ==========  ============================  ============================  ============================  ============================
    Point            Point  Point       Point                         Point                         Point                         Point
    MultiPoint              MultiPoint  MultiPoint                    MultiPoint                    MultiPoint                    MultiPoint
    LineString                          LineString / MultiLineString  LineString / MultiLineString  LineString / MultiLineString  LineString / MultiLineString
    MultiLineString                                                   MultiLineString               MultiLineString               MultiLineString
    Polygon                                                                                         Polygon / MultiPolygon        Polygon / MultiPolygon
    MultiPolygon                                                                                                                  MultiPolygon
    ===============  =====  ==========  ============================  ============================  ============================  ============================

    `keep` only picks the dimension.
    Whether the result is a single or a multi kind still follows the table above.
    Two linestrings with `keep="points"` therefore give a MultiPoint by default.
    With `promote_to_multi="never"`, they give a Point, and a row where they cross twice raises.

    ============  ==================================  ============================  ===========================================================
    `keep`        allowed when                        single / multi kind           what is dropped
    ============  ==================================  ============================  ===========================================================
    `None`        always                              as the table above            parts of a lower dimension than the lower-dimensional input
    `"points"`    always                              Point / MultiPoint            every line and polygon part
    `"lines"`     neither is a point or multipoint    LineString / MultiLineString  every point and polygon part
    `"polygons"`  both are polygons or multipolygons  Polygon / MultiPolygon        every point and line part
    ============  ==================================  ============================  ===========================================================

    Points here are parts of the result, not vertices of its lines or polygons.
    A dropped part does not count towards a split,
    so two polygons that overlap in one area and touch at a point give one polygon.
    A line that crosses itself is cut where it does,
    so it splits even if it lies entirely inside a polygon.

    {computed_by_geos_keeping_z}
    {overlay_z}
    {empty_and_null}
    Two geometries that do not meet, or only meet in parts that are dropped,
    give an empty {output} too.
    A Point result is then the empty point (NaN on every axis).

    {geos_errors}
    The same goes for a result in several parts with `promote_to_multi="never"`.
    {on_row_error}

    {geos_non_finite}
    {early_coord_check}

    {new_ring_orientation}
    It only affects a Polygon or MultiPolygon result.

    {same_crs_pair}
    {overlay_crs}
    A `keep` that cannot be kept also raises at plan time.

    .. code-block:: python

        df.select(geo.intersection("parcel", "flood_zone"))
        df.select(geo.intersection("road", "river", keep="points"))
    """
    validate_one_of("promote_to_multi", promote_to_multi, PromoteToMulti)
    validate_one_of("on_row_error", on_row_error, OnRowError)
    validate_one_of("early_coord_check", early_coord_check, EarlyCoordCheck)
    validate_one_of("new_ring_orientation", new_ring_orientation, NewRingOrientation)
    validate_one_of("new_holes", new_holes, NewHoles)
    if keep is not None:
        validate_one_of("keep", keep, Keep)
    kwargs = {
        "keep": keep,
        "promote_to_multi": promote_to_multi,
        "new_ring_orientation": new_ring_orientation,
        "new_holes": new_holes,
        "on_row_error": on_row_error,
        "early_coord_check": early_coord_check,
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
