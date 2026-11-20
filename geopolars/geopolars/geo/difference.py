"""The difference of two geometries, forwarded to GEOS (B-tier)."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from geopolars.geo._dispatch import ensure_same_crs, on_geometry_pair

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


@doc(output="geometry", paired_with="paired with")
def difference(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    promote_to_multi: PromoteToMulti = "auto",
    new_ring_orientation: NewRingOrientation = "keep",
    new_holes: NewHoles = "opposite",
    on_row_error: OnRowError = "raise",
    early_coord_check: EarlyCoordCheck = "finite_and_empty",
) -> pl.Expr:
    """The geometry you get when you subtract `b` from `a`.

    Taking the difference is only really meaningful if the two geometries have the same
    *dimensionality* (as in a line is 1d, a polygon 2d, not whether or not it has z/m).
    A polygon minus a line or a point covers the same area as before,
    although GEOS may add a vertex where the line crosses its boundary.
    Likewise, a line minus a point covers the same length as before.

    The result is the kind of `a`, and of `a` alone.
    For a LineString or Polygon, the result can split.
    {overlay_splitting}
    The table below gives the output kind for every `promote_to_multi`.

    ===============  ===============  ======================  ====================
    a                `"never"`        `"auto"`                `"always"`
    ===============  ===============  ======================  ====================
    Point            Point            Point                   MultiPoint
    MultiPoint       MultiPoint       MultiPoint              MultiPoint
    LineString       LineString       MultiLineString         MultiLineString
    MultiLineString  MultiLineString  MultiLineString         MultiLineString
    Polygon          Polygon          MultiPolygon            MultiPolygon
    MultiPolygon     MultiPolygon     MultiPolygon            MultiPolygon
    ===============  ===============  ======================  ====================

    {broadcast_single_geometry}

    There are some gotcha's with linestrings:
    A line minus a line that crosses it only gains a vertex where the two cross.
    GEOS also cuts a line where it crosses itself, and a MultiLineString where its parts cross.
    A LineString that crosses itself therefore splits whenever it goes through GEOS,
    even if `b` does not touch it.
    Only an empty `b` leaves it whole (see below).

    {computed_by_geos_keeping_z}
    {overlay_z}

    {null_on_either_side}
    An empty `a` gives an empty {output}, without going through GEOS.
    An empty `b` takes nothing away, and gives `a` back as it is, also without going through GEOS.
    The one exception is a polygon part of `a` whose rings have no vertices.
    It is stored without rings, like every other empty polygon.
    A `b` that covers all of `a` takes everything away, so it gives an empty {output} too.

    {geos_errors}
    The same goes for a result in several parts with `promote_to_multi="never"`.
    {on_row_error}

    {geos_non_finite}
    {early_coord_check}
    A row where `b` is empty does not go to GEOS,
    but still fails on such a vertex in `a` whatever `early_coord_check` is,
    which `on_row_error` then handles.
    GEOS itself does not raise on every such vertex,
    so this row can be stricter than one that GEOS computes.

    {new_ring_orientation}
    It only affects a Polygon or MultiPolygon result.

    {same_crs_pair}
    {overlay_crs}

    .. code-block:: python

        df.select(geo.difference("parcel", "road_reserve"))
        df.select(geo.difference("trail", "private_land", promote_to_multi="never"))
    """
    validate_one_of("promote_to_multi", promote_to_multi, PromoteToMulti)
    validate_one_of("on_row_error", on_row_error, OnRowError)
    validate_one_of("early_coord_check", early_coord_check, EarlyCoordCheck)
    validate_one_of("new_ring_orientation", new_ring_orientation, NewRingOrientation)
    validate_one_of("new_holes", new_holes, NewHoles)
    kwargs = {
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
        return register_plugin_function(
            plugin_path=LIB,
            args=[a, b],
            function_name="difference",
            kwargs=kwargs,
            is_elementwise=True,
        )

    return on_geometry_pair(a, b, build)
