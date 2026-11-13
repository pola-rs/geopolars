from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB, validate_one_of
from geopolars.geo._dispatch import on_geometry
from geopolars.geo.remove_repeats_in_succession import TooShort

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

CrsBounds = Literal[
    "crs",
    "crs_wrapped_longitude",
    "area_of_use",
    "area_of_use_wrapped_longitude",
]
OutOfBoundsAction = Literal["set_empty", "drop", "set_null", "clamp"]

# What each `crs_bounds` asks of `_crs_bounds`, as
# `(allow_wrapped_longitude, within_area_of_use)`.
_CRS_BOUNDS: dict[str, tuple[bool, bool]] = {
    "crs": (False, False),
    "crs_wrapped_longitude": (True, False),
    "area_of_use": (False, True),
    "area_of_use_wrapped_longitude": (True, True),
}


@dataclass(frozen=True, kw_only=True)
class OutOfBounds:
    """Which bounds the coordinates have to lie in, and what to do with the ones outside them:
    as `fix_out_of_bounds` does, with its arguments.
    `is_valid_geoarrow` and `is_valid_ogc` only use `crs_bounds`.
    """

    crs_bounds: CrsBounds = "crs"
    action: OutOfBoundsAction = "drop"
    too_short: TooShort = "null"

    def __post_init__(self) -> None:
        for name, allowed in [
            ("crs_bounds", CrsBounds),
            ("action", OutOfBoundsAction),
            ("too_short", TooShort),
        ]:
            validate_one_of(name, getattr(self, name), allowed)


def _bounds(
    geometry: GeoArrowType, out_of_bounds: OutOfBounds
) -> tuple[float, float, float, float] | None:
    allow_wrapped_longitude, within_area_of_use = _CRS_BOUNDS[out_of_bounds.crs_bounds]
    return geometry._crs_bounds(
        allow_wrapped_longitude=allow_wrapped_longitude,
        within_area_of_use=within_area_of_use,
    )


def _fix_out_of_bounds(
    column: pl.Expr, geometry: GeoArrowType, *, out_of_bounds: OutOfBounds
) -> pl.Expr:
    bounds = _bounds(geometry, out_of_bounds)
    if bounds is None:
        # Nothing bounds the coordinates, so none lie outside.
        return column
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="fix_out_of_bounds",
        kwargs={
            "bounds": bounds,
            "action": out_of_bounds.action,
            "too_short": out_of_bounds.too_short,
        },
        is_elementwise=True,
    )


@doc(selected="outside the bounds")
def fix_out_of_bounds(
    geometry: IntoExprColumn,
    *,
    crs_bounds: CrsBounds = "crs",
    action: OutOfBoundsAction = "drop",
    too_short: TooShort = "null",
) -> pl.Expr:
    """Drops, empties, nulls or clamps the coordinates that lie outside the bounds of the CRS.
    {repairing_function}

    `crs_bounds` says which bounds the coordinates have to lie in.
    They only apply to a column that declares a CRS PROJ can read.
    An opaque SRID is skipped, unless an area of use is asked for, which then raises.
    Without bounds, the column is returned as it is.

    - `"crs"`, the default: in a geographic CRS, latitude has to lie within a quarter turn of the equator,
      and longitude within half a turn of the prime meridian.
      Each is in the unit of its own axis: ±90 and ±180 for degrees, ±100 and ±200 for grads.
      A projected CRS has no bounds of its own.
    - `"crs_wrapped_longitude"`: as `"crs"`, but longitude may run up to a full turn east (360),
      so that data in the 0 to 360 convention fits as well.
    - `"area_of_use"`: as `"crs"`, and `x` and `y` also have to lie inside the area
      PROJ says the CRS is meant for (UTM zone 31N covers 0 to 6 degrees east, for one).
      Data just past it is common and often fine.
      An area of use that crosses the antimeridian only bounds the latitude.
    - `"area_of_use_wrapped_longitude"`: as `"area_of_use"`, with longitude as in `"crs_wrapped_longitude"`.

    `x` is always the longitude and `y` the latitude, as GeoArrow stores them,
    whatever axis order the CRS declares (EPSG:4326 declares latitude first).
    Only `x` and `y` are looked at, but the empty point never lies outside the bounds.
    {empty_point}

    {selected_coordinate_actions}

    `"drop"` is the default.
    With `action="clamp"`, `x` and `y` are moved onto the nearest bound, and `too_short` is ignored.
    A ring's ends move alike, so a closed ring stays closed.
    Clamping can leave the same vertex twice in a row (see `remove_repeats_in_succession`).
    A NaN in `x` or `y` stays NaN (see `filter_coordinates`).

    {too_short}

    {keeps_type_dimension_and_crs}

    ```python
    df.select(geo.fix_out_of_bounds("route", action="clamp"))
    ```
    """
    out_of_bounds = OutOfBounds(
        crs_bounds=crs_bounds, action=action, too_short=too_short
    )
    return on_geometry(
        geometry, partial(_fix_out_of_bounds, out_of_bounds=out_of_bounds)
    )
