"""The offset curve of a line: the line running alongside it at a distance.

Forwarded to GEOS (B-tier).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._typing import EarlyCoordCheck, OnRowError, PromoteToMulti
from geopolars._utils import LIB, validate_one_of
from geopolars.datatypes import LineStringType, MultiLineStringType
from geopolars.geo._dispatch import on_geometry
from geopolars.geo.buffer import (
    _JOIN_STYLES,
    _MITRE_LIMIT,
    _check_one_of,
    _distances_of,
)

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


@doc(output="linestring")
def offset_curve(
    geometry: IntoExprColumn,
    distance: float | IntoExprColumn,
    *,
    quadrant_segments: int = 8,
    join_style: Literal["round", "mitre", "bevel"] = "round",
    mitre_limit: float | None = None,
    promote_to_multi: PromoteToMulti = "auto",
    on_row_error: OnRowError = "raise",
    early_coord_check: EarlyCoordCheck = "finite_and_empty",
) -> pl.Expr:
    """The line running alongside every line, `distance` away from it.
    A positive `distance` offsets to the left, relative to the direction of the line.
    A negative one offsets to the right.
    The curve runs in the same direction as the line.
    `distance` is in the units of the coordinates
    (degrees for longitude/latitude).
    It is one number for every geometry,
    or a column (or expression) with one distance per geometry.
    A null distance gives a null linestring.
    A NaN or infinite distance in a column raises.

    Only linestrings and multilinestrings are accepted.
    The output is always XY:

    =========================  ===================================  ===================================
    in                         out                                  with `promote_to_multi="never"`
    =========================  ===================================  ===================================
    `LineStringXY[Z][M]`       `MultiLineStringXY`                  `LineStringXY`
    `MultiLineStringXY[Z][M]`  `MultiLineStringXY`                  `MultiLineStringXY`
    =========================  ===================================  ===================================

    {computed_by_geos_dropping_z_and_m}
    {empty_and_null}
    A line without length also gives an empty linestring.
    So does an offset that vanishes,
    such as the inside of a hairpin bend narrower than `distance`.

    The curve leaves out the parts that come closer to the line than `distance`.
    Where the line crosses itself or comes close to itself,
    that can split the curve into several pieces, whatever the sign of `distance`.
    A linestring column cannot hold those.
    As any linestring can split, a linestring column gives multilinestrings by default (`promote_to_multi="auto"`),
    which hold every piece. `"always"` does the same.
    With `"never"`, it gives linestrings, and a row that splits fails.
    A multilinestring always gets every piece as a part.
    See :ref:`promote_to_multi <arg-promote_to_multi>` on the Common arguments page.

    `quadrant_segments`, `join_style` and `mitre_limit` shape the curve
    as they shape a buffer (see `buffer`).
    GEOS uses at least 8 `quadrant_segments` for an offset curve,
    so a lower value is refused.

    {geos_errors}
    The same goes for a linestring it splits with `promote_to_multi="never"`.
    {on_row_error}

    {geos_non_finite}
    {early_coord_check}

    .. code-block:: python

        df.select(geo.offset_curve("road", 5.0))
        df.select(geo.offset_curve("road", -5.0, join_style="mitre"))
    """
    validate_one_of("promote_to_multi", promote_to_multi, PromoteToMulti)
    validate_one_of("on_row_error", on_row_error, OnRowError)
    validate_one_of("early_coord_check", early_coord_check, EarlyCoordCheck)
    constant = isinstance(distance, int | float)
    if constant and not math.isfinite(distance):
        msg = f"offset_curve expects a finite distance, got: {distance!r}"
        raise ValueError(msg)
    if quadrant_segments < 8:
        msg = (
            "offset_curve expects `quadrant_segments` of at least 8,"
            f" got: {quadrant_segments!r}"
        )
        raise ValueError(msg)
    _check_one_of("offset_curve", "join_style", join_style, _JOIN_STYLES)
    if mitre_limit is not None and join_style != "mitre":
        msg = (
            "offset_curve only takes `mitre_limit` with join_style='mitre',"
            f" got: {join_style!r}"
        )
        raise ValueError(msg)

    def build(column: pl.Expr, dtype: GeoArrowType) -> pl.Expr:
        if not isinstance(dtype, LineStringType | MultiLineStringType):
            msg = (
                "offset_curve expects a linestring or multilinestring column,"
                f" got: {dtype!r}"
            )
            raise TypeError(msg)
        args = [column] if constant else [column, _distances_of(column, distance)]
        return register_plugin_function(
            plugin_path=LIB,
            args=args,
            function_name="offset_curve",
            kwargs={
                # `None` when the distances are the second argument.
                "distance": float(distance) if constant else None,
                "quadrant_segments": int(quadrant_segments),
                "join_style": join_style,
                "mitre_limit": _MITRE_LIMIT
                if mitre_limit is None
                else float(mitre_limit),
                "promote_to_multi": promote_to_multi,
                "on_row_error": on_row_error,
                "early_coord_check": early_coord_check,
            },
            is_elementwise=True,
        )

    return on_geometry(geometry, build)
