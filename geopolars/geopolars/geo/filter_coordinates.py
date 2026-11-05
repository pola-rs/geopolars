"""Emptying, dropping or nulling the coordinates that are not finite, or empty (A-tier)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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

NonFiniteAction = Literal["set_empty", "drop", "set_null"]


@dataclass(frozen=True, kw_only=True)
class NonFinite:
    """Which coordinates to select, and what to do with them: as `filter_coordinates` does, with its arguments.
    Mirrors `NonFiniteKwargs` in `src/expr/filter_coordinates.rs`.
    """

    select_non_finite: bool = True
    select_empty: bool = True
    ignore_z: bool = False
    ignore_m: bool = False
    action: NonFiniteAction = "drop"
    too_short: TooShort = "null"

    def __post_init__(self) -> None:
        for name, allowed in [("action", NonFiniteAction), ("too_short", TooShort)]:
            validate_one_of(name, getattr(self, name), allowed)


def _filter_coordinates(
    column: pl.Expr, _geometry: GeoArrowType, *, non_finite: NonFinite
) -> pl.Expr:
    if non_finite.action == "set_empty" and not non_finite.select_non_finite:
        # Only empty coordinates are selected, and emptying them changes nothing.
        return column
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="filter_coordinates",
        kwargs=asdict(non_finite),
        is_elementwise=True,
    )


@doc(selected="that is selected")
def filter_coordinates(
    geometry: IntoExprColumn,
    *,
    select_non_finite: bool = True,
    select_empty: bool = True,
    ignore_z: bool = False,
    ignore_m: bool = False,
    action: NonFiniteAction = "drop",
    too_short: TooShort = "null",
) -> pl.Expr:
    """Finds coordinates that are not finite (NaN, inf) or empty,
    and performs an action on then (make empty, null, or drop).
    {repairing_function}

    A coordinate is selected by either of:

    - `select_non_finite=True`, if it has a NaN or infinite value, but is not empty.
      x and y are always looked at.
      With `ignore_z=True`, z is not, and with `ignore_m=True`, m is not.
    - `select_empty=True`, if it is the empty point.
      `ignore_z` and `ignore_m` do not change this.

    {empty_point}

    {flag_for_absent_axis}

    {selected_coordinate_actions}

    `"drop"` is the default.
    With `action="set_empty"`, selecting only empty coordinates changes nothing.

    {too_short}

    {keeps_type_dimension_and_crs}

    ```python
    df.select(geo.filter_coordinates("route", select_empty=False))
    ```
    """
    non_finite = NonFinite(
        select_non_finite=select_non_finite,
        select_empty=select_empty,
        ignore_z=ignore_z,
        ignore_m=ignore_m,
        action=action,
        too_short=too_short,
    )
    return on_geometry(geometry, partial(_filter_coordinates, non_finite=non_finite))
