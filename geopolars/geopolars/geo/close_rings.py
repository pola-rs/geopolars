"""Closing the open rings (and linestrings) of a geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.datatypes import (
    LineStringType,
    MultiLineStringType,
    MultiPolygonType,
    PolygonType,
)
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


@dataclass(frozen=True, kw_only=True)
class CloseRings:
    """How the polygon constructors close open rings: as `close_rings` does, with its flags.
    Mirrors `CloseRingsKwargs` in `src/expr/close_rings.rs`.
    """

    snap_z: bool = False
    snap_m: bool = False


def _close_rings(
    column: pl.Expr, geometry: GeoArrowType, *, closing: CloseRings
) -> pl.Expr:
    if not isinstance(
        geometry,
        (LineStringType, PolygonType, MultiLineStringType, MultiPolygonType),
    ):
        msg = (
            "close_rings expects a linestring, polygon, multilinestring"
            f" or multipolygon column, got: {geometry!r}"
        )
        raise TypeError(msg)
    return register_plugin_function(
        plugin_path=LIB,
        args=[column],
        function_name="close_rings",
        kwargs=asdict(closing),
        is_elementwise=True,
    )


@doc()
def close_rings(
    geometry: IntoExprColumn, *, snap_z: bool = False, snap_m: bool = False
) -> pl.Expr:
    """Closes any unclosed lines into rings.

    {validity_table_start}
    {requires_no_nested_nulls}  {unexpected_results}
    {validity_table_end}

    A ring is closed when its last vertex lies at exactly the same place as its first,
    in x, y and z (see `is_valid_geoarrow`).
    For a polygon, this acts on all of the rings.

    {keeps_type_dimension_and_crs}
    An empty geometry stays empty.
    A ring of fewer than 3 vertices stays too short for `is_valid_geoarrow`, even when closed.

    {closing_rings}

    {flag_for_absent_axis}

    .. code-block:: python

        pl.scan_parquet("parcels.parquet").select(geo.close_rings("parcel"))

    This expression leaves rings that are already closed as-is.
    """
    return on_geometry(
        geometry,
        partial(_close_rings, closing=CloseRings(snap_z=snap_z, snap_m=snap_m)),
    )
