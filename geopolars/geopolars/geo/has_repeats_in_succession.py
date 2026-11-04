"""Whether a geometry has two points that coincide (A-tier)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB, validate_coincidence_tolerances
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


@doc()
def has_repeats_in_succession(
    geometry: IntoExprColumn,
    *,
    abs_tol: float | None = None,
    rel_tol: float | None = None,
    unique_z: bool = True,
    unique_m: bool = False,
) -> pl.Expr:
    """Whether each geometry has two points in a row that lie at the same place.
    One `bool` per row, a multi-geometry included. A null geometry gives a null result.

    The geometries have to be valid by `is_valid_geoarrow` already. This trusts its input.

    | geometry | has repeated points when |
    | - | - |
    | `PointXY[ZM]` | never |
    | `LineStringXY[ZM]`, `MultiPointXY[ZM]` | two vertices in a row lie at the same place |
    | `PolygonXY[ZM]` | two vertices in a row of one ring lie at the same place |
    | `MultiLineStringXY[ZM]`, `MultiPolygonXY[ZM]` | one of its parts has repeated points |

    The ends of a ring lie at the same place, but they are not in a row, so they do not count.

    {coincidence_axes}
    {coincidence_tolerance}

    Repeated points are valid, also by OGC (see `is_valid_ogc`), but some tools may stumble over them.

    ```python
    df.filter(~geo.has_repeats_in_succession("route"))
    ```
    """
    validate_coincidence_tolerances(abs_tol, rel_tol)

    def build(column: pl.Expr, _geometry: GeoArrowType) -> pl.Expr:
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name="has_repeats_in_succession",
            is_elementwise=True,
            kwargs={
                "abs_tol": abs_tol,
                "rel_tol": rel_tol,
                "unique_z": unique_z,
                "unique_m": unique_m,
            },
        )

    return on_geometry(geometry, build)
