"""The line spanning the Hausdorff distance between two geometries (A-tier)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB, validate_one_of
from geopolars.geo._dispatch import ensure_same_crs, on_geometry_pair

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType

HausdorffNonFinite = Literal["raise", "drop", "drop_empty"]


@doc()
def hausdorff_line_between(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    non_finite: HausdorffNonFinite = "raise",
    ignore_z: bool = False,
    ignore_m: bool = False,
    ignore_per_row_errors: bool = False,
) -> pl.Expr:
    """The line spanning the Hausdorff distance between two geometries, row by row.
    The Hausdorff distance is the furthest any point of one geometry lies from the other geometry.
    Here, it is only measured from the vertices (the discrete Hausdorff distance).
    It always runs from `a` to `b`.

    Every vertex of `a` is measured to the nearest point on `b`, and every vertex of `b` to the nearest point on `a`.
    On a tie, a vertex of `a` wins over one of `b`, and an earlier vertex wins over a later one.
    A single geometry on either side is measured against every row of the other.

    As only the vertices are measured from,
    the line can be shorter than the Hausdorff distance between the whole geometries.
    A polygon is measured by its rings, not by its area.
    A point inside a polygon therefore lies as far from it as from its nearest ring.

    Any kind and dimension can be on either side.
    The line only has the axes both `a` and `b` have.
    It has a z only if both have a z, and an m only if both have an m.
    An end of the line on a vertex keeps that vertex's z and m.
    An end between two vertices gets z and m interpolated between them.

    The furthest vertex is found in x and y only, in the plane of the coordinates.
    Longitude/latitude is not a good plane to measure in:
    away from the equator, a degree east is shorter than a degree north.
    For a geographic CRS, the line can therefore run to another vertex than the furthest on the ellipsoid,
    and its geodesic length can be far from the Hausdorff distance there.

    If either geometry is empty, the line is empty too, even if the other geometry would raise.

    A vertex is not finite if it is the empty point, or has a NaN or infinite value.
    {empty_point}
    `non_finite` sets what happens to such a vertex.

    - `"raise"`: a geometry with such a vertex raises.
    - `"drop"`: such a vertex is left out.
      A line or ring then runs straight from the vertex before it to the vertex after it.
      A ring that loses its first vertex also loses its last one, and is no longer closed.
      It is then measured as a line up to the last vertex it keeps.
    - `"drop_empty"`: the empty point is left out, as with `"drop"`.
      Any other vertex that is not finite raises.

    A geometry that `"drop"` or `"drop_empty"` leaves without vertices counts as empty.

    Whether a vertex is finite depends on its x, y, z, and m.
    `ignore_z=True` leaves z out of that check, and `ignore_m=True` leaves out m.
    Neither changes which vertex is the empty point.
    A point row that is the empty point is an empty geometry, not a geometry with an empty vertex.
    It therefore never raises.
    A geometry that would raise gives a null line instead if `ignore_per_row_errors=True`.

    {same_crs_pair}

    ```python
    df.with_columns(geo.hausdorff_line_between("route", "planned_route").alias("worst_deviation"))
    ```
    """
    validate_one_of("non_finite", non_finite, HausdorffNonFinite)
    kwargs = {
        "select_non_finite": non_finite != "drop",
        "select_empty": non_finite == "raise",
        "ignore_z": ignore_z,
        "ignore_m": ignore_m,
        "ignore_per_row_errors": ignore_per_row_errors,
    }

    def build(
        a: pl.Expr, a_dtype: GeoArrowType, b: pl.Expr, b_dtype: GeoArrowType
    ) -> pl.Expr:
        ensure_same_crs(a_dtype, b_dtype)
        return register_plugin_function(
            plugin_path=LIB,
            args=[a, b],
            function_name="hausdorff_line_between",
            kwargs=kwargs,
            is_elementwise=True,
        )

    return on_geometry_pair(a, b, build)
