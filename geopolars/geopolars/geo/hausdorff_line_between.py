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

HausdorffNonFinite = Literal["raise", "skip", "skip_empty"]


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
    Every vertex of `a` is measured to the nearest point on `b`, and every vertex of `b` to the nearest point on `a`.
    The line joins the pair that lies furthest apart, running from the point on `a` to the point on `b`.
    On a tie, a vertex of `a` wins over one of `b`, and an earlier vertex over a later one.
    Its length in x and y is the (discrete) Hausdorff distance between the two geometries,
    as GEOS and shapely give it.
    A single geometry on either side is measured against every row of the other.

    Only the vertices are measured from,
    so the distance can come out shorter than the Hausdorff distance between the whole geometries.
    A polygon is measured by its rings,
    so a point inside a polygon is as far from it as from its nearest ring.

    Any kind and dimension can be on either side.
    The line has the axes both `a` and `b` have: a z only if both have a z, and an m only if both have an m.
    The pair is chosen in x and y only.
    An end on a vertex keeps that vertex's z and m.
    An end between two vertices gets z and m interpolated between them.

    The pair is chosen in the plane of the coordinates.
    For a geographic CRS, the geodesic length of the line is therefore close to,
    but not the same as, the Hausdorff distance on the ellipsoid.

    If either geometry is empty, so is the line.

    `non_finite` sets what happens with a vertex that is not finite:
    the empty point, or one with a NaN or infinite value.
    {empty_point}

    - `"raise"`: a geometry with such a vertex raises.
    - `"skip"`: such a vertex is left out.
      A line or ring runs on from the vertex before it to the vertex after it.
      A ring that loses its first (and so last) vertex is no longer closed:
      it is measured as a line up to the last vertex it keeps.
      A geometry left without vertices counts as empty.
    - `"skip_empty"`: the empty point is left out as with `"skip"`, any other such vertex raises.

    A vertex is finite when x, y, z and m are.
    `ignore_z=True` leaves z out of that, and `ignore_m=True` leaves m out.
    They do not change which vertex is the empty point.
    An empty point row is an empty geometry rather than an empty vertex, so it never raises.
    With `ignore_per_row_errors=True`, a geometry that would raise gives a null line instead.

    {same_crs_pair}

    ```python
    df.with_columns(geo.hausdorff_line_between("route", "planned_route").alias("worst_deviation"))
    ```
    """
    validate_one_of("non_finite", non_finite, HausdorffNonFinite)
    kwargs = {
        "skip_empty": non_finite != "raise",
        "skip_non_finite": non_finite == "skip",
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
