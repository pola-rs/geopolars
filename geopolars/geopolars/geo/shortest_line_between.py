"""The shortest line between two geometries, forwarded to GEOS (B-tier)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.geo._dispatch import ensure_same_crs, on_geometry_pair
from geopolars.geo.crs import to_crs

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


@doc(output="line", paired_with="measured against")
def shortest_line_between(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    in_crs: str | None = None,
    ignore_per_row_errors: bool = False,
    early_finite_check: bool = False,
) -> pl.Expr:
    """The shortest line between two geometries, row by row, as a `LineStringXY`.
    It runs from the point on `a` closest to `b`, to the point on `b` closest to `a`.
    If the geometries intersect, both vertices are the same point.
    The length of the line is the planar distance between the two geometries, in x and y.
    {broadcast_single_geometry}

    Any kind and dimension can be on either side.
    Computed by GEOS in x and y only, so the line has no z or m.

    Distances are measured in the plane of the coordinates.
    With `in_crs`, they are measured in the plane of `in_crs` instead:
    both geometries are reprojected to it first (see `to_crs`),
    and the line is given in it.
    `in_crs` is experimental, and may change or go away.
    Reprojecting both geometries with `to_crs` first gives the same line.

    | `in_crs` | `a` and `b` | line in |
    | - | - | - |
    | `None` | both declare the same CRS, or neither does | the CRS of `a` and `b` |
    | a CRS | both declare a CRS, which may differ | `in_crs` |

    Any other combination raises at plan time.

    No plane gives the shortest line on the ellipsoid everywhere.
    Longitude/latitude is not a good plane to measure in:
    away from the equator, a degree east is shorter than a degree north.
    Web Mercator (`in_crs="EPSG:3857"`) keeps those in proportion,
    but the further from the equator, the more it stretches distances, without bound towards the poles.
    It also puts geometries on either side of the antimeridian a world apart.
    A latitude beyond 90° raises.

    If either geometry is empty, so is the line.

    {geos_errors}

    {geos_non_finite}
    GEOS then gives no meaningful answer.
    Depending on where the vertex is, it raises or might give a line that is not the shortest.
    {early_finite_check}

    ```python
    df.with_columns(geo.shortest_line_between("house", "road").alias("driveway"))
    ```
    """
    kwargs = {
        "ignore_per_row_errors": ignore_per_row_errors,
        "early_finite_check": early_finite_check,
    }

    def build(
        a: pl.Expr, a_dtype: GeoArrowType, b: pl.Expr, b_dtype: GeoArrowType
    ) -> pl.Expr:
        if in_crs is None:
            ensure_same_crs(a_dtype, b_dtype)
        else:
            # NOTE: `in_crs` is experimental, to try out how reprojecting inside an expression works in the API.
            # Other expressions (e.g. `distance`, `hausdorff_line_between`) leave composing `to_crs` to the user.
            if not (a_dtype._declares_crs() and b_dtype._declares_crs()):
                msg = (
                    f"both geometries need a CRS to be reprojected to {in_crs}; "
                    "pass `in_crs=None` to measure in the plane of their own coordinates"
                )
                raise ValueError(msg)
            # GEOS drops z, so how it is reprojected does not matter.
            z_flags = {
                "allow_projection_with_unscaled_z": True,
                "assume_z_in_target_crs": True,
            }

            # The CRSs are compared as written, so a CRS written differently
            # (e.g. as PROJJSON) is still reprojected, which only costs time.
            def in_plane(side: pl.Expr, dtype: GeoArrowType) -> pl.Expr:
                if dtype._crs() == in_crs:
                    return side
                return to_crs(side, in_crs, **z_flags)

            a, b = in_plane(a, a_dtype), in_plane(b, b_dtype)
        return register_plugin_function(
            plugin_path=LIB,
            args=[a, b],
            function_name="shortest_line_between",
            kwargs=kwargs,
            is_elementwise=True,
        )

    return on_geometry_pair(a, b, build)
