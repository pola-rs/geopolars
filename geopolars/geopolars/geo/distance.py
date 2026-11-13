"""How far apart two points are.

Points that declare a CRS are measured along the ellipsoid, in the unit of the CRS
(metres for a CRS in longitude/latitude).
Points without one are measured on the flat plane their coordinates lie in,
in whatever units those are: for longitude/latitude that is degrees,
which shrink east-west towards the poles. Declare the CRS to get metres.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._utils import LIB
from geopolars.datatypes import PointType
from geopolars.geo._dispatch import on_geometry_pair

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


# TODO: add expr for things with multiple points, get the 'closest distance'.
# with flag to also interpolate between points / line?


def _distance(
    a: pl.Expr,
    a_dtype: GeoArrowType,
    b: pl.Expr,
    b_dtype: GeoArrowType,
    *,
    method: Literal["planar", "geodesic"] | None,
) -> pl.Expr:
    # TODO: for other shapes, calculate centroid, and take distance on that.
    if not (isinstance(a_dtype, PointType) and isinstance(b_dtype, PointType)):
        msg = (
            "a distance is measured between two `geoarrow.point` columns, "
            f"got: {a_dtype!r} and {b_dtype!r}"
        )
        raise TypeError(msg)

    declares_crs = a_dtype._declares_crs() or b_dtype._declares_crs()
    if method is None:
        method = "geodesic" if declares_crs else "planar"
    elif method == "geodesic" and not declares_crs:
        msg = "a geodesic distance needs a CRS to measure on, but neither point declares one"
        raise TypeError(msg)

    # Whether both declare the same CRS, and whether PROJ can use it,
    # is for the kernel to check while resolving the schema.
    # The kernel counts the height itself, when both points carry one.
    return register_plugin_function(
        plugin_path=LIB,
        args=[a, b],
        function_name="distance",
        kwargs={"method": method},
        is_elementwise=True,
    )


def distance(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    method: Literal["planar", "geodesic"] | None = None,
) -> pl.Expr:
    """How far apart two points are, row by row, as an `f64`.

    | `method`     | measured                     | in                  |
    |--------------|------------------------------|---------------------|
    | `"planar"`   | Pythagorean                  | coordinate units    |
    | `"geodesic"` | along the CRS's ellipsoid    | CRS units           |

    The distance is calculated according to `method`.
    If not given, points that declare a CRS are measured `"geodesic"`,
    and points without one `"planar"`.
    With either method, points in different CRSs, or a CRS on only one of them, are refused:
    reproject one onto the other with `to_crs` first.
    **Planar** ignores the CRS: for longitude/latitude it comes out in degrees,
    which are not the same length east-west as north-south.
    **Geodesic** takes the points to the longitude/latitude the CRS is defined on
    (for a projected CRS, the one it projects from; no datum is shifted)
    and measures along the geodesic between them, on that CRS's own ellipsoid.
    The result is in the unit of the CRS's `x` and `y` (such as US survey feet for EPSG:2263),
    or in metres when those are degrees.
    Points without a CRS have no ellipsoid, so `"geodesic"` refuses them.

    When both points have a `z`, the difference in height counts too:
    the distance is `sqrt(d² + Δz²)`, with `d` the distance above.
    With `"geodesic"`, `z` is taken to be in the same unit as the result,
    and the geodesic is measured at the surface of the ellipsoid
    (a few parts per million short at aircraft heights).
    If only one of the points has a `z`, there is no height difference to count,
    and they are measured as if neither had one.
    `m` is always ignored.
    An empty point gives a NaN distance.
    Either side may be a single point, in which case its distance to all the other points is measured.

    ```python
    df.select(geo.distance("home", "work"))
    df.select(geo.distance("home", "work", method="planar"))
    ```
    """
    return on_geometry_pair(a, b, partial(_distance, method=method))
