"""Check two rings for overlapping
This is a hacky implementation that wraps two linestrings into a Polygon,
and relates them through GEOS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._typing import OnRowError
from geopolars._utils import LIB, validate_one_of
from geopolars.datatypes import LineStringType
from geopolars.geo._dispatch import ensure_same_crs, on_geometry_pair

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


@doc(input="ring")
def rings_overlap(
    a: IntoExprColumn,
    b: IntoExprColumn,
    *,
    include_touching: bool = False,
    on_row_error: OnRowError = "raise",
) -> pl.Expr:
    """Whether two rings overlap, row by row, as a `bool`:
    whether the areas they enclose share area.
    One ring inside the other, or two equal rings, overlap too.

    | `a` and `b` | out |
    | - | - |
    | `PointXY[ZM]` | refused |
    | `LineStringXY[ZM]` | `bool` |
    | `PolygonXY[ZM]` | refused |
    | `MultiPointXY[ZM]` | refused |
    | `MultiLineStringXY[ZM]` | refused |
    | `MultiPolygonXY[ZM]` | refused |

    Rings that only touch, at a vertex or along an edge, do not overlap,
    unless `include_touching` is set.

    {projected_plane_xy}
    {same_crs_pair}
    A side of a single ring is compared against every ring of the other.

    Every linestring has to be a ring:
    at least four vertices, the last exactly at the first.
    Anything else, including an empty linestring, raises.
    With `on_row_error="null"`, it gives a null result instead.
    A ring that is not simple (see `is_simple`) gives no meaningful answer.
    {null_ring_input}

    ```python
    df.filter(~geo.rings_overlap("hole", "other_hole", include_touching=True))
    ```
    """
    validate_one_of("on_row_error", on_row_error, OnRowError)

    def build(
        a: pl.Expr, a_dtype: GeoArrowType, b: pl.Expr, b_dtype: GeoArrowType
    ) -> pl.Expr:
        if not (
            isinstance(a_dtype, LineStringType) and isinstance(b_dtype, LineStringType)
        ):
            msg = (
                "rings_overlap expects two linestring columns,"
                f" got: {a_dtype!r} and {b_dtype!r}"
            )
            raise TypeError(msg)
        ensure_same_crs(a_dtype, b_dtype, described_as="rings")
        return register_plugin_function(
            plugin_path=LIB,
            args=[a, b],
            function_name="rings_overlap",
            kwargs={
                "include_touching": include_touching,
                "on_row_error": on_row_error,
            },
            is_elementwise=True,
        )

    return on_geometry_pair(a, b, build)
