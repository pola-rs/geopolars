"""Converting between `geoarrow.wkt` and (string) native geometries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.geo.construct._common import _decode_kwargs

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType


def to_wkt(geometry: IntoExprColumn) -> pl.Expr:
    """Encode every geometry as WKT, e.g. `POINT Z(1 2 3)`.
    The output is in a `Wkt` column.
    The geometry's metadata, such as its CRS, is carried over.
    Coordinates are written in full, so they read back exactly.
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[geometry],
        function_name="to_wkt",
        is_elementwise=True,
    )


@doc(encoding="WKT", encoding_dtype="Wkt")
def from_wkt(
    wkt: IntoExprColumn,
    dtype: type[GeoArrowType],
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Decode a `Wkt` or plain string column into `dtype`.

    {decode_into_dtype}

    .. code-block:: python

        pl.read_csv("parcels.csv").select(
            geo.from_wkt("geometry", MultiPolygonXY, crs="EPSG:4326")
        )
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[wkt],
        function_name="from_wkt",
        is_elementwise=True,
        kwargs=_decode_kwargs(dtype, crs),
    )
