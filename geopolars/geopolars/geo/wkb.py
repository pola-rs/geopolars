"""Converting between `geoarrow.wkb` and (binary) native geometries."""

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


def to_wkb(geometry: IntoExprColumn) -> pl.Expr:
    """Encode every geometry as (little-endian, ISO) WKB.
    The output is in a `Wkb` column.
    The geometry's metadata, such as its CRS, is carried over.
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[geometry],
        function_name="to_wkb",
        is_elementwise=True,
    )


@doc(encoding="WKB", encoding_dtype="Wkb")
def from_wkb(
    wkb: IntoExprColumn,
    dtype: type[GeoArrowType],
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Decode a `Wkb` or plain binary column into `dtype`.

    {decode_into_dtype}

    .. code-block:: python

        pl.read_parquet("parcels.parquet").select(
            geo.from_wkb("geometry", MultiPolygonXY, crs="EPSG:4326")
        )
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[wkb],
        function_name="from_wkb",
        is_elementwise=True,
        kwargs=_decode_kwargs(dtype, crs),
    )
