"""The `geoarrow.wkt` extension type.

Geometries encoded as `Well-Known Text <https://libgeos.org/specifications/wkt/>`_,
one value per row, over `String` storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl

from geopolars.datatypes.encoded import EncodedGeometry

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


class Wkt(EncodedGeometry):
    """Geometries encoded as WKT: `geoarrow.wkt` over `String`.
    WKT is supported for exchange, but all operations require it to be in arrow format.
    You'll likely want to convert this using `from_wkt`.
    """

    _extension_name: ClassVar[str] = "geoarrow.wkt"
    _storage_dtype: ClassVar[PolarsDataType] = pl.String
    _display: ClassVar[str] = "wkt"
