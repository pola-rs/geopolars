"""The `geoarrow.wkb` extension type.

Geometries encoded as `Well-Known Binary <https://libgeos.org/specifications/wkb/>`_,
one value per row, over `Binary` storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl

from geopolars.datatypes.encoded import EncodedGeometry

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


class Wkb(EncodedGeometry):
    """Geometries encoded as WKB: `geoarrow.wkb` over `Binary`.
    WKB is supported for exchange, but all operations require it to be in arrow format.
    You'll likely want to convert this using `from_wkb`.
    """

    _extension_name: ClassVar[str] = "geoarrow.wkb"
    _storage_dtype: ClassVar[PolarsDataType] = pl.Binary
    _display: ClassVar[str] = "wkb"
