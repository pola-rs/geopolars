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
    Operations only work on it once it is decoded with `from_wkt`,
    as described under "WKT and WKB" in `docs/source/user-guide/geo/concepts/geometry-types.md`.
    """

    _extension_name: ClassVar[str] = "geoarrow.wkt"
    _storage_dtype: ClassVar[PolarsDataType] = pl.String
    _display: ClassVar[str] = "wkt"
