"""The base of the `geoarrow.wkb` and `geoarrow.wkt` extension types.
Mirrors `Encoded` in `src/geoarrow/encoded.rs`.

These hold geometries serialised one value per row.
They are for exchange only,
as described under "WKT and WKB" in `docs/source/user-guide/geo/concepts/geometry-types.md`,
which is why they are not a `GeoArrowType`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from geopolars import geopolars as _rust

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


class EncodedGeometry(pl.datatypes.BaseExtension):
    """Geometries serialised in some encoding, such as WKB."""

    _extension_name: ClassVar[str]
    #: The storage dtype the encoding lives in.
    _storage_dtype: ClassVar[PolarsDataType]
    #: How the type is shown in a DataFrame header: `wkb`.
    _display: ClassVar[str]

    def __init__(self, *, crs: str | None = None) -> None:
        """`crs` labels the coordinates, see `GeoArrowType.__init__`."""
        super().__init__(
            name=self._extension_name,
            storage=self._storage_dtype,
            metadata=_rust.extension_metadata(crs=crs),
        )

    def _declares_crs(self) -> bool:
        """Whether the metadata names a CRS. Read in Rust, like all metadata."""
        return _rust.declares_crs(self.ext_metadata())

    def __repr__(self) -> str:
        # `Wkb`, or `Wkb({"crs":"EPSG:4326"})` with metadata
        metadata = self.ext_metadata()
        if metadata is None:
            return type(self).__name__
        return f"{type(self).__name__}(meta={metadata})"

    def _string_repr(self) -> str:
        return self._display

    @classmethod
    def ext_from_params(
        cls, name: str, storage: PolarsDataType, metadata: str | None
    ) -> Any:
        """Rebuild the type from what was crossed over the boundary."""
        if storage != cls._storage_dtype:
            msg = (
                f"unsupported {name!r} storage: {storage!r}; "
                f"expected {cls._storage_dtype!r}"
            )
            raise ValueError(msg)

        # Bypasses __init__ on purpose, so `metadata` is carried through
        # verbatim rather than reset to None.
        slf = cls.__new__(cls)
        slf._name = name
        slf._storage = storage
        slf._metadata = metadata
        return slf
