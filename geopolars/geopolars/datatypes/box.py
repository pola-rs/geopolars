"""The `geoarrow.box` extension type.
Mirrors `GeoBox` in `src/geoarrow/bbox.rs`.

Per the `GeoArrow spec <https://geoarrow.org/format.html>`_:
an array of axis-aligned rectangles is a struct of their bounds,
`Struct<xmin, ymin, [zmin], [mmin], xmax, ymax, [zmax], [mmax]>`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from geopolars import geopolars as _rust
from geopolars.datatypes import dimension
from geopolars.datatypes.dimension import XY, XYM, XYZ, XYZM, Dimension

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


class BoxType(pl.datatypes.BaseExtension):
    """Base class for `geoarrow.box`.
    Instantiate a concrete subclass (`BoxXY()`), never this one.

    `xmin > xmax` is allowed: the box then crosses the antimeridian.
    A range from `inf` to `-inf` is empty.
    """

    _extension_name: ClassVar[str] = "geoarrow.box"

    #: The coordinates this concrete type bounds. Empty on `BoxType` itself.
    _dimension: ClassVar[Dimension] = ()

    #: The concrete type per dimension, filled in by `__init_subclass__`.
    _by_dimension: ClassVar[dict[Dimension, type[BoxType]]] = {}

    #: Storage type for this concrete type.
    _box_storage: ClassVar[pl.Struct]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._box_storage = dimension.box_storage(cls._dimension)
        cls._by_dimension[cls._dimension] = cls

    def __init__(self, *, crs: str | None = None) -> None:
        """`crs` labels the coordinates, see `GeoArrowType.__init__`."""
        super().__init__(
            name=self._extension_name,
            storage=self._box_storage,
            metadata=_rust.extension_metadata(crs=crs),
        )

    def _declares_crs(self) -> bool:
        """Whether the metadata names a CRS. Read in Rust, like all metadata."""
        return _rust.declares_crs(self.ext_metadata())

    def _with_projjson_crs(self) -> BoxType:
        """The same type, with its CRS written as PROJJSON. See `geo.crs_to_projjson`."""
        metadata = _rust.with_projjson_crs(self.ext_metadata())
        return self.ext_from_params(self._extension_name, self._box_storage, metadata)

    def _longitude_turn(self) -> float | None:
        """How far `x` goes in one full turn around the globe, in the CRS's unit:
        `360.0` for degrees. `None` without a CRS, or if `x` is not a longitude.
        Asked of PROJ, in Rust.
        """
        return _rust.longitude_turn(self.ext_metadata())

    def __repr__(self) -> str:
        # `BoxXY`, or `BoxXY({"crs":"EPSG:4326"})` with metadata
        metadata = self.ext_metadata()
        if metadata is None:
            return type(self).__name__
        return f"{type(self).__name__}(meta={metadata})"

    def _string_repr(self) -> str:
        # Shown as `ext[box[xyzm]]` in a DataFrame header
        return f"box[{''.join(self._dimension)}]"

    @classmethod
    def of_dimension(cls, dimension: Dimension) -> type[BoxType]:
        """The concrete box bounding these coordinates."""
        return cls._by_dimension[dimension]

    @classmethod
    def _with_metadata_of(cls, other: pl.datatypes.BaseExtension) -> BoxType:
        """This (concrete) box type, carrying `other`'s metadata.
        See `GeoArrowType._with_metadata_of`.
        """
        metadata = other.ext_metadata()
        return cls.ext_from_params(cls._extension_name, cls._box_storage, metadata)

    @classmethod
    def ext_from_params(
        cls, name: str, storage: PolarsDataType, metadata: str | None
    ) -> Any:
        """Rebuild a box type from what was crossed over the boundary."""
        target = cls._by_dimension.get(dimension.box_dimension_of(storage) or ())
        if target is None:
            supported = ", ".join(
                "/".join(dimension.box_storage(dim).to_schema())
                for dim in cls._by_dimension
            )
            msg = (
                f"unsupported {name!r} storage: {storage!r}; "
                f"expected a struct of f64 bounds, one of {supported}"
            )
            raise ValueError(msg)

        # Bypasses __init__ on purpose, so `metadata` is carried through
        # verbatim rather than reset to None.
        slf = target.__new__(target)
        slf._name = name
        slf._storage = storage
        slf._metadata = metadata
        return slf


class BoxXY(BoxType):
    """A 2D box: `geoarrow.box` over `Struct<xmin, ymin, xmax, ymax: f64>`."""

    _dimension: ClassVar[Dimension] = XY


class BoxXYZ(BoxType):
    """A 3D box: `Struct<xmin, ymin, zmin, xmax, ymax, zmax: f64>`."""

    _dimension: ClassVar[Dimension] = XYZ


class BoxXYM(BoxType):
    """A 2D box with a range of measures: `Struct<xmin, ymin, mmin, xmax, ymax, mmax: f64>`."""

    _dimension: ClassVar[Dimension] = XYM


class BoxXYZM(BoxType):
    """A 3D box with a range of measures:
    `Struct<xmin, ymin, zmin, mmin, xmax, ymax, zmax, mmax: f64>`.
    """

    _dimension: ClassVar[Dimension] = XYZM
