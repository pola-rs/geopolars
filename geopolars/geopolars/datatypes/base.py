"""What every GeoArrow dtype has in common.
Mirrors `Geo` in `src/geoarrow/geo.rs`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from geopolars import geopolars as _rust
from geopolars.datatypes import dimension

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

    from geopolars.datatypes.dimension import Dimension


class GeoArrowType(pl.datatypes.BaseExtension):
    """Functionality common between all GeoArrow types implemented like an interface.
    Instantiate a concrete subclass (`PointXY()`), never one of these.
    """

    #: Extension name this geometry is registered under. Set per geometry.
    _extension_name: ClassVar[str]

    #: How the geometry is shown in a DataFrame header: `point[xy]`.
    _display: ClassVar[str]

    #: `List` layers between the storage and the coordinate struct: a point
    #: stores one coordinate per row, a linestring a list of them.
    _nesting: ClassVar[int]

    # `True` if this geometry's innermost lists are rings.
    _rings: ClassVar[bool] = False

    #: The coordinates this concrete type carries. Empty on a geometry's base
    #: class, which stands for every dimension at once.
    _dimension: ClassVar[Dimension] = ()

    #: The concrete type per dimension, filled in by `__init_subclass__`.
    _by_dimension: ClassVar[dict[Dimension, type[GeoArrowType]]]

    #: Storage type for this concrete type, derived from the two above.
    _geo_storage: ClassVar[PolarsDataType]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls._dimension:
            # A geometry's base class: it collects the dimensions declared under it.
            cls._by_dimension = {}
        else:
            cls._geo_storage = dimension.storage(cls._dimension, cls._nesting)
            cls._by_dimension[cls._dimension] = cls

    def __init__(self, *, crs: str | None = None) -> None:
        """
        `crs` declares the coordinate reference system the coordinates are in,
        in any form PROJ accepts: e.g. `"EPSG:4326"`, WKT or PROJJSON.
        It is only a label: nothing is checked or reprojected.
        """
        # `BaseExtension.__eq__` and rust compare metadata as a *string*.
        # Two spellings of the same CRS still make two different dtypes.
        # For this reason, we keep a single source of truth for JSON in Rust.
        super().__init__(
            name=self._extension_name,
            storage=self._geo_storage,
            metadata=_rust.extension_metadata(crs=crs),
        )

    def _declares_crs(self) -> bool:
        """Whether the metadata names a CRS. Read in Rust, like all metadata."""
        return _rust.declares_crs(self.ext_metadata())

    def _crs(self) -> str | None:
        """The CRS the metadata names, in a form PROJ accepts. `None` without one."""
        return _rust.crs(self.ext_metadata())

    def _with_crs(self, crs: str) -> GeoArrowType:
        """sets metadata's crs to  `crs`.
        Every other metadata key is kept.
        """
        metadata = _rust.with_crs(self.ext_metadata(), crs)
        return self.ext_from_params(self._extension_name, self._geo_storage, metadata)

    def _longitude_turn(self) -> float | None:
        """How far `x` goes in one full turn around the globe, in the CRS's unit:
        `360.0` for degrees. `None` without a CRS, or if `x` is not a longitude.
        Asked of PROJ, in Rust.
        """
        return _rust.longitude_turn(self.ext_metadata())

    def _crs_bounds(
        self, *, allow_wrapped_longitude: bool, within_area_of_use: bool
    ) -> tuple[float, float, float, float] | None:
        """Where `x` and `y` may lie in the CRS, as `(xmin, ymin, xmax, ymax)`.
        `None` without a CRS, or when there is nothing to bound them by.
        See `is_valid_geoarrow`. Asked of PROJ, in Rust.
        """
        return _rust.crs_bounds(
            self.ext_metadata(),
            allow_wrapped_longitude=allow_wrapped_longitude,
            within_area_of_use=within_area_of_use,
        )

    def _is_projected(self) -> bool:
        """Whether the CRS is projected, so that `x` and `y` lie on a plane.
        `False` without a CRS. Evaluated using PROJ, in Rust.
        """
        return _rust.is_projected(self.ext_metadata())

    def _is_equal_area(self) -> bool:
        """Whether the CRS is projected with an equal-area projection.
        `False` without a CRS, or with one that is not projected.
        Evaluated using PROJ, in Rust.
        """
        return _rust.is_equal_area(self.ext_metadata())

    def __repr__(self) -> str:
        # `PointXY`, or `PointXY({"crs":"EPSG:4326"})` with metadata
        metadata = self.ext_metadata()
        if metadata is None:
            return type(self).__name__
        return f"{type(self).__name__}(meta={metadata})"

    def _string_repr(self) -> str:
        # This mirrors the rust `Geo::dyn_display`
        dimension = "".join(self._dimension)
        crs = _rust.crs_label(self.ext_metadata())
        if crs is None:
            return f"{self._display}[{dimension}]"
        return f"{self._display}[{dimension}] {crs}"

    @classmethod
    def of_dimension(cls, dimension: Dimension) -> type[GeoArrowType]:
        """The concrete type of this geometry carrying these coordinates."""
        return cls._by_dimension[dimension]

    @classmethod
    def dimensions(cls) -> tuple[type[GeoArrowType], ...]:
        """Every concrete type of this geometry, one per dimension."""
        return tuple(cls._by_dimension.values())

    @classmethod
    def _with_metadata_of(cls, other: pl.datatypes.BaseExtension) -> GeoArrowType:
        """This (concrete) type, carrying `other`'s metadata.

        For an operation that builds one geometry out of another (or out of a box):
        a fresh `cls()` has no metadata and would drop the CRS.
        The metadata is copied as the string it is, never parsed here,
        so the result compares equal to the same dtype built in Rust.
        """
        metadata = other.ext_metadata()
        return cls.ext_from_params(cls._extension_name, cls._geo_storage, metadata)

    @classmethod
    def ext_from_params(
        cls, name: str, storage: PolarsDataType, metadata: str | None
    ) -> Any:
        """Rebuild a geometry type from what was crossed over the boundary."""
        target = cls._by_dimension.get(
            dimension.dimension_of(storage, cls._nesting) or ()
        )
        if target is None:
            shape = "a list of " * cls._nesting + "a struct of"
            supported = ", ".join("/".join(dim) for dim in cls._by_dimension)
            msg = (
                f"unsupported {name!r} storage: {storage!r}; "
                f"this version supports separated f64 coordinates only, "
                f"as {shape} {supported}"
            )
            # BUG: Polars unwraps this error (`crates/polars-python/src/extension.rs`),
            # so `pl.from_arrow` on interleaved coordinates (GeoPandas' default) panics instead of raising.
            raise ValueError(msg)

        # Bypasses __init__ on purpose, so `metadata` is carried through
        # verbatim rather than reset to None.
        slf = target.__new__(target)
        slf._name = name
        slf._storage = storage
        # TODO: Show metadata in the long name.
        # Must be stored on the type as well.
        # would require parsing on pyhton side too.

        slf._metadata = metadata
        return slf
