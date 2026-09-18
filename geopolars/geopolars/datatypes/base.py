"""What every GeoArrow dtype has in common.
Mirrors `Geo` in `src/geoarrow/geo.rs`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from geopolars.datatypes import dimension

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

    from geopolars.datatypes.dimension import Dimension


class GeoArrowType(pl.datatypes.BaseExtension):
    """Functionality common between all GeoArrow types
    implemented like an interface.
    Instantiate a concrete subclass (`PointXY()`), never one of these.
    """

    #: Extension name this geometry is registered under. Set per geometry.
    _extension_name: ClassVar[str]

    #: How the geometry is shown in a DataFrame header: `point[xy]`.
    _display: ClassVar[str]

    #: `List` layers between the storage and the coordinate struct: a point
    #: stores one coordinate per row, a linestring a list of them.
    _nesting: ClassVar[int]

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

    def __init__(self) -> None:
        # Metadata is None for now.
        # It is the slot GeoArrow reserves for `crs`,
        # `crs_type` and `edges`, encoded as a JSON string -- the one place
        # JSON is involved, and it describes the type, never the coordinates.
        #
        # NOTE: when CRS lands, mind that `BaseExtension.__eq__` compares
        # metadata as a *string*, byte for byte. Two spellings of the same CRS
        # will read as two different dtypes unless we canonicalise here. Rust's
        # `dyn_eq` has no such constraint and can compare semantically.
        super().__init__(
            name=self._extension_name, storage=self._geo_storage, metadata=None
        )

    def __repr__(self) -> str:
        return type(self).__name__

    def _string_repr(self) -> str:
        # Shown as `ext[point[xyzm]]` in a DataFrame header
        return f"{self._display}[{''.join(self._dimension)}]"

    @classmethod
    def of_dimension(cls, dimension: Dimension) -> type[GeoArrowType]:
        """The concrete type of this geometry carrying these coordinates."""
        return cls._by_dimension[dimension]

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
            raise ValueError(msg)

        # Bypasses __init__ on purpose, so `metadata` is carried through
        # verbatim rather than reset to None.
        slf = target.__new__(target)
        slf._name = name
        slf._storage = storage
        slf._metadata = metadata
        return slf
