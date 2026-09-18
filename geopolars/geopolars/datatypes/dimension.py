"""Which coordinates a geometry carries, and how it stores them.
Mirrors `Dimension` in `src/geoarrow/dimension.rs`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

#: The coordinate names of each dimension, in the field order the spec fixes.
XY = ("x", "y")
XYZ = ("x", "y", "z")
XYM = ("x", "y", "m")
XYZM = ("x", "y", "z", "m")

ALL = (XY, XYZ, XYM, XYZM)

Dimension = tuple[str, ...]


def coordinates(dimension: Dimension) -> pl.Struct:
    """The separated-coordinate struct an array of these coordinates is stored as."""
    return pl.Struct(dict.fromkeys(dimension, pl.Float64))


def storage(dimension: Dimension, nesting: int) -> PolarsDataType:
    """The storage of a geometry nesting its coordinates `nesting` lists deep."""
    dtype: PolarsDataType = coordinates(dimension)
    for _ in range(nesting):
        dtype = pl.List(dtype)
    return dtype


def dimension_of(storage: PolarsDataType, nesting: int) -> Dimension | None:
    """Read the dimension off such a storage type.

    This is the *only* place the dimension is derived. Everything downstream
    matches on the concrete class instead of re-inspecting struct fields.
    """
    inner = storage
    for _ in range(nesting):
        if not isinstance(inner, pl.List):
            return None
        inner = inner.inner
    if not isinstance(inner, pl.Struct):
        return None
    schema = inner.to_schema()
    if any(dtype != pl.Float64 for dtype in schema.values()):
        return None
    return tuple(schema.keys())
