"""Adding the coverings Polars writes into GeoParquet."""

from __future__ import annotations

from typing import TypeVar

import polars as pl

from geopolars.datatypes import GeoArrowType
from geopolars.datatypes.encoded import EncodedGeometry
from geopolars.geo.bounds import bounds

FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)


def with_covering(frame: FrameT) -> FrameT:
    """Add a bbox covering column for every geometry column.

    Each row of the covering is the box around the whole geometry,
    which readers can filter on without touching the geometries.
    It is the box `geo.bounds` gives.

    The covering of a column `<name>` is named `<name>_bbox`,
    or `bbox` for a column named `geometry` or `geography`.
    Polars writes a box column under that name as the GeoParquet covering of its geometry.
    This needs GeoParquet writing to be switched on,
    as described under "GeoParquet" in `docs/source/user-guide/geo/io.md`.

    A covering whose name is already taken raises a `DuplicateError`,
    as does a frame with both a `geometry` and a `geography` geometry column.
    An encoded column, such as `Wkb`, cannot be bounded and raises a `SchemaError`.
    Decode it first, for example with `geo.from_wkb`.
    """
    schema = frame.collect_schema()
    coverings = _with_covering_names(schema)
    for name, bbox in coverings.items():
        if bbox in schema:
            msg = f"the covering of `{name}` would be written to `{bbox}`, which exists"
            raise pl.exceptions.DuplicateError(msg)
    return frame.with_columns(
        bounds(name).alias(bbox) for name, bbox in coverings.items()
    )


def _with_covering_names(schema: pl.Schema) -> dict[str, str]:
    coverings: dict[str, str] = {}
    for name, dtype in schema.items():
        if isinstance(dtype, EncodedGeometry):
            msg = (
                f"a covering cannot bound the encoded column `{name}` ({dtype!r}); "
                "decode it with `geo.from_wkb` first"
            )
            raise pl.exceptions.SchemaError(msg)
        if isinstance(dtype, GeoArrowType):
            bbox = _covering_name(name)
            if bbox in coverings.values():
                msg = f"the coverings of two geometry columns would both be `{bbox}`"
                raise pl.exceptions.DuplicateError(msg)
            coverings[name] = bbox
    return coverings


# Mirrors `covering_name` in Polars' `polars-parquet/src/arrow/geo.rs`.
# Polars refuses to write a box under any other name.
def _covering_name(name: str) -> str:
    return "bbox" if name in ("geometry", "geography") else f"{name}_bbox"
