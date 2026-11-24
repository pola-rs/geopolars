"""Adding the coverings Polars writes into GeoParquet,
as described under "Bounding boxes" in `docs/source/user-guide/geo/io.md`.
"""

from __future__ import annotations

from typing import TypeVar

import polars as pl

from geopolars.datatypes import GeoArrowType
from geopolars.datatypes.encoded import EncodedGeometry
from geopolars.geo.bounds import _envelope

FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)


def with_covering(frame: FrameT) -> FrameT:
    """Add a bbox covering for every geometry column `<name>`.

    The covering is a column `bbox` for a column named `geometry` or `geography`,
    and `<name>_bbox` for any other, as Polars and geopandas name it.
    It has the bounds of each geometry (see `geo.bounds`),
    which readers can filter on without touching the geometries.
    A multi-geometry gets one box around all of its parts.
    A box crossing the antimeridian has `xmin > xmax`.
    WKB has to be decoded with `geo.from_wkb` first, to be bounded.

    With GeoParquet writing switched on, Polars writes it as the covering,
    as described under "Bounding boxes" in `docs/source/user-guide/geo/io.md`.
    """
    schema = frame.collect_schema()
    coverings = _coverings(schema)
    for name, bbox in coverings.items():
        if bbox in schema:
            msg = f"the covering of `{name}` would be written to `{bbox}`, which exists"
            raise pl.exceptions.DuplicateError(msg)
    return frame.with_columns(
        _envelope(name).alias(bbox) for name, bbox in coverings.items()
    )


def _coverings(schema: pl.Schema) -> dict[str, str]:
    """The bbox column of every geometry column."""
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


def _covering_name(name: str) -> str:
    return "bbox" if name in ("geometry", "geography") else f"{name}_bbox"
