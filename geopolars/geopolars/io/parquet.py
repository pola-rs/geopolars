"""Reading and writing GeoParquet files.

A GeoParquet file is a Parquet file with a `geo` key in its metadata.
It says which columns hold geometries, how, and in which CRS.
Reading and writing the file is done with Polars' own readers and writers.
This module only has the steps around them:
`from_geoparquet` gives the geometry columns their dtypes after reading,
and `geoparquet_metadata` (with `with_covering`) gives the `geo` key to write.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import polars as pl

# Private, but the same expansion `pl.scan_parquet` does, and there is no public one.
from polars.io._expand_paths import _expand_paths

from geopolars import geopolars as _rust
from geopolars.datatypes import BoxType, GeoArrowType
from geopolars.datatypes.encoded import EncodedGeometry
from geopolars.geo.bounds import _envelope

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from polars._typing import StorageOptionsDict
    from polars.io.cloud import CredentialProviderFunction

FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)


def from_geoparquet(
    frame: FrameT,
    source: str | Path | Sequence[str | Path],
    *,
    check_every_file: bool = True,
    storage_options: StorageOptionsDict | None = None,
    credential_provider: CredentialProviderFunction | Literal["auto"] | None = "auto",
    glob: bool = True,
    hidden_file_prefix: str | Sequence[str] | None = None,
) -> FrameT:
    """
    Geoparquet only saves metadata stuff in the geo footer.
    Polars saves the type information (e.g. PolygonXY),
    but other software does not.
    We can figure out the dimensions from struct names though,
    and we can re-read the footer (which is why we need `path`).

    `frame` is what `pl.read_parquet` or `pl.scan_parquet` read from `source`.
    The columns get geopolars dtypes, with the metadata of the files in `source`.
    Native encodings become e.g. `PolygonXY`,
    and WKB becomes `Wkb` (which you can decode to native yourself with `geo.from_wkb`).
    A file without GeoParquet metadata leaves the frame as it is.


    .. code-block:: python

        pl.scan_parquet("countries/*.parquet").pipe(
            gpl.from_geoparquet, "countries/*.parquet"
        ).select(geo.from_wkb("geometry", MultiPolygonXY))
    """
    paths = (
        _expand_paths(
            source,
            glob=glob,
            hidden_file_prefix=hidden_file_prefix,
            storage_options=storage_options,
            credential_provider=credential_provider,
        )
        .collect()
        .to_series()
        .to_list()
    )
    if not check_every_file:
        paths = paths[:1]
    geos = [
        pl.read_parquet_metadata(
            path,
            storage_options=storage_options,
            credential_provider=credential_provider,
        ).get("geo")
        for path in paths
    ]
    if all(geo is None for geo in geos):
        return frame

    # Polars reads with the first file's schema too.
    schema = frame.collect_schema()
    first: dict[str, pl.DataType] | None = None
    for path, geo in zip(paths, geos, strict=True):
        if geo is None:
            msg = f"{path!r} has no GeoParquet metadata, but {paths[0]!r} does"
            raise pl.exceptions.SchemaError(msg)
        dtypes = _geometry_dtypes(geo, schema)
        if first is None:
            first = dtypes
        elif dtypes != first:
            msg = (
                f"{path!r} describes its geometry columns as {dtypes}, "
                f"but {paths[0]!r} as {first}; "
                "pass `check_every_file=False` to read all of them as the first"
            )
            raise pl.exceptions.SchemaError(msg)

    assert first is not None
    return frame.with_columns(
        _storage_of(name, dtype).ext.to(dtype)
        for name, dtype in first.items()
        # A file Polars wrote can already have it.
        if schema[name] != dtype
    )


def _storage_of(name: str, dtype: pl.DataType) -> pl.Expr:
    """The column `name`, as the storage `dtype` wraps."""
    storage = pl.col(name).ext.storage()
    if not isinstance(dtype, BoxType):
        return storage
    # A covering column can have its bounds in any order: put them in the box's.
    bounds = [storage.struct.field(field) for field in dtype.ext_storage().to_schema()]
    # A row without a geometry must not have a bbox, but some writers (geopandas)
    # give it one with every bound null instead.
    # We follow the spec.
    all_null = pl.all_horizontal(bound.is_null() for bound in bounds)
    return pl.when(~all_null).then(pl.struct(bounds)).alias(name)


def _geometry_dtypes(geo: str, schema: pl.Schema) -> dict[str, pl.DataType]:
    """The dtype of every geometry column the `geo` metadata describes,
    for a file Polars reads with `schema`.
    Converted in Rust, like all metadata.
    """
    frame = _rust.geoparquet_dtypes(geo, pl.DataFrame(schema=schema))
    return dict(frame.schema)


def with_covering(frame: FrameT) -> FrameT:
    """Add a bbox covering for every geometry column `<name>`.

    The covering is a column `<name>_bbox` with the bounds of each geometry
    (see `geo.bounds`), which readers can filter on without touching the geometries.
    A multi-geometry gets one box around all of its parts.
    A box crossing the antimeridian has `xmin > xmax`.
    WKB has to be decoded with `geo.from_wkb` first, to be bounded.

    Describe it in the file with `geoparquet_metadata(frame, covering=True)`.
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


def geoparquet_metadata(
    frame: pl.DataFrame | pl.LazyFrame, *, covering: bool = False
) -> dict[str, str]:
    """The `geo` metadata that makes a Parquet file of `frame` GeoParquet.

    Pass it to Polars' writer, next to any metadata of your own:

    .. code-block:: python

        df.write_parquet(path, metadata=gpl.geoparquet_metadata(df))
        lf.sink_parquet(path, metadata={**gpl.geoparquet_metadata(lf), "owner": "me"})

    Polars reads its own files back with their dtypes without it,
    but other GeoParquet readers (such as geopandas or GDAL) need it.
    Every geometry column is described.
    The geometries are written in their own encoding: native, or WKB for a `Wkb` column.
    The WKB we write is modern, so some older readers may not accept it.
    A frame without geometry columns gets no metadata: `{}`.

    GeoParquet 1.1 does not support M values and no WKT.
    Columns with such data are refused.
    CRS that cannot be written with PROJJSON are also refused.
    Only the schema is needed, so this refuses before anything is written.

    With `covering`, every geometry column `<name>` is described with the covering
    `<name>_bbox` that `with_covering` adds.
    """
    schema = frame.collect_schema()
    coverings = _coverings(schema) if covering else {}
    geo = _rust.geoparquet_metadata(pl.DataFrame(schema=schema), coverings)
    return {} if geo is None else {"geo": geo}


def _coverings(schema: pl.Schema) -> dict[str, str]:
    """The bbox column of every geometry column: `<name>_bbox`."""
    coverings = {}
    for name, dtype in schema.items():
        if isinstance(dtype, EncodedGeometry):
            msg = (
                f"a covering cannot bound the encoded column `{name}` ({dtype!r}); "
                "decode it with `geo.from_wkb` first"
            )
            raise pl.exceptions.SchemaError(msg)
        if isinstance(dtype, GeoArrowType):
            coverings[name] = f"{name}_bbox"
    return coverings
