"""The ways a test can read and write GeoParquet.

Every GeoParquet test runs against our wrapper and against Polars' own
(experimental) GeoParquet support, which is switched on by environment variables.
Where Polars does not do what the test asks yet, mark the test with
`@pytest.mark.parquet_xfail("polars", reason=...)`: it is then a strict xfail
for that backend only, and turns red once Polars catches up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import polars as pl
import pytest

import geopolars as gpl

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class Parquet(NamedTuple):
    """One way to read and write GeoParquet."""

    name: str
    read: Callable[..., pl.DataFrame]
    scan: Callable[..., pl.LazyFrame]
    #: `(df, path, *, covering=False)`
    write: Callable[..., None]
    #: `(lf, path, *, covering=False)`
    sink: Callable[..., None]
    #: The environment variables it needs.
    env: dict[str, str]


def _wrapper_read(source: Any, **kwargs: Any) -> pl.DataFrame:
    return pl.read_parquet(source).pipe(gpl.from_geoparquet, source, **kwargs)


def _wrapper_scan(source: Any, **kwargs: Any) -> pl.LazyFrame:
    return pl.scan_parquet(source).pipe(gpl.from_geoparquet, source, **kwargs)


def _wrapper_write(df: pl.DataFrame, path: Path, *, covering: bool = False) -> None:
    if covering:
        df = df.pipe(gpl.with_covering)
    df.write_parquet(path, metadata=gpl.geoparquet_metadata(df, covering=covering))


def _wrapper_sink(lf: pl.LazyFrame, path: Path, *, covering: bool = False) -> None:
    if covering:
        lf = lf.pipe(gpl.with_covering)
    lf.sink_parquet(path, metadata=gpl.geoparquet_metadata(lf, covering=covering))


def _polars_write(df: pl.DataFrame, path: Path, *, covering: bool = False) -> None:
    if covering:
        df = df.pipe(gpl.with_covering)
    df.write_parquet(path)


def _polars_sink(lf: pl.LazyFrame, path: Path, *, covering: bool = False) -> None:
    if covering:
        lf = lf.pipe(gpl.with_covering)
    lf.sink_parquet(path)


BACKENDS = {
    "wrapper": Parquet(
        "wrapper", _wrapper_read, _wrapper_scan, _wrapper_write, _wrapper_sink, {}
    ),
    "polars": Parquet(
        "polars",
        pl.read_parquet,
        pl.scan_parquet,
        _polars_write,
        _polars_sink,
        {"POLARS_GEOPARQUET_READ": "1", "POLARS_GEOPARQUET_WRITE": "1.1"},
    ),
    # Polars without GeoParquet: only the extension types in its own schema.
    "plain": Parquet(
        "plain", pl.read_parquet, pl.scan_parquet, _polars_write, _polars_sink, {}
    ),
}


def use_backend(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Parquet:
    """The backend `request` is parametrized with, set up, with its xfails applied."""
    parquet = BACKENDS[request.param]
    for name, value in parquet.env.items():
        monkeypatch.setenv(name, value)
    for marker in request.node.iter_markers("parquet_xfail"):
        if parquet.name in marker.args:
            request.applymarker(pytest.mark.xfail(reason=marker.kwargs["reason"]))
    return parquet


def _polars_xfail(reason: str) -> pytest.MarkDecorator:
    return pytest.mark.parquet_xfail("polars", reason=reason)


#: Where Polars' GeoParquet does not do what the GeoParquet spec (and our wrapper) does.
NOT_REFUSED = _polars_xfail(
    "Polars does not refuse GeoParquet metadata it cannot honour"
)
NO_CHECK_EVERY_FILE = _polars_xfail(
    "Polars only compares the files' dtypes once the query runs, "
    "and has no `check_every_file` to opt out"
)
NO_COVERINGS = _polars_xfail("Polars neither reads nor writes bbox coverings")
STRING_CRS_IS_REFUSED = _polars_xfail(
    "Polars only writes a PROJJSON CRS or OGC:CRS84: it refuses any other string one, "
    "even with `geo` given"
)
REFUSED_AS_INVALID_OPERATION = _polars_xfail(
    "Polars refuses what GeoParquet cannot hold with an InvalidOperationError"
)
BOX_BLOCKS_THE_WRITE = _polars_xfail(
    "a geoarrow.box column fails Polars' whole GeoParquet write"
)


@pytest.fixture(params=["wrapper", "polars"])
def parquet(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Parquet:
    """Parametrize a test over our wrapper and Polars' own GeoParquet."""
    return use_backend(request, monkeypatch)
