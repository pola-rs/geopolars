"""The ways a test can read and write GeoParquet.

Every GeoParquet test runs against Polars' own (experimental) GeoParquet support,
which is switched on by environment variables.
Where Polars does not do what the test asks yet, mark the test with
`@pytest.mark.parquet_xfail("polars", reason=...)`: it is then a strict xfail
for that backend only, and turns red once Polars catches up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

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


def _polars_write(df: pl.DataFrame, path: Path, *, covering: bool = False) -> None:
    if covering:
        df = df.pipe(gpl.with_covering)
    df.write_parquet(path)


def _polars_sink(lf: pl.LazyFrame, path: Path, *, covering: bool = False) -> None:
    if covering:
        lf = lf.pipe(gpl.with_covering)
    lf.sink_parquet(path)


BACKENDS = {
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


@pytest.fixture(autouse=True)
def _without_geoparquet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the GeoParquet variables, so one exported in the shell cannot switch it on."""
    for name in ("POLARS_GEOPARQUET_READ", "POLARS_GEOPARQUET_WRITE"):
        monkeypatch.delenv(name, raising=False)


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


COVERING_FIELDS_OUT_OF_ORDER = _polars_xfail(
    "Polars reads a covering whose fields are not in xmin/ymin/xmax/ymax order "
    "as a plain struct, not a box"
)


@pytest.fixture(params=["polars"])
def parquet(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Parquet:
    """Polars' own GeoParquet."""
    return use_backend(request, monkeypatch)
