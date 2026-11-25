"""GeoParquet is off in every io test, so an exported variable cannot switch it on.
Use the `geoparquet` fixture to switch Polars' own (experimental) GeoParquet support on.
"""

from __future__ import annotations

import pytest

_VARIABLES = {"POLARS_GEOPARQUET_READ": "1", "POLARS_GEOPARQUET_WRITE": "1.1"}


@pytest.fixture(autouse=True)
def _without_geoparquet(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def geoparquet(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in _VARIABLES.items():
        monkeypatch.setenv(name, value)
