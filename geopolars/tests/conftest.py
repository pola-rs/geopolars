from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
from polars.lazyframe import engine as plengine
from polars.testing import assert_schema_equal

if TYPE_CHECKING:
    from collections.abc import Iterable


def _check_lazy_schema(
    lf: pl.LazyFrame, df: pl.DataFrame, optimizations: pl.QueryOptFlags
) -> None:
    """Compare `lf.collect_schema()` against the schema of the collected `df`."""
    if optimizations._pyoptflags.eager:
        return

    expected = lf.collect_schema()
    assert_schema_equal(expected, df.schema)


@pytest.fixture(autouse=True)
def _check_lazy_collect_schema(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert every lazy query's declared schema matches what it produces.

    Mirrors the fixture of the same name in `py-polars/tests/conftest.py`.
    This is important for a plugin,
    as it needs to manually declare output types in the `#[polars_expr]` attribute.

    Opt a test out with `@pytest.mark.may_fail_lazy_schema`.
    """
    if request.node.get_closest_marker("may_fail_lazy_schema") is not None:
        return

    wrapped_collect = plengine._LocalEngine.collect
    wrapped_collect_all = plengine._LocalEngine.collect_all

    def collect(self: Any, lf: pl.LazyFrame, **kwargs: Any) -> Any:
        out = wrapped_collect(self, lf, **kwargs)
        # `background=True` hands back an `InProcessQuery`, with nothing to compare.
        if isinstance(out, pl.DataFrame):
            _check_lazy_schema(lf, out, kwargs["optimizations"])
        return out

    def collect_all(
        self: Any, lfs: Iterable[pl.LazyFrame], **kwargs: Any
    ) -> list[pl.DataFrame]:
        lfs = list(lfs)
        out = wrapped_collect_all(self, lfs, **kwargs)
        for lf, df in zip(lfs, out, strict=True):
            _check_lazy_schema(lf, df, kwargs["optimizations"])
        return out

    monkeypatch.setattr(plengine._LocalEngine, "collect", collect)
    monkeypatch.setattr(plengine._LocalEngine, "collect_all", collect_all)
