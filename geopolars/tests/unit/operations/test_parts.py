"""The parts of a multi-geometry, as a list."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from geopolars import geo
from geopolars.datatypes import LineStringXY, MultiLineStringXY
from tests.unit.conftest import Dimension

RD = "EPSG:28992"

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))

Line = list[dict[str, float]]


def _line(*xy: tuple[float, float]) -> Line:
    return [{"x": x, "y": y} for x, y in xy]


def _multilines(*multis: list[Line] | None, crs: str | None = None) -> pl.DataFrame:
    storage = pl.Series("multi", list(multis), dtype=pl.List(_XY_VERTICES))
    return pl.DataFrame(storage.ext.to(MultiLineStringXY(crs=crs)))


def _parts(df: pl.DataFrame) -> list:
    """Every part's storage, per row."""
    parts = df.select(geo.parts(df.columns[0])).to_series()
    return parts.list.eval(pl.element().ext.storage()).to_list()


def test_a_multilinestring_gives_its_linestrings_in_order() -> None:
    first, second = _line((0, 0), (1, 1)), _line((5, 5), (6, 7), (8, 8))
    df = _multilines([first, second], crs=RD)

    out = df.select(geo.parts("multi"))

    assert out.schema["multi"] == pl.List(LineStringXY(crs=RD))
    assert _parts(df) == [[first, second]]


def test_a_null_or_empty_multi_geometry() -> None:
    assert _parts(_multilines(None, [])) == [None, []]


def _storage_of_parts(df: pl.DataFrame, name: str) -> pl.DataFrame:
    return df.select(geo.parts(name).list.eval(pl.element().ext.storage()).alias(name))


def test_a_multipoint_gives_its_points(
    line_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipoints(line_coords)

    out = df.select(geo.parts("multipoint"))

    assert out.schema["multipoint"] == pl.List(dimension.point_dtype())
    assert_frame_equal(
        _storage_of_parts(df, "multipoint"),
        df.select(pl.col("multipoint").ext.storage()),
    )


def test_a_multilinestring_keeps_its_dimension(
    ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multilinestrings(ring_coords)

    out = df.select(geo.parts("multilinestring"))

    assert out.schema["multilinestring"] == pl.List(dimension.linestring_dtype())
    assert_frame_equal(
        _storage_of_parts(df, "multilinestring"),
        df.select(pl.col("multilinestring").ext.storage()),
    )


def test_a_multipolygon_gives_its_polygons(
    multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    df = dimension.multipolygons(multipolygon_coords)

    out = df.select(geo.parts("multipolygon"))

    assert out.schema["multipolygon"] == pl.List(dimension.polygon_dtype())
    assert_frame_equal(
        _storage_of_parts(df, "multipolygon"),
        df.select(pl.col("multipolygon").ext.storage()),
    )


def test_the_parts_build_the_multilinestring_back() -> None:
    df = _multilines([_line((0, 0), (1, 1)), _line((5, 5), (6, 7))], [], None, crs=RD)

    out = df.select(geo.multi_line_string_from_line_strings(geo.parts("multi")))

    assert_frame_equal(out, df)


def test_rejects_a_single_geometry() -> None:
    storage = pl.Series("line", [_line((0, 0), (1, 1))], dtype=_XY_VERTICES)
    df = pl.DataFrame(storage.ext.to(LineStringXY()))

    with pytest.raises(TypeError, match="multipoint, multilinestring or multipolygon"):
        df.select(geo.parts("line"))
