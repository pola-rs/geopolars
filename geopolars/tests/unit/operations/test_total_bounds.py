"""The bounding box of a whole column."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import (
    BoxXY,
    BoxXYZM,
    LineStringXY,
    MultiPolygonXYZM,
    PolygonXYZM,
)

_INF = float("inf")

_POLYGONS = [
    "POLYGON ZM ((0 0 1 2, 4 0 1 2, 4 3 5 -1, 0 0 1 2))",
    "POLYGON ZM ((-2 1 0 0, 1 1 0 0, 1 7 9 3, -2 1 0 0))",
]
_MULTIPOLYGONS = [
    "MULTIPOLYGON ZM (((0 0 1 2, 4 0 1 2, 4 3 5 -1, 0 0 1 2)), ((10 10 0 0, 11 10 0 0, 11 12 0 8, 10 10 0 0)))",
    "MULTIPOLYGON ZM (((-2 1 0 0, 1 1 -3 0, 1 7 9 3, -2 1 0 0)))",
]


def _geometries(
    wkts: list[str | None], dtype: type, crs: str | None = None
) -> pl.DataFrame:
    return pl.DataFrame({"g": wkts}, schema={"g": pl.String}).select(
        geo.from_wkt("g", dtype, crs=crs)
    )


def _bounds_of(df: pl.DataFrame, **kwargs: bool) -> dict | None:
    out = df.select(geo.total_bounds("g", **kwargs)).to_series()
    assert out.len() == 1
    return out.ext.storage().to_list()[0]


@pytest.mark.parametrize(
    ("wkts", "dtype"),
    [(_POLYGONS, PolygonXYZM), (_MULTIPOLYGONS, MultiPolygonXYZM)],
)
def test_the_box_surrounds_every_rows_box(wkts: list[str], dtype: type) -> None:
    df = _geometries(wkts, dtype)
    boxes = df.select(geo.bounds("g")).to_series().ext.storage().struct.unnest()
    expected = boxes.select(
        *(pl.col(c).min() for c in boxes.columns if c.endswith("min")),
        *(pl.col(c).max() for c in boxes.columns if c.endswith("max")),
    ).row(0, named=True)

    assert _bounds_of(df) == expected
    assert df.select(geo.total_bounds("g")).dtypes == [BoxXYZM()]


def test_a_null_row_makes_the_box_null_unless_ignored() -> None:
    df = _geometries([_POLYGONS[0], None, _POLYGONS[1]], PolygonXYZM)

    assert _bounds_of(df) is None
    assert _bounds_of(df, ignore_nulls=True) == _bounds_of(df.drop_nulls())


@pytest.mark.parametrize("ignore_nulls", [False, True])
@pytest.mark.parametrize("wkts", [[], [None, None]])
def test_no_geometries_give_a_null_box(
    wkts: list[str | None], ignore_nulls: bool
) -> None:
    assert _bounds_of(_geometries(wkts, PolygonXYZM), ignore_nulls=ignore_nulls) is None


def test_rows_without_coordinates_give_the_empty_box() -> None:
    df = _geometries(["LINESTRING EMPTY", "LINESTRING EMPTY"], LineStringXY)

    assert _bounds_of(df) == {"xmin": _INF, "ymin": _INF, "xmax": -_INF, "ymax": -_INF}


def test_a_geographic_box_crosses_the_antimeridian() -> None:
    df = _geometries(
        ["LINESTRING (170 0, 175 1)", "LINESTRING (185 0, 190 1)"],
        LineStringXY,
        crs="EPSG:4326",
    )
    out = df.select(geo.total_bounds("g"))

    assert out.dtypes == [BoxXY(crs="EPSG:4326")]
    assert _bounds_of(df) == {"xmin": 170.0, "ymin": 0.0, "xmax": -170.0, "ymax": 1.0}


def test_chunks_and_slices_give_the_same_box() -> None:
    df = _geometries(_MULTIPOLYGONS + _MULTIPOLYGONS[::-1], MultiPolygonXYZM)
    chunked = pl.concat([df.slice(0, 1), df.slice(1, 2), df.slice(3)], rechunk=False)

    assert chunked.n_chunks() == 3
    assert _bounds_of(chunked) == _bounds_of(df)
    assert _bounds_of(df.slice(1, 2)) == _bounds_of(
        df.slice(1, 1).vstack(df.slice(2, 1))
    )


def test_one_box_per_group() -> None:
    df = _geometries(_POLYGONS * 2, PolygonXYZM).with_columns(
        group=pl.Series([0, 1, 0, 1])
    )
    out = df.group_by("group").agg(geo.total_bounds("g")).sort("group")
    per_group = [_bounds_of(df.filter(group=k)) for k in (0, 1)]

    assert out["g"].ext.storage().to_list() == per_group


def test_namespace_matches_the_functional_api() -> None:
    df = _geometries([_POLYGONS[0], None], PolygonXYZM)

    assert_frame_equal(
        df.select(gpl.col("g").geo.total_bounds(ignore_nulls=True)),
        df.select(geo.total_bounds("g", ignore_nulls=True)),
    )
