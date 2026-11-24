"""Reading and writing geometry columns as Parquet.

A geometry only lives in the dtype: what a file holds is the extension name,
the metadata string and the storage. These tests pin that a round trip through
a file is enough to get the same dtype back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from geopolars import geo
from geopolars.datatypes import BoxXY, PointType, PointXY
from tests.unit.conftest import Dimension
from tests.unit.io.conftest import (
    Parquet,
    use_backend,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(params=["plain", "polars"])
def parquet(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Parquet:
    """Also Polars without GeoParquet, which keeps any extension type."""
    return use_backend(request, monkeypatch)


def _m_is_refused(parquet: Parquet, dimension: Dimension) -> None:
    """GeoParquet 1.1 has no M values, so only a plain file can hold them."""
    if dimension.has_m and parquet.name != "plain":
        pytest.skip(
            "GeoParquet 1.1 cannot store M values; "
            "see test_refuses_what_geoparquet_cannot_hold"
        )


def test_point_survives_a_parquet_round_trip(
    parquet: Parquet,
    coords: pl.DataFrame,
    dimension: Dimension,
    tmp_path: Path,
) -> None:
    _m_is_refused(parquet, dimension)
    df = coords.select(dimension.point())
    path = tmp_path / "points.parquet"
    parquet.write(df, path)
    back = parquet.read(path)

    assert back.schema["point"] == dimension.point_dtype()
    assert_frame_equal(back, df)


def test_linestring_survives_a_parquet_round_trip(
    parquet: Parquet,
    line_coords: pl.DataFrame,
    dimension: Dimension,
    tmp_path: Path,
) -> None:
    _m_is_refused(parquet, dimension)
    df = dimension.lines(line_coords)
    path = tmp_path / "lines.parquet"
    parquet.write(df, path)
    back = parquet.read(path)

    assert back.schema["line"] == dimension.linestring_dtype()
    assert_frame_equal(back, df)


def test_polygon_survives_a_parquet_round_trip(
    parquet: Parquet,
    ring_coords: pl.DataFrame,
    dimension: Dimension,
    tmp_path: Path,
) -> None:
    _m_is_refused(parquet, dimension)
    df = dimension.polygons(ring_coords)
    path = tmp_path / "polygons.parquet"
    parquet.write(df, path)
    back = parquet.read(path)

    assert back.schema["polygon"] == dimension.polygon_dtype()
    assert_frame_equal(back, df)


def test_metadata_survives_a_parquet_round_trip(
    parquet: Parquet, tmp_path: Path
) -> None:
    """The metadata string is the slot `crs` lives in, so losing it on a write
    would silently drop a file's coordinate reference system.
    PROJJSON, as that is the only CRS GeoParquet holds."""
    metadata = '{"crs":{"type":"GeographicCRS","name":"WGS 84"}}'
    dtype = PointType.ext_from_params(
        "geoarrow.point", PointXY().ext_storage(), metadata
    )
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}).select(
        point=pl.struct("x", "y").ext.to(dtype)
    )
    path = tmp_path / "crs.parquet"
    parquet.write(df, path)
    back = parquet.read(path)

    assert back.schema["point"].ext_metadata() == metadata
    assert back.schema["point"] == dtype


def test_a_null_geometry_survives_a_parquet_round_trip(
    parquet: Parquet,
    dimension: Dimension,
    tmp_path: Path,
) -> None:
    """Nulls live at the outermost level only, and that is where a file has to
    put them back: an empty line is not a null one."""
    _m_is_refused(parquet, dimension)
    df = pl.DataFrame(
        {"vertices": [[{name: 1.0 for name in dimension.coords}], [], None]},
        schema={
            "vertices": pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
        },
    ).select(geo.line_string("vertices").alias("line"))
    path = tmp_path / "nulls.parquet"
    parquet.write(df, path)
    back = parquet.read(path)

    assert back["line"].is_null().to_list() == [False, False, True]
    assert_frame_equal(back, df)


def test_box_survives_a_parquet_round_trip(parquet: Parquet, tmp_path: Path) -> None:
    if parquet.name != "plain":
        pytest.skip(
            "GeoParquet refuses a box that covers no geometry; "
            "see test_a_box_that_covers_no_geometry_is_refused"
        )
    df = pl.DataFrame({"a": [0.0], "b": [1.0], "c": [2.0], "d": [3.0]}).select(
        geo.box("a", "b", "c", "d", crs="EPSG:4326").alias("box")
    )
    path = tmp_path / "boxes.parquet"
    parquet.write(df, path)
    back = parquet.read(path)

    assert back.schema["box"] == BoxXY(crs="EPSG:4326")
    assert_frame_equal(back, df)
