"""Reading and writing geometry columns as Parquet.

A geometry only lives in the dtype: what a file holds is the extension name,
the metadata string and the storage. These tests pin that a round trip through
a file is enough to get the same dtype back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from geopolars.datatypes import GeoPoint, PointXY
from polars.testing import assert_frame_equal

from geopolars import geometry
from tests.unit.conftest import Dimension

if TYPE_CHECKING:
    from pathlib import Path


def test_point_survives_a_parquet_round_trip(
    coords: pl.DataFrame, dimension: Dimension, tmp_path: Path
) -> None:
    df = coords.select(dimension.point())
    path = tmp_path / "points.parquet"
    df.write_parquet(path)
    back = pl.read_parquet(path)

    assert back.schema["point"] == dimension.point_dtype()
    assert_frame_equal(back, df)


def test_linestring_survives_a_parquet_round_trip(
    line_coords: pl.DataFrame, dimension: Dimension, tmp_path: Path
) -> None:
    df = dimension.lines(line_coords)
    path = tmp_path / "lines.parquet"
    df.write_parquet(path)
    back = pl.read_parquet(path)

    assert back.schema["line"] == dimension.linestring_dtype()
    assert_frame_equal(back, df)


def test_polygon_survives_a_parquet_round_trip(
    ring_coords: pl.DataFrame, dimension: Dimension, tmp_path: Path
) -> None:
    df = dimension.polygons(ring_coords)
    path = tmp_path / "polygons.parquet"
    df.write_parquet(path)
    back = pl.read_parquet(path)

    assert back.schema["polygon"] == dimension.polygon_dtype()
    assert_frame_equal(back, df)


def test_metadata_survives_a_parquet_round_trip(tmp_path: Path) -> None:
    """The metadata string is the slot `crs` lives in, so losing it on a write
    would silently drop a file's coordinate reference system."""
    metadata = '{"crs":"EPSG:4326"}'
    dtype = GeoPoint.ext_from_params(
        "geoarrow.point", PointXY().ext_storage(), metadata
    )
    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}).select(
        point=pl.struct("x", "y").ext.to(dtype)
    )
    path = tmp_path / "crs.parquet"
    df.write_parquet(path)
    back = pl.read_parquet(path)

    assert back.schema["point"].ext_metadata() == metadata
    assert back.schema["point"] == dtype


def test_a_missing_geometry_survives_a_parquet_round_trip(
    dimension: Dimension, tmp_path: Path
) -> None:
    """Nulls live at the outermost level only, and that is where a file has to
    put them back: an empty line is not a missing one."""
    df = pl.DataFrame(
        {"vertices": [[{name: 1.0 for name in dimension.coords}], [], None]},
        schema={
            "vertices": pl.List(pl.Struct(dict.fromkeys(dimension.coords, pl.Float64)))
        },
    ).select(geometry.linestring("vertices").alias("line"))
    path = tmp_path / "nulls.parquet"
    df.write_parquet(path)
    back = pl.read_parquet(path)

    assert back["line"].is_null().to_list() == [False, False, True]
    assert_frame_equal(back, df)
