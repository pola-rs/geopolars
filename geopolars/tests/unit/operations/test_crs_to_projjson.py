"""Writing a geometry's CRS as PROJJSON, without moving the coordinates."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from geopolars import geo
from geopolars.datatypes import BoxXY, PointXY, Wkb
from tests.unit.operations.test_set_crs import _geometries, _points

if TYPE_CHECKING:
    from pathlib import Path

    from tests.unit.conftest import Dimension

WGS84 = "EPSG:4326"
PROJJSON_ID = {"authority": "EPSG", "code": 4326}


def _crs_of(dtype: pl.DataType) -> Any:
    return json.loads(dtype.ext_metadata())["crs"]


def test_every_geometry_gets_its_crs_as_projjson_and_keeps_its_coordinates(
    dimension: Dimension,
    coords: pl.DataFrame,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
    multipolygon_coords: pl.DataFrame,
) -> None:
    geometries = _geometries(
        dimension, coords, line_coords, ring_coords, multipolygon_coords
    )
    for df, dtype in geometries:
        (name,) = df.columns
        labelled = df.select(geo.set_crs(name, WGS84))

        out = labelled.lazy().select(geo.crs_to_projjson(name))

        assert type(out.collect_schema()[name]) is dtype
        assert _crs_of(out.collect_schema()[name])["id"] == PROJJSON_ID
        assert_frame_equal(
            out.select(pl.col(name).ext.storage()).collect(),
            df.select(pl.col(name).ext.storage()),
        )


def test_wkb_gets_its_crs_as_projjson() -> None:
    wkb = _points(crs=WGS84).select(geo.to_wkb("point"))

    out = wkb.select(geo.crs_to_projjson("point"))

    assert isinstance(out.schema["point"], Wkb)
    assert _crs_of(out.schema["point"])["id"] == PROJJSON_ID
    assert_frame_equal(
        out.select(pl.col("point").ext.storage()),
        wkb.select(pl.col("point").ext.storage()),
    )


def test_a_box_gets_its_crs_as_projjson() -> None:
    """Such as the covering `with_covering` adds."""
    boxes = _points(crs=WGS84).select(geo.bounds("point"))

    out = boxes.select(geo.crs_to_projjson("point"))

    assert isinstance(out.schema["point"], BoxXY)
    assert _crs_of(out.schema["point"])["id"] == PROJJSON_ID
    assert_frame_equal(
        out.select(pl.col("point").ext.storage()),
        boxes.select(pl.col("point").ext.storage()),
    )


@pytest.mark.parametrize(
    "crs", [None, json.dumps({"type": "GeographicCRS", "name": "WGS 84"})]
)
def test_no_crs_or_a_projjson_one_is_left_as_it_is(crs: str | None) -> None:
    points = _points(crs=crs)
    assert_frame_equal(points.select(geo.crs_to_projjson("point")), points)


def test_every_other_metadata_key_is_kept() -> None:
    metadata = '{"edges":"spherical","crs":"EPSG:4326"}'
    dtype = PointXY.ext_from_params(
        PointXY._extension_name, PointXY._geo_storage, metadata
    )
    points = _points().select(pl.col("point").ext.storage().ext.to(dtype))

    out = points.select(geo.crs_to_projjson("point"))

    assert json.loads(out.schema["point"].ext_metadata())["edges"] == "spherical"
    assert _crs_of(out.schema["point"])["id"] == PROJJSON_ID


def test_a_crs_proj_cannot_read_is_refused_while_the_plan_is_built() -> None:
    points = _points().select(
        pl.col("point").ext.storage().ext.to(PointXY(crs="nonsense"))
    )
    lf = points.lazy().select(geo.crs_to_projjson("point"))

    with pytest.raises(pl.exceptions.ComputeError, match="PROJ cannot read"):
        lf.collect_schema()


def test_a_non_geometry_is_refused() -> None:
    lf = pl.LazyFrame({"x": [1.0]}).select(geo.crs_to_projjson("x"))

    with pytest.raises(TypeError):
        lf.collect_schema()


def test_polars_writes_it_as_geoparquet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    path = tmp_path / "points.parquet"

    _points(crs=WGS84).select(geo.crs_to_projjson("point")).write_parquet(path)

    geo_metadata = json.loads(pl.read_parquet_metadata(path)["geo"])
    assert geo_metadata["columns"]["point"]["crs"]["id"] == PROJJSON_ID
