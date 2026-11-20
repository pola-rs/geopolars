"""Writing GeoParquet"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
import shapely
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import (
    BoxXY,
    MultiLineStringXYZ,
    PointType,
    PointXY,
    PointXYM,
    PointXYZ,
    PolygonXYZM,
    Wkb,
    Wkt,
)
from tests.unit.io.test_geoparquet import DATA, OTHER_DTYPES, SPEC_GEOMETRIES, _read

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import PolarsDataType

CRS84 = "OGC:CRS84"
XY = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]


def _point(metadata: str) -> PolarsDataType:
    """A `PointXY` with metadata the constructor would not write."""
    return PointType.ext_from_params(
        "geoarrow.point", PointXY().ext_storage(), metadata
    )


def _write(df: pl.DataFrame, path: Path, *, covering: bool = False) -> None:
    if covering:
        df = df.pipe(gpl.with_covering)
    df.write_parquet(path, metadata=gpl.geoparquet_metadata(df, covering=covering))


def _sink(lf: pl.LazyFrame, path: Path, *, covering: bool = False) -> None:
    if covering:
        lf = lf.pipe(gpl.with_covering)
    lf.sink_parquet(path, metadata=gpl.geoparquet_metadata(lf, covering=covering))


def _geo_of(path: Path) -> dict[str, Any]:
    return json.loads(pl.read_parquet_metadata(path)["geo"])


def _written(tmp_path: Path, dtype: PolarsDataType) -> dict[str, Any]:
    """The `geo` metadata written for a `geometry` column of `dtype`."""
    path = tmp_path / "written.parquet"
    _write(pl.DataFrame(schema={"geometry": dtype}), path)
    return _geo_of(path)["columns"]["geometry"]


def test_writes_a_native_column(tmp_path: Path) -> None:
    path = tmp_path / "points.parquet"
    df = pl.DataFrame({"geometry": XY}).select(
        pl.col("geometry").ext.to(PointXY(crs="EPSG:4326"))
    )
    _write(df, path)
    geo = _geo_of(path)

    assert geo["version"] == "1.1.0"
    assert geo["primary_column"] == "geometry"
    column = geo["columns"]["geometry"]
    assert column["encoding"] == "point"
    assert column["geometry_types"] == ["Point"]
    # Converted by PROJ, as GeoParquet only takes PROJJSON.
    assert column["crs"]["id"] == {"authority": "EPSG", "code": 4326}
    assert "edges" not in column
    assert "epoch" not in column


def test_a_z_coordinate_is_in_the_geometry_type(tmp_path: Path) -> None:
    column = _written(tmp_path, MultiLineStringXYZ())
    assert column["encoding"] == "multilinestring"
    assert column["geometry_types"] == ["MultiLineString Z"]


def test_no_crs_is_written_as_null(tmp_path: Path) -> None:
    """Leaving the key out would claim the data is in OGC:CRS84."""
    for dtype in [PointXY(), _point('{"crs":null}')]:
        column = _written(tmp_path, dtype)
        assert "crs" in column
        assert column["crs"] is None


def test_crs84_is_left_out(tmp_path: Path) -> None:
    """because it is the default"""
    assert "crs" not in _written(tmp_path, PointXY(crs=CRS84))


def test_writes_spherical_edges(tmp_path: Path) -> None:
    column = _written(tmp_path, _point('{"crs":"OGC:CRS84","edges":"spherical"}'))
    assert column["edges"] == "spherical"


def test_writes_wkb_without_geometry_types(tmp_path: Path) -> None:
    """Knowing them would mean reading every value. `[]` says they are unknown."""
    column = _written(tmp_path, Wkb())
    assert column["encoding"] == "WKB"
    assert column["geometry_types"] == []


@pytest.mark.parametrize(
    ("dtype", "error", "match"),
    [
        (PointXYM(), pl.exceptions.SchemaError, "M values"),
        (PolygonXYZM(), pl.exceptions.SchemaError, "M values"),
        (Wkt(), pl.exceptions.SchemaError, "WKT"),
        (
            _point('{"crs":"4326","crs_type":"srid"}'),
            pl.exceptions.ComputeError,
            "SRID",
        ),
        (_point('{"crs":"not a crs"}'), pl.exceptions.ComputeError, "PROJ"),
        (_point('{"edges":"karney"}'), pl.exceptions.ComputeError, "karney"),
        (_point('{"edges":"geodesic-ish"}'), pl.exceptions.ComputeError, "edges"),
    ],
)
def test_refuses_what_geoparquet_cannot_hold(
    tmp_path: Path, dtype: PolarsDataType, error: type[Exception], match: str
) -> None:
    path = tmp_path / "refused.parquet"
    with pytest.raises(error, match=match):
        _write(pl.DataFrame(schema={"geometry": dtype}), path)
    assert not path.exists()


def test_describes_every_geometry_column(tmp_path: Path) -> None:
    path = tmp_path / "columns.parquet"
    schema = {"id": pl.Int64, "a": PointXY(), "name": pl.String, "b": Wkb()}
    _write(pl.DataFrame(schema=schema), path)
    geo = _geo_of(path)

    assert geo["primary_column"] == "a"
    assert list(geo["columns"]) == ["a", "b"]


@pytest.mark.parametrize("dtype", OTHER_DTYPES, ids=repr)
def test_writes_next_to_a_column_of_any_dtype(dtype: pl.DataType) -> None:
    df = pl.DataFrame(schema={"other": dtype, "geometry": PointXY()})
    geo = json.loads(gpl.geoparquet_metadata(df)["geo"])
    assert list(geo["columns"]) == ["geometry"]


def test_a_frame_without_geometry_has_no_metadata() -> None:
    assert gpl.geoparquet_metadata(pl.DataFrame({"id": [1]})) == {}


def _frames() -> list[pl.DataFrame]:
    xyz = [{"x": 1.0, "y": 2.0, "z": 3.0}, None]
    wkb = shapely.to_wkb(shapely.points([[1.0, 2.0], [3.0, 4.0]]))
    projjson = json.dumps({"type": "GeographicCRS", "name": "WGS 84"})
    spherical = _point('{"crs":"OGC:CRS84","edges":"spherical"}')
    return [
        pl.DataFrame({"geometry": XY}).select(pl.col("geometry").ext.to(dtype))
        for dtype in [PointXY(crs=CRS84), PointXY(), PointXY(crs=projjson), spherical]
    ] + [
        pl.DataFrame({"geometry": xyz}).select(pl.col("geometry").ext.to(PointXYZ())),
        pl.DataFrame({"geometry": wkb}, schema={"geometry": pl.Binary}).select(
            pl.col("geometry").ext.to(Wkb(crs=CRS84))
        ),
        pl.DataFrame({"geometry": [*wkb, None]}, schema={"geometry": pl.Binary}).select(
            pl.col("geometry").ext.to(Wkb())
        ),
    ]


@pytest.mark.parametrize("df", _frames())
def test_round_trips(tmp_path: Path, df: pl.DataFrame) -> None:
    path = tmp_path / "round_trip.parquet"
    _write(df, path)
    assert_frame_equal(pl.read_parquet(path), df)


def test_a_crs_comes_back_as_projjson(tmp_path: Path) -> None:
    """GeoParquet only holds PROJJSON, so that is what a file gives back."""
    path = tmp_path / "epsg.parquet"
    df = pl.DataFrame({"geometry": XY}).select(
        pl.col("geometry").ext.to(PointXY(crs="EPSG:4326"))
    )
    _write(df, path)
    back = _read(path).schema["geometry"]

    assert back != df.schema["geometry"]
    crs = json.loads(back.ext_metadata())["crs"]
    assert crs["id"] == {"authority": "EPSG", "code": 4326}


@pytest.mark.parametrize("name", SPEC_GEOMETRIES)
@pytest.mark.parametrize("encoding", ["native", "wkb"])
def test_the_spec_test_data_round_trips(
    tmp_path: Path, name: str, encoding: str
) -> None:
    df = _read(DATA / f"data-{name}-encoding_{encoding}.parquet")
    path = tmp_path / "rewritten.parquet"
    _write(df, path)
    assert_frame_equal(pl.read_parquet(path), df)


def test_sink_writes_the_same_metadata(tmp_path: Path) -> None:
    df = pl.DataFrame({"geometry": XY}).select(
        pl.col("geometry").ext.to(PointXY(crs=CRS84))
    )
    written, sunk = tmp_path / "written.parquet", tmp_path / "sunk.parquet"
    _write(df, written)
    _sink(df.lazy(), sunk)

    assert _geo_of(sunk) == _geo_of(written)
    assert_frame_equal(pl.read_parquet(sunk), df)


def test_a_lazy_frame_is_refused_before_it_runs() -> None:
    lf = pl.LazyFrame(schema={"geometry": PointXYM()})
    with pytest.raises(pl.exceptions.SchemaError, match="M values"):
        gpl.geoparquet_metadata(lf)


def _lines(crs: str | None = None) -> pl.DataFrame:
    """Two linestrings and a null one."""
    return pl.DataFrame(
        {"x": [[0.0, 2.0], [170.0, 190.0], None], "y": [[1.0, 3.0], [0.0, 1.0], None]}
    ).select(geo.line_string("x", "y", crs=crs).alias("geometry"))


def test_a_covering_is_the_bounds_of_every_geometry(tmp_path: Path) -> None:
    path = tmp_path / "covered.parquet"
    df = _lines()
    _write(df, path, covering=True)
    back = pl.read_parquet(path)

    assert back.columns == ["geometry", "geometry_bbox"]
    assert_frame_equal(
        back.select(pl.col("geometry_bbox").alias("geometry")),
        df.select(geo.bounds("geometry")),
    )
    assert _geo_of(path)["columns"]["geometry"]["covering"] == {
        "bbox": {
            bound: ["geometry_bbox", bound]
            for bound in ["xmin", "ymin", "xmax", "ymax"]
        }
    }


def test_a_covering_keeps_the_geometrys_crs(tmp_path: Path) -> None:
    path = tmp_path / "covered.parquet"
    _write(_lines(crs=CRS84), path, covering=True)
    assert pl.read_parquet(path).schema["geometry_bbox"] == BoxXY(crs=CRS84)


def test_a_covering_crosses_the_antimeridian(tmp_path: Path) -> None:
    path = tmp_path / "covered.parquet"
    _write(_lines(crs=CRS84), path, covering=True)
    boxes = pl.read_parquet(path)["geometry_bbox"].ext.storage().to_list()

    assert boxes == [
        {"xmin": 0.0, "ymin": 1.0, "xmax": 2.0, "ymax": 3.0},
        {"xmin": 170.0, "ymin": 0.0, "xmax": -170.0, "ymax": 1.0},
        None,
    ]


def test_a_covering_with_z_bounds_z(tmp_path: Path) -> None:
    path = tmp_path / "covered.parquet"
    df = pl.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]}).select(
        geo.point("x", "y", z="z").alias("geometry")
    )
    _write(df, path, covering=True)

    bbox = _geo_of(path)["columns"]["geometry"]["covering"]["bbox"]
    assert list(bbox) == ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
    assert pl.read_parquet(path)["geometry_bbox"].ext.storage().to_list() == [
        {"xmin": 1.0, "ymin": 2.0, "zmin": 3.0, "xmax": 1.0, "ymax": 2.0, "zmax": 3.0}
    ]


def test_a_multigeometry_gets_one_box_around_all_its_parts(tmp_path: Path) -> None:
    path = tmp_path / "covered.parquet"
    df = pl.DataFrame({"x": [[0.0, 5.0]], "y": [[1.0, -1.0]]}).select(
        geo.multi_point("x", "y").alias("geometry")
    )
    _write(df, path, covering=True)

    assert pl.read_parquet(path)["geometry_bbox"].ext.storage().to_list() == [
        {"xmin": 0.0, "ymin": -1.0, "xmax": 5.0, "ymax": 1.0}
    ]


def test_every_geometry_column_gets_its_own_covering(tmp_path: Path) -> None:
    path = tmp_path / "covered.parquet"
    df = _lines().with_columns(pl.col("geometry").alias("other"))
    _write(df, path, covering=True)

    assert pl.read_parquet(path).columns == [
        "geometry",
        "other",
        "geometry_bbox",
        "other_bbox",
    ]
    columns = _geo_of(path)["columns"]
    assert columns["other"]["covering"]["bbox"]["xmin"] == ["other_bbox", "xmin"]


def test_no_covering_unless_asked(tmp_path: Path) -> None:
    path = tmp_path / "plain.parquet"
    _write(_lines(), path)

    assert pl.read_parquet(path).columns == ["geometry"]
    assert "covering" not in _geo_of(path)["columns"]["geometry"]


def test_sink_writes_the_same_covering(tmp_path: Path) -> None:
    written, sunk = tmp_path / "written.parquet", tmp_path / "sunk.parquet"
    _write(_lines(crs=CRS84), written, covering=True)
    _sink(_lines(crs=CRS84).lazy(), sunk, covering=True)

    assert _geo_of(sunk) == _geo_of(written)
    assert_frame_equal(pl.read_parquet(sunk), pl.read_parquet(written))


def test_a_covering_refuses_wkb() -> None:
    df = pl.DataFrame(schema={"geometry": Wkb()})
    with pytest.raises(pl.exceptions.SchemaError, match="from_wkb"):
        gpl.with_covering(df)
    with pytest.raises(pl.exceptions.SchemaError, match="from_wkb"):
        gpl.geoparquet_metadata(df, covering=True)


def test_a_covering_refuses_a_taken_name() -> None:
    df = _lines().with_columns(geometry_bbox=pl.lit(1))
    with pytest.raises(pl.exceptions.DuplicateError, match="geometry_bbox"):
        gpl.with_covering(df)


def test_a_covering_has_to_be_added_to_be_described() -> None:
    with pytest.raises(pl.exceptions.SchemaError, match="geometry_bbox"):
        gpl.geoparquet_metadata(_lines(), covering=True)
