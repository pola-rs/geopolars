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
from tests.unit.io.test_geoparquet import DATA, OTHER_DTYPES, SPEC_GEOMETRIES

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import PolarsDataType

CRS84 = "OGC:CRS84"
XY = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]

pytestmark = pytest.mark.usefixtures("geoparquet")


def _point(metadata: str) -> PolarsDataType:
    """A `PointXY` with metadata the constructor would not write."""
    return PointType.ext_from_params(
        "geoarrow.point", PointXY().ext_storage(), metadata
    )


def _geo_of(path: Path) -> dict[str, Any]:
    return json.loads(pl.read_parquet_metadata(path)["geo"])


def _written(tmp_path: Path, dtype: PolarsDataType) -> dict[str, Any]:
    """The `geo` metadata written for a `geometry` column of `dtype`."""
    path = tmp_path / "written.parquet"
    pl.DataFrame(schema={"geometry": dtype}).write_parquet(path)
    return _geo_of(path)["columns"]["geometry"]


def test_writes_a_native_column(tmp_path: Path) -> None:
    path = tmp_path / "points.parquet"
    df = pl.DataFrame({"geometry": XY}).select(
        pl.col("geometry").ext.to(PointXY(crs="EPSG:4326"))
    )
    df.with_columns(geo.crs_to_projjson("geometry")).write_parquet(path)
    geo_metadata = _geo_of(path)

    assert geo_metadata["version"] == "1.1.0"
    assert geo_metadata["primary_column"] == "geometry"
    column = geo_metadata["columns"]["geometry"]
    assert column["encoding"] == "point"
    assert column["geometry_types"] == ["Point"]
    # Converted by `crs_to_projjson`, as GeoParquet only takes PROJJSON.
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
    ("dtype", "match"),
    [
        (PointXYM(), "cannot store M values"),
        (PolygonXYZM(), "cannot store M values"),
        (Wkt(), "geoarrow.wkt"),
        (_point('{"crs":"4326","crs_type":"srid"}'), "'srid'"),
        (_point('{"crs":"not a crs"}'), "PROJJSON"),
        (_point('{"edges":"karney"}'), "karney"),
        (_point('{"edges":"geodesic-ish"}'), "edges"),
    ],
)
def test_refuses_what_geoparquet_cannot_hold(
    tmp_path: Path, dtype: PolarsDataType, match: str
) -> None:
    path = tmp_path / "refused.parquet"
    with pytest.raises(pl.exceptions.InvalidOperationError, match=match):
        pl.DataFrame(schema={"geometry": dtype}).write_parquet(path)
    assert not path.exists()


def test_describes_every_geometry_column(tmp_path: Path) -> None:
    path = tmp_path / "columns.parquet"
    schema = {"id": pl.Int64, "a": PointXY(), "name": pl.String, "b": Wkb()}
    pl.DataFrame(schema=schema).write_parquet(path)
    geo = _geo_of(path)

    assert geo["primary_column"] == "a"
    assert list(geo["columns"]) == ["a", "b"]


@pytest.mark.parametrize("dtype", OTHER_DTYPES, ids=repr)
def test_writes_next_to_a_column_of_any_dtype(
    tmp_path: Path, dtype: pl.DataType
) -> None:
    path = tmp_path / "points.parquet"
    pl.DataFrame(schema={"other": dtype, "geometry": PointXY()}).write_parquet(path)
    geo = _geo_of(path)
    assert list(geo["columns"]) == ["geometry"]


def test_a_frame_without_geometry_has_no_metadata(tmp_path: Path) -> None:
    path = tmp_path / "plain.parquet"
    pl.DataFrame({"id": [1]}).write_parquet(path)
    assert "geo" not in pl.read_parquet_metadata(path)


def _frames() -> list[pl.DataFrame]:
    def points(dtype: PolarsDataType) -> pl.DataFrame:
        return pl.DataFrame({"geometry": XY}).select(pl.col("geometry").ext.to(dtype))

    xyz = [{"x": 1.0, "y": 2.0, "z": 3.0}, None]
    wkb = shapely.to_wkb(shapely.points([[1.0, 2.0], [3.0, 4.0]]))
    projjson = json.dumps({"type": "GeographicCRS", "name": "WGS 84"})
    spherical = _point('{"crs":"OGC:CRS84","edges":"spherical"}')
    return [
        points(PointXY(crs=CRS84)),
        points(PointXY()),
        points(PointXY(crs=projjson)),
        points(spherical),
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
    df.write_parquet(path)
    assert_frame_equal(pl.read_parquet(path), df)


def test_a_crs_comes_back_as_projjson(tmp_path: Path) -> None:
    """GeoParquet only holds PROJJSON, so that is what a file gives back."""
    path = tmp_path / "epsg.parquet"
    df = pl.DataFrame({"geometry": XY}).select(
        pl.col("geometry").ext.to(PointXY(crs="EPSG:4326"))
    )
    df.with_columns(geo.crs_to_projjson("geometry")).write_parquet(path)
    back = pl.read_parquet(path).schema["geometry"]

    assert back != df.schema["geometry"]
    crs = json.loads(back.ext_metadata())["crs"]
    assert crs["id"] == {"authority": "EPSG", "code": 4326}


@pytest.mark.parametrize("name", SPEC_GEOMETRIES)
@pytest.mark.parametrize("encoding", ["native", "wkb"])
def test_the_spec_test_data_round_trips(
    tmp_path: Path, name: str, encoding: str
) -> None:
    df = pl.read_parquet(DATA / f"data-{name}-encoding_{encoding}.parquet")
    path = tmp_path / "rewritten.parquet"
    df.write_parquet(path)
    assert_frame_equal(pl.read_parquet(path), df)


def test_sink_writes_the_same_metadata(tmp_path: Path) -> None:
    df = pl.DataFrame({"geometry": XY}).select(
        pl.col("geometry").ext.to(PointXY(crs=CRS84))
    )
    written, sunk = tmp_path / "written.parquet", tmp_path / "sunk.parquet"
    df.write_parquet(written)
    df.lazy().sink_parquet(sunk)

    assert _geo_of(sunk) == _geo_of(written)
    assert_frame_equal(pl.read_parquet(sunk), df)


def test_a_lazy_frame_is_refused_before_it_runs(tmp_path: Path) -> None:
    path = tmp_path / "refused.parquet"
    lf = pl.LazyFrame(schema={"geometry": PointXYM()})
    with pytest.raises(
        pl.exceptions.InvalidOperationError, match="cannot store M values"
    ):
        lf.sink_parquet(path)
    assert not path.exists()


def _lines(crs: str | None = None) -> pl.DataFrame:
    """Two linestrings and a null one."""
    return pl.DataFrame(
        {"x": [[0.0, 2.0], [170.0, 190.0], None], "y": [[1.0, 3.0], [0.0, 1.0], None]}
    ).select(geo.line_string("x", "y", crs=crs).alias("geometry"))


def test_a_covering_is_the_bounds_of_every_geometry(tmp_path: Path) -> None:
    path = tmp_path / "covered.parquet"
    df = _lines()
    df.pipe(gpl.with_covering).write_parquet(path)
    back = pl.read_parquet(path)

    assert back.columns == ["geometry", "bbox"]
    assert_frame_equal(
        back.select(pl.col("bbox").alias("geometry")),
        df.select(geo.bounds("geometry")),
    )
    assert _geo_of(path)["columns"]["geometry"]["covering"] == {
        "bbox": {bound: ["bbox", bound] for bound in ["xmin", "ymin", "xmax", "ymax"]}
    }


def test_a_covering_keeps_the_geometrys_crs(tmp_path: Path) -> None:
    path = tmp_path / "covered.parquet"
    _lines(crs=CRS84).pipe(gpl.with_covering).write_parquet(path)
    assert pl.read_parquet(path).schema["bbox"] == BoxXY(crs=CRS84)


def test_a_covering_crosses_the_antimeridian(tmp_path: Path) -> None:
    path = tmp_path / "covered.parquet"
    _lines(crs=CRS84).pipe(gpl.with_covering).write_parquet(path)
    boxes = pl.read_parquet(path)["bbox"].ext.storage().to_list()

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
    df.pipe(gpl.with_covering).write_parquet(path)

    bbox = _geo_of(path)["columns"]["geometry"]["covering"]["bbox"]
    assert list(bbox) == ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
    assert pl.read_parquet(path)["bbox"].ext.storage().to_list() == [
        {"xmin": 1.0, "ymin": 2.0, "zmin": 3.0, "xmax": 1.0, "ymax": 2.0, "zmax": 3.0}
    ]


def test_a_multigeometry_gets_one_box_around_all_its_parts(tmp_path: Path) -> None:
    path = tmp_path / "covered.parquet"
    df = pl.DataFrame({"x": [[0.0, 5.0]], "y": [[1.0, -1.0]]}).select(
        geo.multi_point("x", "y").alias("geometry")
    )
    df.pipe(gpl.with_covering).write_parquet(path)

    assert pl.read_parquet(path)["bbox"].ext.storage().to_list() == [
        {"xmin": 0.0, "ymin": -1.0, "xmax": 5.0, "ymax": 1.0}
    ]


def test_every_geometry_column_gets_its_own_covering(tmp_path: Path) -> None:
    path = tmp_path / "covered.parquet"
    df = _lines().with_columns(pl.col("geometry").alias("other"))
    df.pipe(gpl.with_covering).write_parquet(path)

    assert pl.read_parquet(path).columns == [
        "geometry",
        "other",
        "bbox",
        "other_bbox",
    ]
    columns = _geo_of(path)["columns"]
    assert columns["other"]["covering"]["bbox"]["xmin"] == ["other_bbox", "xmin"]


def test_no_covering_unless_asked(tmp_path: Path) -> None:
    path = tmp_path / "plain.parquet"
    _lines().write_parquet(path)

    assert pl.read_parquet(path).columns == ["geometry"]
    assert "covering" not in _geo_of(path)["columns"]["geometry"]


def test_a_box_that_covers_no_geometry_is_refused(tmp_path: Path) -> None:
    """A box is GeoArrow, but no GeoParquet geometry: unless it is a covering, it is refused."""
    path = tmp_path / "boxed.parquet"
    df = _lines().with_columns(geo.bounds("geometry").alias("box"))
    with pytest.raises(
        pl.exceptions.InvalidOperationError, match="cannot write box column 'box'"
    ):
        df.write_parquet(path)
    assert not path.exists()


def test_a_covering_can_be_converted_after_it_is_added(tmp_path: Path) -> None:
    """The covering takes the geometry's CRS, so it needs converting too."""
    path = tmp_path / "covered.parquet"
    df = (
        _lines(crs="EPSG:4326")
        .pipe(gpl.with_covering)
        .with_columns(geo.crs_to_projjson("geometry"), geo.crs_to_projjson("bbox"))
    )
    df.write_parquet(path)

    covering = _geo_of(path)["columns"]["geometry"]["covering"]
    assert covering["bbox"]["xmin"] == ["bbox", "xmin"]


def test_sink_writes_the_same_covering(tmp_path: Path) -> None:
    written, sunk = tmp_path / "written.parquet", tmp_path / "sunk.parquet"
    _lines(crs=CRS84).pipe(gpl.with_covering).write_parquet(written)
    _lines(crs=CRS84).lazy().pipe(gpl.with_covering).sink_parquet(sunk)

    assert _geo_of(sunk) == _geo_of(written)
    assert_frame_equal(pl.read_parquet(sunk), pl.read_parquet(written))


def test_a_covering_refuses_wkb() -> None:
    df = pl.DataFrame(schema={"geometry": Wkb()})
    with pytest.raises(pl.exceptions.SchemaError, match="from_wkb"):
        gpl.with_covering(df)


def test_a_covering_refuses_a_taken_name() -> None:
    df = _lines().with_columns(bbox=pl.lit(1))
    with pytest.raises(pl.exceptions.DuplicateError, match="bbox"):
        gpl.with_covering(df)
