"""Reading GeoParquet.

Tests against files from https://github.com/opengeospatial/geoparquet/tree/v1.1.0+p1,
and files with manually-input geo-metadata (plain storage)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import shapely
from polars.testing import assert_frame_equal

from geopolars import geo
from geopolars.datatypes import (
    BoxXY,
    BoxXYZ,
    GeoArrowType,
    LineStringXY,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PointXY,
    PolygonXY,
    PolygonXYZ,
    Wkb,
)
from tests.unit.io.conftest import (
    NO_CHECK_EVERY_FILE,
    NO_COVERINGS,
    NOT_REFUSED,
    Parquet,
)

DATA = Path(__file__).parents[2] / "data" / "geoparquet"

#: The geometries `data-<name>-*` holds.
SPEC_GEOMETRIES: dict[str, type[GeoArrowType]] = {
    "point": PointXY,
    "linestring": LineStringXY,
    "polygon": PolygonXY,
    "multipoint": MultiPointXY,
    "multilinestring": MultiLineStringXY,
    "multipolygon": MultiPolygonXY,
}


#: Dtypes of columns next to the geometries, that the plugin's Polars needs a feature for.
OTHER_DTYPES: list[pl.DataType] = [
    pl.Int8(),
    pl.Int16(),
    pl.Int128(),
    pl.UInt8(),
    pl.UInt16(),
    pl.UInt128(),
    pl.Float16(),
    pl.Decimal(10, 2),
    pl.Date(),
    pl.Datetime("us", "Europe/Amsterdam"),
    pl.Duration(),
    pl.Time(),
    pl.Categorical(),
    pl.Enum(["a", "b"]),
    pl.Array(pl.Int64, 2),
]


@pytest.mark.parametrize(("name", "geometry"), SPEC_GEOMETRIES.items())
def test_the_spec_test_data_reads_as_its_wkt(
    parquet: Parquet, name: str, geometry: type[GeoArrowType]
) -> None:
    """Native and WKB files without a `crs`, each against the same geometries as WKT."""
    expected = pl.read_csv(DATA / f"data-{name}-wkt.csv").with_columns(
        geo.from_wkt("geometry", geometry, crs="OGC:CRS84")
    )

    native = parquet.read(DATA / f"data-{name}-encoding_native.parquet")
    assert native.schema["geometry"] == geometry(crs="OGC:CRS84")
    assert_frame_equal(native, expected)

    wkb = parquet.read(DATA / f"data-{name}-encoding_wkb.parquet")
    assert wkb.schema["geometry"] == Wkb(crs="OGC:CRS84")
    assert_frame_equal(wkb.with_columns(geo.from_wkb("geometry", geometry)), expected)


@NO_COVERINGS
def test_the_spec_example_reads(parquet: Parquet) -> None:
    """WKB of polygons and multipolygons, a PROJJSON CRS and a `bbox` covering."""
    df = parquet.read(DATA / "example.parquet")

    wkb = df.schema["geometry"]
    assert isinstance(wkb, Wkb)
    assert wkb._declares_crs()
    # The covering is a box, in the geometry's CRS,
    # even though the file has its bounds out of order (xmax, xmin, ...).
    assert isinstance(df.schema["bbox"], BoxXY)
    assert df.schema["bbox"].ext_metadata() == wkb.ext_metadata()
    first = df["bbox"].ext.storage()[0]
    assert list(first) == ["xmin", "ymin", "xmax", "ymax"]
    assert first["xmin"] < first["xmax"]

    polygons = df.select(geo.from_wkb("geometry", MultiPolygonXY))
    # Polygons are promoted, and the CRS comes along.
    assert polygons.schema["geometry"].ext_metadata() == wkb.ext_metadata()
    assert polygons["geometry"].null_count() == 0


POINTS = pl.DataFrame(
    {"id": [1, 2], "geometry": [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]}
)


def _geo(**column: Any) -> dict[str, Any]:
    """`geo` metadata for a `geometry` column of points, plus `column`."""
    return {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {"encoding": "point", "geometry_types": ["Point"], **column}
        },
    }


def _write(
    path: Path, geo_metadata: dict[str, Any] | None, df: pl.DataFrame = POINTS
) -> Path:
    metadata = None if geo_metadata is None else {"geo": json.dumps(geo_metadata)}
    df.write_parquet(path, metadata=metadata)
    return path


def test_a_missing_crs_is_crs84(parquet: Parquet, tmp_path: Path) -> None:
    path = _write(tmp_path / "points.parquet", _geo())
    back = parquet.read(path)

    assert back.schema["geometry"] == PointXY(crs="OGC:CRS84")
    assert_frame_equal(back.with_columns(pl.col("geometry").ext.storage()), POINTS)


def test_a_null_crs_is_no_crs(parquet: Parquet, tmp_path: Path) -> None:
    path = _write(tmp_path / "points.parquet", _geo(crs=None))
    assert parquet.read(path).schema["geometry"] == PointXY()


def test_a_projjson_crs_is_the_crs(parquet: Parquet, tmp_path: Path) -> None:
    projjson = {"type": "GeographicCRS", "name": "WGS 84"}
    path = _write(tmp_path / "points.parquet", _geo(crs=projjson))
    assert parquet.read(path).schema["geometry"] == PointXY(crs=json.dumps(projjson))


def test_the_dimension_comes_from_the_storage(parquet: Parquet, tmp_path: Path) -> None:
    ring = [{"x": 0.0, "y": 0.0, "z": 1.0}, {"x": 1.0, "y": 0.0, "z": 1.0}]
    df = pl.DataFrame({"geometry": [[ring], None]})
    geo_metadata = _geo(crs=None)
    geo_metadata["columns"]["geometry"] |= {
        "encoding": "polygon",
        "geometry_types": [],
    }
    path = _write(tmp_path / "polygons.parquet", geo_metadata, df)
    back = parquet.read(path)

    assert back.schema["geometry"] == PolygonXYZ()
    assert back["geometry"].is_null().to_list() == [False, True]


def test_wkb_stays_wkb_with_the_crs(parquet: Parquet, tmp_path: Path) -> None:
    df = pl.DataFrame(
        {"geometry": shapely.to_wkb(shapely.points([[1.0, 2.0], [3.0, 4.0]]))},
        schema={"geometry": pl.Binary},
    )
    geo_metadata = _geo()
    geo_metadata["columns"]["geometry"]["encoding"] = "WKB"
    path = _write(tmp_path / "wkb.parquet", geo_metadata, df)
    back = parquet.read(path)

    assert back.schema["geometry"] == Wkb(crs="OGC:CRS84")
    points = back.select(geo.from_wkb("geometry", PointXY))
    assert points.schema["geometry"] == PointXY(crs="OGC:CRS84")
    assert_frame_equal(
        points.select(pl.col("geometry").ext.storage()), POINTS.select("geometry")
    )


@pytest.mark.parametrize("dtype", OTHER_DTYPES, ids=repr)
def test_reads_next_to_a_column_of_any_dtype(
    parquet: Parquet, tmp_path: Path, dtype: pl.DataType
) -> None:
    df = pl.DataFrame(schema={"other": dtype, "geometry": POINTS.schema["geometry"]})
    path = _write(tmp_path / "points.parquet", _geo(), df)
    back = parquet.read(path)

    assert back.schema["other"] == dtype
    assert back.schema["geometry"] == PointXY(crs="OGC:CRS84")


def test_a_file_polars_wrote_keeps_its_dtype(parquet: Parquet, tmp_path: Path) -> None:
    """Such a file already has the extension type, so nothing needs converting."""
    df = POINTS.with_columns(pl.col("geometry").ext.to(PointXY(crs="OGC:CRS84")))
    path = _write(tmp_path / "points.parquet", _geo(), df)
    assert_frame_equal(pl.read_parquet(path), df)
    assert_frame_equal(parquet.read(path), df)


def test_the_scan_stays_lazy(parquet: Parquet, tmp_path: Path) -> None:
    path = _write(tmp_path / "points.parquet", _geo())
    lf = parquet.scan(path)

    assert lf.select("id").collect()["id"].to_list() == [1, 2]
    filtered = lf.filter(pl.col("id") == 2).collect()
    assert filtered.schema["geometry"] == PointXY(crs="OGC:CRS84")
    assert filtered.height == 1


def test_eager_and_lazy_agree(parquet: Parquet, tmp_path: Path) -> None:
    path = _write(tmp_path / "points.parquet", _geo())
    assert_frame_equal(parquet.read(path), parquet.scan(path).collect())


def test_files_that_agree_read_together(parquet: Parquet, tmp_path: Path) -> None:
    """Only the dtypes have to agree: a `bbox` is per file."""
    _write(tmp_path / "a.parquet", _geo(bbox=[1, 2, 1, 2]))
    _write(tmp_path / "b.parquet", _geo(bbox=[3, 4, 3, 4]))

    for source in [
        tmp_path / "*.parquet",
        [tmp_path / "a.parquet", tmp_path / "b.parquet"],
    ]:
        back = parquet.read(source)
        assert back.schema["geometry"] == PointXY(crs="OGC:CRS84")
        assert back.height == 4


@NO_CHECK_EVERY_FILE
def test_files_that_disagree_are_refused(parquet: Parquet, tmp_path: Path) -> None:
    _write(tmp_path / "a.parquet", _geo(crs=None))
    _write(tmp_path / "b.parquet", _geo())
    source = tmp_path / "*.parquet"

    with pytest.raises(pl.exceptions.SchemaError, match="check_every_file"):
        parquet.scan(source)
    trusted = parquet.read(source, check_every_file=False)
    assert trusted.schema["geometry"] == PointXY()


@NO_CHECK_EVERY_FILE
def test_files_without_metadata_among_geoparquet_are_refused(
    parquet: Parquet, tmp_path: Path
) -> None:
    _write(tmp_path / "a.parquet", _geo())
    _write(tmp_path / "b.parquet", None)

    with pytest.raises(pl.exceptions.SchemaError, match="no GeoParquet metadata"):
        parquet.scan(tmp_path / "*.parquet")


def test_plain_parquet_is_read_as_is(parquet: Parquet, tmp_path: Path) -> None:
    path = _write(tmp_path / "plain.parquet", None)
    assert_frame_equal(parquet.read(path), pl.read_parquet(path))


@NOT_REFUSED
def test_a_geometry_column_the_file_lacks_is_refused(
    parquet: Parquet, tmp_path: Path
) -> None:
    path = _write(tmp_path / "points.parquet", _geo(), POINTS.select("id"))
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match="geometry"):
        parquet.scan(path)


@NOT_REFUSED
@pytest.mark.parametrize(
    ("geo_metadata", "match"),
    [
        (_geo(epoch=2021.47), "epoch"),
        (_geo() | {"version": "2.0.0"}, "2.0.0"),
        (_geo(encoding="geometry"), "encoding"),
        (_geo() | {"primary_column": "elsewhere"}, "elsewhere"),
    ],
)
def test_metadata_it_cannot_honour_is_refused(
    parquet: Parquet, tmp_path: Path, geo_metadata: dict[str, Any], match: str
) -> None:
    path = _write(tmp_path / "points.parquet", geo_metadata)
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        parquet.scan(path)


def test_fields_it_does_not_use_are_ignored(parquet: Parquet, tmp_path: Path) -> None:
    """Readers must not reject what a newer minor version or a vendor adds."""
    geo_metadata = _geo(
        orientation="counterclockwise",
        bbox=[1, 2, 3, 4],
        vendor={"a": 1},
    ) | {"creator": {"library": "elsewhere"}}
    path = _write(tmp_path / "points.parquet", geo_metadata)
    assert parquet.read(path).schema["geometry"] == PointXY(crs="OGC:CRS84")


def test_spherical_edges_are_kept(parquet: Parquet, tmp_path: Path) -> None:
    path = _write(tmp_path / "points.parquet", _geo(edges="spherical"))
    dtype = parquet.read(path).schema["geometry"]
    assert dtype.ext_metadata() == '{"crs":"OGC:CRS84","edges":"spherical"}'


@NOT_REFUSED
@pytest.mark.parametrize(
    ("encoding", "df"),
    [
        # Points, but stored as a linestring.
        ("point", pl.DataFrame({"geometry": [[{"x": 1.0, "y": 2.0}]]})),
        ("WKB", pl.DataFrame({"geometry": ["POINT (1 2)"]})),
    ],
)
def test_storage_that_does_not_match_the_encoding_is_refused(
    parquet: Parquet, tmp_path: Path, encoding: str, df: pl.DataFrame
) -> None:
    path = _write(tmp_path / "points.parquet", _geo(encoding=encoding), df)
    with pytest.raises(pl.exceptions.SchemaError, match="stored as"):
        parquet.scan(path)


def _covering(column: str = "bbox", **paths: list[str]) -> dict[str, Any]:
    """A bbox covering in `column`, with `paths` replacing the default ones."""
    bounds = ["xmin", "ymin", "xmax", "ymax"]
    return {"bbox": {bound: [column, bound] for bound in bounds} | paths}


_BOUNDS = {"xmin": 1.0, "ymin": 2.0, "xmax": 3.0, "ymax": 4.0}


@NO_COVERINGS
def test_a_covering_reads_as_a_box_in_the_geometrys_crs(
    parquet: Parquet, tmp_path: Path
) -> None:
    df = POINTS.with_columns(bbox=pl.lit(_BOUNDS))
    path = _write(tmp_path / "points.parquet", _geo(covering=_covering()), df)
    back = parquet.read(path)

    assert back.schema["bbox"] == BoxXY(crs="OGC:CRS84")
    assert back["bbox"].ext.storage().to_list() == [_BOUNDS, _BOUNDS]


@NO_COVERINGS
def test_a_covering_without_any_bound_is_a_null_box(
    parquet: Parquet, tmp_path: Path
) -> None:
    """The way geopandas writes the bbox of a null geometry."""
    df = POINTS.with_columns(
        geometry=pl.when(pl.col("id") == 1).then("geometry"),
        bbox=pl.when(pl.col("id") == 1)
        .then(pl.lit(_BOUNDS))
        .otherwise(pl.struct(**{bound: pl.lit(None, pl.Float64) for bound in _BOUNDS})),
    )
    path = _write(tmp_path / "points.parquet", _geo(covering=_covering()), df)
    back = parquet.read(path)

    assert df["bbox"].null_count() == 0
    assert back["bbox"].ext.storage().to_list() == [_BOUNDS, None]


@NO_COVERINGS
def test_a_covering_with_z_reads_as_a_3d_box(parquet: Parquet, tmp_path: Path) -> None:
    bounds = {
        "xmin": 1.0,
        "ymin": 2.0,
        "zmin": 5.0,
        "xmax": 3.0,
        "ymax": 4.0,
        "zmax": 6.0,
    }
    df = POINTS.with_columns(bbox=pl.lit(bounds))
    covering = _covering(zmin=["bbox", "zmin"], zmax=["bbox", "zmax"])
    path = _write(tmp_path / "points.parquet", _geo(covering=covering), df)

    assert parquet.read(path).schema["bbox"] == BoxXYZ(crs="OGC:CRS84")


def test_a_covering_whose_column_is_missing_is_ignored(
    parquet: Parquet, tmp_path: Path
) -> None:
    path = _write(tmp_path / "points.parquet", _geo(covering=_covering()))
    assert parquet.read(path).columns == ["id", "geometry"]


@NO_COVERINGS
@pytest.mark.parametrize(
    ("covering", "match"),
    [
        (_covering(ymax=["elsewhere", "ymax"]), "same column"),
        (_covering(ymax=["bbox", "maxy"]), "its field `ymax`"),
        (_covering(zmin=["bbox", "zmin"]), "`zmin` and `zmax`"),
        (_covering(xmin=["bbox"]), "invalid GeoParquet metadata"),
    ],
)
def test_a_malformed_covering_is_refused(
    parquet: Parquet, tmp_path: Path, covering: dict[str, Any], match: str
) -> None:
    df = POINTS.with_columns(bbox=pl.lit(_BOUNDS))
    path = _write(tmp_path / "points.parquet", _geo(covering=covering), df)
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        parquet.scan(path)


@NO_COVERINGS
def test_a_covering_that_is_not_bounds_is_refused(
    parquet: Parquet, tmp_path: Path
) -> None:
    df = POINTS.with_columns(bbox=pl.lit({"xmin": 1.0, "ymin": 2.0}))
    path = _write(tmp_path / "points.parquet", _geo(covering=_covering()), df)
    with pytest.raises(pl.exceptions.SchemaError, match="xmin/ymin/xmax/ymax"):
        parquet.scan(path)
