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

import geopolars as gpl
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
    COVERING_FIELDS_OUT_OF_ORDER,
    EPOCH_REFUSED_BY_THE_WRAPPER,
    OTHER_ERRORS_BY_THE_WRAPPER,
    REFUSED_BY_THE_WRAPPER,
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


@COVERING_FIELDS_OUT_OF_ORDER
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


@OTHER_ERRORS_BY_THE_WRAPPER
def test_files_that_disagree_are_refused(parquet: Parquet, tmp_path: Path) -> None:
    _write(tmp_path / "a.parquet", _geo(crs=None))
    _write(tmp_path / "b.parquet", _geo())

    with pytest.raises(pl.exceptions.SchemaError, match="data type mismatch"):
        parquet.scan(tmp_path / "*.parquet").collect()


def test_the_wrapper_can_trust_the_first_file(tmp_path: Path) -> None:
    _write(tmp_path / "a.parquet", _geo(crs=None))
    _write(tmp_path / "b.parquet", _geo())
    source = tmp_path / "*.parquet"

    trusted = pl.read_parquet(source).pipe(
        gpl.from_geoparquet, source, check_every_file=False
    )
    assert trusted.schema["geometry"] == PointXY()


@OTHER_ERRORS_BY_THE_WRAPPER
def test_files_without_metadata_among_geoparquet_are_refused(
    parquet: Parquet, tmp_path: Path
) -> None:
    _write(tmp_path / "a.parquet", _geo())
    _write(tmp_path / "b.parquet", None)

    with pytest.raises(pl.exceptions.SchemaError, match="data type mismatch"):
        parquet.scan(tmp_path / "*.parquet").collect()


def test_plain_parquet_is_read_as_is(parquet: Parquet, tmp_path: Path) -> None:
    path = _write(tmp_path / "plain.parquet", None)
    assert_frame_equal(parquet.read(path), pl.read_parquet(path))


@REFUSED_BY_THE_WRAPPER
def test_a_geometry_column_the_file_lacks_is_ignored(
    parquet: Parquet, tmp_path: Path
) -> None:
    path = _write(tmp_path / "points.parquet", _geo(), POINTS.select("id"))
    with pytest.warns(UserWarning, match="column not found"):
        assert parquet.read(path).columns == ["id"]


@EPOCH_REFUSED_BY_THE_WRAPPER
def test_an_epoch_is_kept(parquet: Parquet, tmp_path: Path) -> None:
    path = _write(tmp_path / "points.parquet", _geo(epoch=2021.47))
    dtype = parquet.read(path).schema["geometry"]
    assert dtype.ext_metadata() == '{"crs":"OGC:CRS84","epoch":2021.47}'


@REFUSED_BY_THE_WRAPPER
def test_a_primary_column_it_lacks_is_ignored(parquet: Parquet, tmp_path: Path) -> None:
    path = _write(tmp_path / "points.parquet", _geo() | {"primary_column": "elsewhere"})
    assert parquet.read(path).schema["geometry"] == PointXY(crs="OGC:CRS84")


@REFUSED_BY_THE_WRAPPER
@pytest.mark.parametrize(
    ("geo_metadata", "match"),
    [
        (_geo() | {"version": "2.0.0"}, "unsupported version '2.0.0'"),
        (_geo(encoding="geometry"), "unknown encoding 'geometry'"),
    ],
)
def test_metadata_it_cannot_honour_is_read_as_plain_storage(
    parquet: Parquet, tmp_path: Path, geo_metadata: dict[str, Any], match: str
) -> None:
    path = _write(tmp_path / "points.parquet", geo_metadata)
    with pytest.warns(UserWarning, match=match):
        df = parquet.read(path)

    assert df.schema["geometry"] == POINTS.schema["geometry"]
    repaired = df.select(pl.col("geometry").ext.to(PointXY(crs="OGC:CRS84")))
    assert repaired.schema["geometry"] == PointXY(crs="OGC:CRS84")


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


@REFUSED_BY_THE_WRAPPER
@pytest.mark.parametrize(
    ("encoding", "df", "repair"),
    [
        # Points, but stored as a linestring.
        (
            "point",
            pl.DataFrame({"geometry": [[{"x": 1.0, "y": 2.0}]]}),
            pl.col("geometry").list.first().ext.to(PointXY(crs="OGC:CRS84")),
        ),
        (
            "WKB",
            pl.DataFrame({"geometry": ["POINT (1 2)"]}),
            geo.from_wkt("geometry", PointXY, crs="OGC:CRS84"),
        ),
    ],
)
def test_storage_that_does_not_match_the_encoding_is_read_as_is(
    parquet: Parquet, tmp_path: Path, encoding: str, df: pl.DataFrame, repair: pl.Expr
) -> None:
    path = _write(tmp_path / "points.parquet", _geo(encoding=encoding), df)
    with pytest.warns(UserWarning, match=f"does not match encoding '{encoding}'"):
        back = parquet.read(path)

    assert back.schema == df.schema
    assert back.select(repair).to_dicts() == [{"geometry": {"x": 1.0, "y": 2.0}}]


def _covering(column: str = "bbox", **paths: list[str]) -> dict[str, Any]:
    """A bbox covering in `column`, with `paths` replacing the default ones."""
    bounds = ["xmin", "ymin", "xmax", "ymax"]
    return {"bbox": {bound: [column, bound] for bound in bounds} | paths}


_BOUNDS = {"xmin": 1.0, "ymin": 2.0, "xmax": 3.0, "ymax": 4.0}


def test_a_covering_reads_as_a_box_in_the_geometrys_crs(
    parquet: Parquet, tmp_path: Path
) -> None:
    df = POINTS.with_columns(bbox=pl.lit(_BOUNDS))
    path = _write(tmp_path / "points.parquet", _geo(covering=_covering()), df)
    back = parquet.read(path)

    assert back.schema["bbox"] == BoxXY(crs="OGC:CRS84")
    assert back["bbox"].ext.storage().to_list() == [_BOUNDS, _BOUNDS]


def test_a_covering_without_any_bound_can_be_made_a_null_box(
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
    back = parquet.read(path).with_columns(geo.null_out_invalid_geoarrow("bbox"))

    assert df["bbox"].null_count() == 0
    assert back["bbox"].ext.storage().to_list() == [_BOUNDS, None]


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


@REFUSED_BY_THE_WRAPPER
@pytest.mark.parametrize(
    "covering",
    [
        _covering(ymax=["elsewhere", "ymax"]),
        _covering(ymax=["bbox", "maxy"]),
        _covering(zmin=["bbox", "zmin"]),
        _covering(xmin=["bbox"]),
    ],
)
def test_a_malformed_covering_is_read_as_a_struct(
    parquet: Parquet, tmp_path: Path, covering: dict[str, Any]
) -> None:
    df = POINTS.with_columns(bbox=pl.lit(_BOUNDS))
    path = _write(tmp_path / "points.parquet", _geo(covering=covering), df)
    with pytest.warns(UserWarning, match="invalid bbox paths"):
        back = parquet.read(path)

    assert back.schema["bbox"] == df.schema["bbox"]
    bound = pl.col("bbox").struct.field
    repaired = back.select(
        geo.box(
            bound("xmin"), bound("ymin"), bound("xmax"), bound("ymax"), crs="OGC:CRS84"
        )
    )
    assert repaired.to_series().ext.storage().to_list() == [_BOUNDS, _BOUNDS]


@REFUSED_BY_THE_WRAPPER
def test_a_covering_that_is_not_bounds_is_read_as_a_struct(
    parquet: Parquet, tmp_path: Path
) -> None:
    df = POINTS.with_columns(bbox=pl.lit({"xmin": 1.0, "ymin": 2.0}))
    path = _write(tmp_path / "points.parquet", _geo(covering=_covering()), df)
    with pytest.warns(UserWarning, match="is not a struct of Float64 fields"):
        back = parquet.read(path)

    assert back.schema["bbox"] == df.schema["bbox"]
    # Too few bounds to repair from, but the geometries can be bounded again.
    repaired = back.with_columns(geo.bounds("geometry").alias("bbox"))
    assert repaired.schema["bbox"] == BoxXY(crs="OGC:CRS84")
