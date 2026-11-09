"""`to_crs`.

EPSG:4326 to EPSG:3857 and EPSG:4979 to EPSG:4978 are worked out by hand below,
so their expected values do not come from PROJ.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import polars as pl
import pyarrow as pa
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal

from geopolars import geo
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    LineStringXYZ,
    MultiLineStringXY,
    MultiLineStringXYZ,
    MultiPointXY,
    MultiPointXYZ,
    MultiPolygonXY,
    MultiPolygonXYZ,
    PointXY,
    PointXYZ,
    PolygonXY,
    PolygonXYZ,
)

WGS84 = "EPSG:4326"
WGS84_3D = "EPSG:4979"
GEOCENTRIC = "EPSG:4978"
WEB_MERCATOR = "EPSG:3857"
# New York Long Island in US survey feet, with NAVD88 heights in US survey feet.
NY_FEET = "EPSG:2263+6360"
# NAD83 longitude/latitude, with NAVD88 heights in metres.
NAD83_METRES = "EPSG:4269+5703"
WORLD_MERCATOR = "EPSG:3395"
INF = float("inf")
# The WGS 84 ellipsoid. Web Mercator projects onto a sphere of its semi-major axis.
SEMI_MAJOR_AXIS = 6378137.0
ECCENTRICITY_SQUARED = (2 - 1 / 298.257223563) / 298.257223563
# In metres, far below what either worked-out projection can be off by.
TOLERANCE = 1e-6

LonLat = tuple[float, float]


def _points(crs: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [5.0, 5.0, None],
            "y": [52.0, 52.0, None],
            "z": [0.0, 1000.0, None],
            "m": [7.0, 8.0, None],
        }
    ).select(geo.point("x", "y", "z", "m", crs=crs).alias("point"))


def test_x_and_y_that_depend_on_height_get_it_from_z() -> None:
    out = _points(WGS84_3D).select(geo.to_crs("point", GEOCENTRIC))

    x, y, z, m = out["point"].ext.storage().struct.unnest().get_columns()
    assert x.to_list()[:2] == pytest.approx([3919986.754103, 3920600.072800])
    assert y.to_list()[:2] == pytest.approx([342954.402156, 343008.060589])
    assert z.to_list()[:2] == pytest.approx([5002803.345483, 5003591.356236])
    assert_series_equal(m, pl.Series("m", [7.0, 8.0, None]), check_names=False)
    assert out["point"].null_count() == 1


def test_z_is_converted_to_the_unit_of_the_target_crs() -> None:
    lines = pl.DataFrame(
        {"x": [[1e6, 1e6]], "y": [[2e5, 2e5]], "z": [[100.0, 0.0]]}
    ).select(geo.line_string("x", "y", "z", crs=NY_FEET).alias("line"))

    out = lines.select(geo.z(geo.to_crs("line", NAD83_METRES)))

    assert out["line"].to_list() == [pytest.approx([30.480061, 0.0])]


@pytest.mark.parametrize(
    ("source", "target", "flag"),
    [
        (WGS84_3D, WEB_MERCATOR, "allow_projection_with_unscaled_z"),
        (WGS84, WEB_MERCATOR, "allow_projection_with_unscaled_z"),
        (WGS84, WGS84_3D, "assume_z_in_target_crs"),
    ],
)
def test_a_z_that_one_crs_does_not_specify_needs_a_flag(
    source: str, target: str, flag: str
) -> None:
    points = _points(source)

    with pytest.raises(ComputeError, match=flag):
        points.lazy().select(geo.to_crs("point", target)).collect_schema()

    out = points.select(geo.to_crs("point", target, **{flag: True}))
    assert_series_equal(
        out.select(geo.z("point"))["point"], points.select(geo.z("point"))["point"]
    )


def test_a_geometry_without_z_ignores_the_flags() -> None:
    points = pl.DataFrame({"x": [5.0], "y": [52.0]}).select(
        geo.point("x", "y", crs=WGS84).alias("point")
    )

    plain = points.select(geo.to_crs("point", WEB_MERCATOR))
    flagged = points.select(
        geo.to_crs(
            "point",
            WEB_MERCATOR,
            allow_projection_with_unscaled_z=True,
            assume_z_in_target_crs=True,
        )
    )

    assert plain.equals(flagged)


def test_a_bound_crs_reprojects_a_geometry_without_z() -> None:
    # A BoundCRS, of which PROJ cannot tell whether it specifies z.
    bessel = "+proj=longlat +ellps=bessel +towgs84=565.4,50.3,465.6 +no_defs +type=crs"
    points = pl.DataFrame({"x": [5.0], "y": [52.0]}).select(
        geo.point("x", "y", crs=bessel).alias("point")
    )

    out = points.select(geo.to_crs("point", WEB_MERCATOR))

    # (5, 52) in Web Mercator, give or take the datum shift.
    xy = out.select(geo.x("point"), geo.y("point").alias("y")).row(0)
    assert xy == pytest.approx((556597.5, 6800125.5), abs=200)


def _web_mercator(lon: float, lat: float) -> LonLat:
    return (
        SEMI_MAJOR_AXIS * math.radians(lon),
        SEMI_MAJOR_AXIS * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)),
    )


def _geocentric(lon: float, lat: float, height: float) -> tuple[float, float, float]:
    lam, phi = math.radians(lon), math.radians(lat)
    n = SEMI_MAJOR_AXIS / math.sqrt(1 - ECCENTRICITY_SQUARED * math.sin(phi) ** 2)
    return (
        (n + height) * math.cos(phi) * math.cos(lam),
        (n + height) * math.cos(phi) * math.sin(lam),
        (n * (1 - ECCENTRICITY_SQUARED) + height) * math.sin(phi),
    )


def _points_at(lon_lat: list[LonLat | None], crs: str = WGS84) -> pl.DataFrame:
    """A point column, with a null point for each `None`."""
    xs = [None if p is None else p[0] for p in lon_lat]
    ys = [None if p is None else p[1] for p in lon_lat]
    return pl.DataFrame(
        {"x": xs, "y": ys}, schema={"x": pl.Float64, "y": pl.Float64}
    ).select(geo.point("x", "y", crs=crs).alias("point"))


def _xy(df: pl.DataFrame, name: str = "point") -> list[LonLat | None]:
    xs = df.select(geo.x(name)).to_series()
    ys = df.select(geo.y(name)).to_series()
    return [None if x is None else (x, y) for x, y in zip(xs, ys)]


def _in_web_mercator(lon_lat: list[LonLat | None]) -> list[object]:
    return [
        None if p is None else pytest.approx(_web_mercator(*p), abs=TOLERANCE)
        for p in lon_lat
    ]


NULL_PATTERNS = {
    "runs of one": [True, False, True, False, True],
    "null first": [False, True, True],
    "null last": [True, True, False],
    "all null": [False, False, False],
    "no nulls": [True, True, True],
}


@pytest.mark.parametrize("valid", list(NULL_PATTERNS.values()), ids=list(NULL_PATTERNS))
def test_each_run_of_points_between_nulls_is_reprojected(valid: list[bool]) -> None:
    lon_lat = [(4.0 + i, 50.0 + i) if v else None for i, v in enumerate(valid)]

    out = _points_at(lon_lat).select(geo.to_crs("point", WEB_MERCATOR))

    assert _xy(out) == _in_web_mercator(lon_lat)


@pytest.mark.parametrize(("offset", "length"), [(3, 13), (9, 1), (5, 0), (1, 19)])
def test_a_sliced_point_column_reprojects_its_own_rows(
    offset: int, length: int
) -> None:
    """The slices start partway through a byte of the validity bitmap."""
    lon_lat = [None if i % 3 == 1 else (4.0 + i, 50.0 + i / 10) for i in range(20)]
    sliced = _points_at(lon_lat).slice(offset, length)

    out = sliced.select(geo.to_crs("point", WEB_MERCATOR))

    assert _xy(out) == _in_web_mercator(lon_lat[offset : offset + length])


def test_a_column_of_many_chunks_reprojects_each() -> None:
    lon_lat = [None if i % 4 == 0 else (4.0 + i / 10, 50.0 + i / 10) for i in range(40)]
    whole = _points_at(lon_lat)
    chunked = pl.concat([whole.slice(k, 7) for k in range(0, 40, 7)], rechunk=False)
    assert chunked.n_chunks() > 1

    out = chunked.select(geo.to_crs("point", WEB_MERCATOR))

    assert _xy(out) == _in_web_mercator(lon_lat)


def test_a_null_point_is_left_alone_whatever_it_holds() -> None:
    """Under the null is a latitude past the pole, which PROJ refuses."""
    storage = pa.StructArray.from_arrays(
        [pa.array([5.0, 5.0, 6.0]), pa.array([52.0, 95.0, 53.0])],
        names=["x", "y"],
        mask=pa.array([False, True, False]),
    )
    points = pl.DataFrame(pl.Series("point", storage).ext.to(PointXY(crs=WGS84)))

    out = points.select(geo.to_crs("point", WEB_MERCATOR))

    assert _xy(out) == _in_web_mercator([(5.0, 52.0), None, (6.0, 53.0)])
    # `None` if polars masks the field too, but never what PROJ makes of 95.
    assert out["point"].ext.storage().struct.field("y")[1] in (95.0, None)


def _with_two_failing(kind: str) -> pl.DataFrame:
    """A column named `point`, whatever its kind, where PROJ fails on (6, 95) and then on (8, 96)."""
    lon_lat = [(5.0, 52.0), (6.0, 95.0), (7.0, 52.0), (8.0, 96.0)]
    if kind == "point":
        return _points_at(lon_lat)
    if kind == "point after nulls":
        return _points_at([None, None, *lon_lat[:2], None, *lon_lat[2:]])
    xs, ys = zip(*lon_lat)
    return pl.DataFrame({"x": [list(xs)], "y": [list(ys)]}).select(
        geo.line_string("x", "y", crs=WGS84).alias("point")
    )


@pytest.mark.parametrize("kind", ["point", "point after nulls", "linestring"])
def test_the_error_names_the_first_coordinate_proj_fails_on(kind: str) -> None:
    with pytest.raises(ComputeError, match=r"failed to reproject \(6, 95\)") as error:
        _with_two_failing(kind).select(geo.to_crs("point", WEB_MERCATOR))

    assert "(8, 96)" not in str(error.value)


@pytest.mark.parametrize("infinite", [(INF, 52.0), (5.0, INF)], ids=["x", "y"])
@pytest.mark.parametrize("with_null", [False, True], ids=["", "with a null"])
def test_an_infinite_coordinate_comes_back_infinite_without_failing(
    infinite: LonLat, with_null: bool
) -> None:
    lon_lat: list[LonLat | None] = [(5.0, 52.0), infinite, (6.0, 53.0)]
    if with_null:
        lon_lat.append(None)

    out = _xy(_points_at(lon_lat).select(geo.to_crs("point", WEB_MERCATOR)))

    assert out[1] is not None
    assert out[1][0] == INF
    assert [out[0], out[2]] == _in_web_mercator([(5.0, 52.0), (6.0, 53.0)])


def _ring(dims: str, dx: float) -> list[dict[str, float]]:
    """A closed square in the Netherlands, `dx` degrees east, with a height per corner if `dims="xyz"`."""
    corners = [(4.0, 52.0), (5.0, 52.0), (5.0, 53.0), (4.0, 53.0), (4.0, 52.0)]
    return [
        {"x": x + dx, "y": y, **({"z": 10.0 * i} if dims == "xyz" else {})}
        for i, (x, y) in enumerate(corners)
    ]


Geometry = Callable[[list[dict[str, float]]], object]

#: Per kind: its XY and XYZ dtypes, and how to make one of it from a ring.
KINDS: dict[str, tuple[type[GeoArrowType], type[GeoArrowType], Geometry]] = {
    "point": (PointXY, PointXYZ, lambda ring: ring[1]),
    "linestring": (LineStringXY, LineStringXYZ, lambda ring: ring),
    "multipoint": (MultiPointXY, MultiPointXYZ, lambda ring: ring[:3]),
    "polygon": (PolygonXY, PolygonXYZ, lambda ring: [ring]),
    "multilinestring": (
        MultiLineStringXY,
        MultiLineStringXYZ,
        lambda ring: [ring, ring[1:3]],
    ),
    "multipolygon": (MultiPolygonXY, MultiPolygonXYZ, lambda ring: [[ring], [ring]]),
}


def _column(kind: str, dims: str, crs: str, rows: int = 3) -> pl.DataFrame:
    """`rows` geometries of `kind` in column `g`, each further east, with the second one null."""
    xy, xyz, geometry = KINDS[kind]
    dtype = xyz if dims == "xyz" else xy
    values = [None if i == 1 else geometry(_ring(dims, dx=i / 10)) for i in range(rows)]
    storage = pl.Series("g", values, dtype=dtype._geo_storage)
    return pl.DataFrame(storage.ext.to(dtype(crs=crs)))


def _flat(values: object) -> list[object]:
    """Nested lists flattened, leaving out a null geometry."""
    if values is None:
        return []
    if isinstance(values, list):
        return [v for value in values for v in _flat(value)]
    return [values]


Axis = Callable[[str], pl.Expr]
XYZ = (geo.x, geo.y, geo.z)


def _coordinates(
    df: pl.DataFrame, axes: tuple[Axis, ...] = (geo.x, geo.y)
) -> list[tuple[float, ...]]:
    columns = [_flat(df.select(axis("g")).to_series().to_list()) for axis in axes]
    return list(zip(*columns, strict=True))


@pytest.mark.parametrize("kind", list(KINDS))
def test_every_coordinate_of_every_kind_is_reprojected(kind: str) -> None:
    column = _column(kind, "xy", WGS84)

    out = column.select(geo.to_crs("g", WEB_MERCATOR))

    assert _coordinates(out) == [
        pytest.approx(_web_mercator(*xy), abs=TOLERANCE) for xy in _coordinates(column)
    ]


@pytest.mark.parametrize("kind", list(KINDS))
def test_with_z_every_coordinate_of_every_kind_is_reprojected_in_3d(
    kind: str,
) -> None:
    column = _column(kind, "xyz", WGS84_3D)

    out = column.select(geo.to_crs("g", GEOCENTRIC))

    assert _coordinates(out, XYZ) == [
        pytest.approx(_geocentric(*xyz), abs=TOLERANCE)
        for xyz in _coordinates(column, XYZ)
    ]


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_z_that_stays_out_of_proj_leaves_x_and_y_reprojected(kind: str) -> None:
    column = _column(kind, "xyz", WGS84)

    out = column.select(
        geo.to_crs("g", WEB_MERCATOR, allow_projection_with_unscaled_z=True)
    )

    assert _coordinates(out, XYZ) == [
        pytest.approx((*_web_mercator(x, y), z), abs=TOLERANCE)
        for x, y, z in _coordinates(column, XYZ)
    ]


@pytest.mark.parametrize("dims", ["xy", "xyz"])
@pytest.mark.parametrize("kind", list(KINDS))
def test_an_empty_column_of_every_kind_stays_empty(kind: str, dims: str) -> None:
    column = _column(kind, dims, WGS84, rows=0)

    out = column.select(
        geo.to_crs("g", WEB_MERCATOR, allow_projection_with_unscaled_z=True)
    )

    assert out.height == 0


def test_columns_in_different_crss_each_use_their_own_transformation() -> None:
    """Two columns to one target, and one column to two targets, in one query."""
    mercator = _web_mercator(5.0, 52.0)
    df = pl.concat(
        [
            _points_at([(5.0, 52.0)]).rename({"point": "lonlat"}),
            _points_at([mercator], crs=WEB_MERCATOR).rename({"point": "mercator"}),
        ],
        how="horizontal",
    )

    out = df.select(
        geo.to_crs("lonlat", WEB_MERCATOR),
        geo.to_crs("mercator", WEB_MERCATOR),
        geo.to_crs("mercator", WGS84).alias("back"),
    )

    assert _xy(out, "lonlat") == [pytest.approx(mercator, abs=TOLERANCE)]
    assert _xy(out, "mercator") == [pytest.approx(mercator, abs=TOLERANCE)]
    assert _xy(out, "back") == [pytest.approx((5.0, 52.0), abs=1e-9)]


# More target CRSs than the 16 transformations kept per process, and per thread.
UTM_ZONES = range(1, 21)


def _on_central_meridians() -> pl.DataFrame:
    """Per UTM zone, a point on the equator at its central meridian, which that zone puts at (500000, 0)."""
    return pl.concat(
        [
            _points_at([(-183.0 + 6 * zone, 0.0)]).rename({"point": f"zone {zone}"})
            for zone in UTM_ZONES
        ],
        how="horizontal",
    )


def _to_own_zone(zone: int) -> pl.Expr:
    return geo.to_crs(f"zone {zone}", f"EPSG:{32600 + zone}")


def test_more_crs_pairs_than_are_kept_each_get_their_own() -> None:
    points = _on_central_meridians()

    together = points.select([_to_own_zone(zone) for zone in UTM_ZONES])
    # Every zone twice over, so each pair comes back after the caches were emptied.
    one_by_one = [
        points.select(_to_own_zone(zone)) for zone in [*UTM_ZONES, *UTM_ZONES]
    ]

    for out in [together, *one_by_one]:
        for name in out.columns:
            assert _xy(out, name) == [pytest.approx((500000.0, 0.0), abs=TOLERANCE)], (
                name
            )


def test_the_streaming_engine_gives_what_the_in_memory_one_does() -> None:
    """Enough rows for several morsels, so several threads each get their own transformations."""
    i = pl.col("i")
    null = i % 7 == 3
    lf = (
        pl.select(i=pl.int_range(300_000))
        .lazy()
        .select(
            x=pl.when(~null).then((i % 300) * 0.5 - 75.0),
            y=pl.when(~null).then((i % 140) * 0.5 - 35.0),
        )
        .select(geo.point("x", "y", crs=WGS84).alias("point"))
        .select(
            "point",
            geo.to_crs("point", WEB_MERCATOR).alias("web"),
            geo.to_crs("point", WORLD_MERCATOR).alias("world"),
        )
    )

    streamed = lf.collect(engine="streaming")

    assert_frame_equal(streamed, lf.collect())
    sample = streamed.gather_every(997)
    assert _xy(sample, "web") == _in_web_mercator(_xy(sample, "point"))
