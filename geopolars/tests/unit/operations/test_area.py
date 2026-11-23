"""The area a geometry encloses: on the plane without a CRS, along the
ellipsoid with one."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_series_equal

from geopolars import geo
from geopolars._typing import DistanceMethod
from geopolars.datatypes import (
    GeoArrowType,
    LineStringXY,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PointXY,
    PolygonXY,
)
from tests.unit.conftest import XY, Dimension

_SQUARE = [
    {"x": 0.0, "y": 0.0},
    {"x": 4.0, "y": 0.0},
    {"x": 4.0, "y": 4.0},
    {"x": 0.0, "y": 4.0},
    {"x": 0.0, "y": 0.0},
]

_XY_VERTICES = pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))
_XY_RINGS = pl.List(_XY_VERTICES)


def _rings(rings: list[list[dict[str, float]]], crs: str | None = None) -> pl.DataFrame:
    """A one-row polygon column, built straight from XY rings."""
    return pl.DataFrame({"rings": [rings]}, schema={"rings": _XY_RINGS}).select(
        geo.polygons_from_rings_in_single_column("rings", crs=crs).alias("polygon")
    )


def _relabelled(rings: list[list[dict[str, float]]]) -> pl.DataFrame:
    """As `_rings`, but relabelled rather than built,
    for rings a constructor refuses but that can still arrive around it."""
    return pl.DataFrame({"rings": [rings]}, schema={"rings": _XY_RINGS}).select(
        pl.col("rings").ext.to(PolygonXY()).alias("polygon")
    )


Area = Callable[..., pl.Expr]


def _area_by(distance_method: DistanceMethod) -> Area:
    return partial(geo.area, distance_method=distance_method)


@pytest.fixture(params=["auto", "planar"])
def planar_area(request: pytest.FixtureRequest) -> Area:
    """`"auto"` on a geometry without a CRS is `"planar"`."""
    return _area_by(request.param)


@pytest.fixture(params=["auto", "geodesic"])
def geodesic_area(request: pytest.FixtureRequest) -> Area:
    """`"auto"` on a geometry with a CRS that is not equal-area is `"geodesic"`."""
    return _area_by(request.param)


@pytest.fixture(params=["auto", "explicit"])
def area(request: pytest.FixtureRequest) -> Area:
    """`"auto"`, or the method it picks for the test's `crs`."""
    if request.param == "auto":
        return geo.area
    return _area_by("geodesic" if request.node.callspec.params["crs"] else "planar")


def _areas(area: Area, df: pl.DataFrame, name: str = "polygon") -> list[float | None]:
    return df.select(area(name)).to_series().to_list()


def test_a_polygon_is_its_exterior_ring_less_its_holes(
    planar_area: Area, ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """`a` is a 4x4 square with a triangular hole of 0.5 in it; `b` is a
    triangle with legs of 2."""
    df = dimension.polygons(ring_coords)

    assert _areas(planar_area, df) == [15.5, 2.0]


def test_z_and_m_are_left_out(
    planar_area: Area, ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """The area of the footprint on the xy plane. A sloped roof does not cover
    more ground than a flat one, and `m` is a measure, not an axis."""
    flat = _areas(planar_area, XY.polygons(ring_coords))

    assert _areas(planar_area, dimension.polygons(ring_coords)) == flat


def test_winding_order_does_not_change_the_area(planar_area: Area) -> None:
    """The spec fixes no winding order, so a ring that runs the other way is
    the same ring. Position is what says which one is the hole."""
    assert _areas(planar_area, _rings([_SQUARE])) == _areas(
        planar_area, _rings([_SQUARE[::-1]])
    )


def test_a_hole_is_the_second_ring_whichever_way_it_winds(planar_area: Area) -> None:
    hole = [
        {"x": 1.0, "y": 1.0},
        {"x": 2.0, "y": 1.0},
        {"x": 2.0, "y": 2.0},
        {"x": 1.0, "y": 2.0},
        {"x": 1.0, "y": 1.0},
    ]

    assert _areas(planar_area, _rings([_SQUARE, hole])) == [15.0]
    assert _areas(planar_area, _rings([_SQUARE, hole[::-1]])) == [15.0]


def test_a_malformed_ring_encloses_nothing(planar_area: Area) -> None:
    """Fewer than three distinct vertices cannot bound anything."""
    there_and_back = [
        {"x": 1.0, "y": 1.0},
        {"x": 5.0, "y": 5.0},
        {"x": 1.0, "y": 1.0},
    ]

    assert _areas(planar_area, _relabelled([there_and_back])) == [0.0]


def test_an_empty_ring_takes_nothing_with_it(planar_area: Area) -> None:
    assert _areas(planar_area, _relabelled([_SQUARE, []])) == [16.0]


def test_an_empty_polygon_encloses_nothing(planar_area: Area) -> None:
    """No exterior ring at all, which is an area of zero rather than no area."""
    assert _areas(planar_area, _rings([])) == [0.0]


def test_a_null_polygon_has_no_area(planar_area: Area) -> None:
    df = pl.DataFrame({"polygon": [[_SQUARE], None]}, schema={"polygon": _XY_RINGS})

    df = df.select(pl.col("polygon").ext.to(PolygonXY()))

    assert _areas(planar_area, df) == [16.0, None]


def test_the_area_survives_coordinates_far_from_the_origin(planar_area: Area) -> None:
    """A metre-scale parcel in a projected CRS. Squaring the coordinates before
    subtracting them -- the shoelace sum written out about the origin -- throws
    away the digits the answer is made of."""
    ox, oy = 5_123_456.789, 4_987_654.321
    unit = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    ring = [{"x": ox + x, "y": oy + y} for x, y in unit]

    (got,) = _areas(planar_area, _rings([ring]))

    assert got == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("distance_method", ["auto", "planar"])
def test_the_namespace_matches_the_function(
    distance_method: DistanceMethod, ring_coords: pl.DataFrame
) -> None:
    df = XY.polygons(ring_coords)
    namespace = pl.col("polygon").geo.area(distance_method=distance_method)

    assert_series_equal(
        df.select(namespace).to_series(),
        df.select(geo.area("polygon", distance_method=distance_method)).to_series(),
    )


def test_every_form_of_column_gives_the_same_answer(
    planar_area: Area, ring_coords: pl.DataFrame
) -> None:
    """A name, a `pl.col(...)`, a `Series` and a chained expression all work."""
    df = XY.polygons(ring_coords)
    expected = df.select(planar_area("polygon")).to_series()

    assert_series_equal(df.select(planar_area(pl.col("polygon"))).to_series(), expected)
    assert_series_equal(pl.select(planar_area(df["polygon"])).to_series(), expected)
    translated = geo.translate("polygon", (10.0, -3.0))
    assert_series_equal(df.select(planar_area(translated)).to_series(), expected)


def test_rejects_a_non_geometry_while_resolving_the_schema(planar_area: Area) -> None:
    lf = pl.LazyFrame({"polygon": [1.0]}).select(planar_area("polygon"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


WGS84 = "EPSG:4326"

# A degree square on the equator, from GeographicLib's `PolygonArea` docs, in metres².
_DEGREE_SQUARE = [
    {"x": 0.0, "y": 0.0},
    {"x": 1.0, "y": 0.0},
    {"x": 1.0, "y": 1.0},
    {"x": 0.0, "y": 1.0},
    {"x": 0.0, "y": 0.0},
]
_DEGREE_SQUARE_AREA = 12_308_778_361.469_452


def test_with_a_crs_it_is_the_area_on_the_ellipsoid_in_square_meters(
    geodesic_area: Area,
) -> None:
    """Not the 1.0 square degree it is on the plane."""
    (got,) = _areas(geodesic_area, _rings([_DEGREE_SQUARE], crs=WGS84))

    assert got == pytest.approx(_DEGREE_SQUARE_AREA, rel=1e-9)


def test_with_a_crs_winding_order_does_not_change_the_area(geodesic_area: Area) -> None:
    """A clockwise ring is not taken to enclose the rest of the planet."""
    forward = _areas(geodesic_area, _rings([_DEGREE_SQUARE], crs=WGS84))

    assert _areas(geodesic_area, _rings([_DEGREE_SQUARE[::-1]], crs=WGS84)) == forward


def test_with_a_crs_a_hole_is_taken_out(geodesic_area: Area) -> None:
    """The square less the same square: whichever way the hole winds."""
    for hole in (_DEGREE_SQUARE, _DEGREE_SQUARE[::-1]):
        (got,) = _areas(geodesic_area, _rings([_DEGREE_SQUARE, hole], crs=WGS84))

        assert got == pytest.approx(0.0, abs=1e-3)


def test_with_a_crs_the_ellipsoid_is_the_one_the_crs_declares(
    geodesic_area: Area,
) -> None:
    """On a unit sphere, the triangle from the equator to the pole over a
    quarter turn of longitude is an eighth of the sphere."""
    octant = [
        {"x": 0.0, "y": 0.0},
        {"x": 90.0, "y": 0.0},
        {"x": 0.0, "y": 90.0},
        {"x": 0.0, "y": 0.0},
    ]
    sphere = "+proj=longlat +R=1 +type=crs"

    (got,) = _areas(geodesic_area, _rings([octant], crs=sphere))

    assert got == pytest.approx(4 * math.pi / 8)


def test_a_projected_crs_is_measured_on_its_own_ellipsoid(geodesic_area: Area) -> None:
    """The Dutch national grid is in metres already, but measured along the
    ellipsoid the area is the same as in the lon/lat it projects from
    (Amersfoort, on Bessel 1841), not the planar area of the grid."""
    ring = [
        {"x": 4.9, "y": 52.0},
        {"x": 5.1, "y": 52.0},
        {"x": 5.1, "y": 52.2},
        {"x": 4.9, "y": 52.2},
        {"x": 4.9, "y": 52.0},
    ]
    lonlat = _rings([ring], crs="EPSG:4289")
    rd = lonlat.select(geo.to_crs("polygon", "EPSG:28992"))

    (expected,) = _areas(geodesic_area, lonlat)
    (got,) = _areas(geodesic_area, rd)

    # Projecting bends the geodesic edges a little, which moves the area by
    # a few square metres out of hundreds of millions.
    assert got == pytest.approx(expected, rel=1e-6)


def test_a_projected_crs_in_feet_is_measured_in_square_feet(
    geodesic_area: Area,
) -> None:
    """New York Long Island is in US survey feet, projected from NAD83:
    the same ring is the same geodesic_area, in square feet rather than square metres."""
    ring = [
        {"x": -74.0, "y": 40.7},
        {"x": -73.9, "y": 40.7},
        {"x": -73.9, "y": 40.8},
        {"x": -74.0, "y": 40.8},
        {"x": -74.0, "y": 40.7},
    ]
    lonlat = _rings([ring], crs="EPSG:4269")
    feet = lonlat.select(geo.to_crs("polygon", "EPSG:2263"))
    meters_per_us_foot = 1200 / 3937

    (square_meters,) = _areas(geodesic_area, lonlat)
    (got,) = _areas(geodesic_area, feet)

    assert got == pytest.approx(square_meters / meters_per_us_foot**2, rel=1e-6)


def test_with_a_crs_z_and_m_are_left_out(
    geodesic_area: Area, ring_coords: pl.DataFrame, dimension: Dimension
) -> None:
    def with_crs(df: pl.DataFrame, of: Dimension) -> pl.DataFrame:
        dtype = of.polygon_dtype(crs=WGS84)
        return df.select(pl.col("polygon").ext.storage().ext.to(dtype))

    flat = _areas(geodesic_area, with_crs(XY.polygons(ring_coords), XY))

    got = _areas(geodesic_area, with_crs(dimension.polygons(ring_coords), dimension))

    assert got == flat


def test_with_a_crs_a_null_polygon_has_no_area(geodesic_area: Area) -> None:
    df = pl.DataFrame(
        {"polygon": [[_DEGREE_SQUARE], None]}, schema={"polygon": _XY_RINGS}
    )

    df = df.select(pl.col("polygon").ext.to(PolygonXY(crs=WGS84)))

    assert _areas(geodesic_area, df) == [pytest.approx(_DEGREE_SQUARE_AREA), None]


def test_with_a_crs_an_empty_polygon_encloses_nothing(geodesic_area: Area) -> None:
    assert _areas(geodesic_area, _rings([], crs=WGS84)) == [0.0]


def test_a_crs_not_on_longitude_latitude_is_refused(geodesic_area: Area) -> None:
    """Earth-centred XYZ has no ellipsoid surface to measure an area on."""
    df = _rings([_DEGREE_SQUARE], crs="EPSG:4978")

    with pytest.raises(ComputeError, match="not defined on longitude/latitude"):
        _areas(geodesic_area, df)


def test_a_crs_not_on_longitude_latitude_is_refused_at_plan_time(
    geodesic_area: Area,
) -> None:
    lf = (
        _rings([_DEGREE_SQUARE], crs="EPSG:4978")
        .lazy()
        .select(geodesic_area("polygon"))
    )

    with pytest.raises(ComputeError, match="not defined on longitude/latitude"):
        lf.collect_schema()


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_the_result_is_named_after_the_geometry(area: Area, crs: str | None) -> None:
    df = _rings([_DEGREE_SQUARE], crs=crs)

    assert df.select(area("polygon")).schema == pl.Schema({"polygon": pl.Float64})


def _declaring(
    df: pl.DataFrame, dtype: type[GeoArrowType], crs: str | None
) -> pl.DataFrame:
    """`df`'s only column, declaring `crs`."""
    (name,) = df.columns
    return df.select(pl.col(name).ext.storage().ext.to(dtype(crs=crs)))


def _multipolygons(
    multis: list[list[list[list[dict[str, float]]]] | None], crs: str | None = None
) -> pl.DataFrame:
    """A multipolygon column, built straight from XY polygons."""
    storage = pl.Series("multipolygon", multis, dtype=pl.List(_XY_RINGS))
    return pl.DataFrame(storage.ext.to(MultiPolygonXY(crs=crs)))


def test_a_multipolygon_is_the_sum_of_its_polygons(
    planar_area: Area, multipolygon_coords: pl.DataFrame, dimension: Dimension
) -> None:
    """`p` is the 15.5 square with its hole and the triangle of 2.0;
    `q` is a 3x3 square."""
    df = dimension.multipolygons(multipolygon_coords)

    assert _areas(planar_area, df, "multipolygon") == [17.5, 9.0]


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_a_multipolygon_measures_each_polygon_as_a_polygon(
    area: Area, crs: str | None, multipolygon_coords: pl.DataFrame
) -> None:
    multis = _declaring(XY.multipolygons(multipolygon_coords), MultiPolygonXY, crs)
    separate = _areas(
        area, _declaring(XY.polygons(multipolygon_coords), PolygonXY, crs)
    )

    got = _areas(area, multis, "multipolygon")

    assert got == [pytest.approx(sum(separate[:2])), pytest.approx(sum(separate[2:]))]


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_a_multipolygon_without_polygons_encloses_nothing(
    area: Area, crs: str | None
) -> None:
    """An empty polygon inside one still encloses nothing, like on its own."""
    df = _multipolygons([[], [[]], None], crs=crs)

    assert _areas(area, df, "multipolygon") == [0.0, 0.0, None]


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_a_sliced_multipolygon_keeps_its_own_polygons(
    area: Area, crs: str | None
) -> None:
    df = _multipolygons(
        [[[_SQUARE]], [[_DEGREE_SQUARE], [_SQUARE, _DEGREE_SQUARE]]], crs=crs
    )

    (got,) = _areas(area, df.slice(1), "multipolygon")

    assert got == pytest.approx(_areas(area, df, "multipolygon")[1])


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
def test_a_multipolygon_gives_an_f64_named_after_the_geometry(
    area: Area, crs: str | None
) -> None:
    df = _multipolygons([[[_DEGREE_SQUARE]]], crs=crs)

    assert df.select(area("multipolygon")).schema == pl.Schema(
        {"multipolygon": pl.Float64}
    )


def test_planar_ignores_the_crs() -> None:
    """Square degrees: the plane the coordinates lie in, not the ellipsoid."""
    df = _rings([_DEGREE_SQUARE], crs=WGS84)

    assert _areas(_area_by("planar"), df) == [1.0]


def test_geodesic_refuses_a_geometry_without_a_crs_at_plan_time() -> None:
    lf = (
        _rings([_SQUARE]).lazy().select(geo.area("polygon", distance_method="geodesic"))
    )

    with pytest.raises(TypeError, match="declares none"):
        lf.collect_schema()


@pytest.mark.parametrize(
    "crs",
    ["EPSG:3035", "EPSG:6933", "ESRI:54009", "ESRI:53008"],
    ids=["lambert-azimuthal", "cylindrical", "mollweide", "sinusoidal"],
)
def test_an_equal_area_crs_is_measured_on_its_plane(crs: str) -> None:
    """Its plane keeps areas, so there is no need to go to the ellipsoid."""
    df = _rings([_SQUARE], crs=crs)

    assert _areas(geo.area, df) == _areas(_area_by("planar"), df) == [16.0]


@pytest.mark.parametrize("crs", ["EPSG:3857", "EPSG:32631"], ids=["mercator", "utm"])
def test_any_other_projected_crs_is_measured_on_the_ellipsoid(crs: str) -> None:
    """A 4 m square near the origin of the projection."""
    square = [{"x": p["x"] + 500_000.0, "y": p["y"]} for p in _SQUARE]
    df = _rings([square], crs=crs)

    got = _areas(geo.area, df)

    assert got == _areas(_area_by("geodesic"), df)
    assert got != _areas(_area_by("planar"), df)


def test_planar_lines_up_across_chunks_and_slices() -> None:
    hole = [
        {"x": 1.0, "y": 1.0},
        {"x": 2.0, "y": 1.0},
        {"x": 2.0, "y": 2.0},
        {"x": 1.0, "y": 1.0},
    ]
    polygons = pl.concat(
        [_rings([_SQUARE]), _rings([_SQUARE, hole]), _rings([_DEGREE_SQUARE])],
        rechunk=False,
    )
    multis = pl.concat(
        [
            _multipolygons([[[_SQUARE]], None]),
            _multipolygons([[[_DEGREE_SQUARE], [_SQUARE, hole]]]),
        ],
        rechunk=False,
    )

    planar = _area_by("planar")
    assert _areas(planar, polygons.slice(1)) == [15.5, 1.0]
    assert _areas(planar, multis.slice(1), "multipolygon") == [None, 16.5]


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
@pytest.mark.parametrize("distance_method", ["auto", "planar", "geodesic"])
def test_only_polygons_are_accepted_while_resolving_the_schema(
    distance_method: DistanceMethod,
    crs: str | None,
    coords: pl.DataFrame,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
) -> None:
    """Points and linestrings, and their multi forms, enclose nothing:
    not even a closed linestring, which is a curve rather than a ring."""
    if distance_method == "geodesic" and crs is None:
        pytest.skip("refused for having no CRS instead")
    fn = _area_by(distance_method)
    point = _declaring(coords.select(XY.point()), PointXY, crs)
    line = _declaring(XY.lines(line_coords), LineStringXY, crs)
    multipoint = _declaring(XY.multipoints(line_coords), MultiPointXY, crs)
    multiline = _declaring(XY.multilinestrings(ring_coords), MultiLineStringXY, crs)

    for df, name in (
        (point, "point"),
        (line, "linestring"),
        (multipoint, "multipoint"),
        (multiline, "multilinestring"),
    ):
        lf = df.lazy().select(fn(df.columns[0]))
        with pytest.raises(TypeError, match=f"does not accept a `geoarrow.{name}`"):
            lf.collect_schema()


def _non_polygons(
    crs: str | None,
    coords: pl.DataFrame,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
) -> list[pl.DataFrame]:
    """A point, linestring, multipoint and multilinestring column, each with a null."""
    columns = [
        (coords.select(XY.point()), PointXY),
        (XY.lines(line_coords), LineStringXY),
        (XY.multipoints(line_coords), MultiPointXY),
        (XY.multilinestrings(ring_coords), MultiLineStringXY),
    ]
    out = []
    for df, dtype in columns:
        declared = _declaring(df, dtype, crs)
        out.append(pl.concat([declared, declared.clear(1)]))
    return out


@pytest.mark.parametrize("crs", [None, WGS84], ids=["planar", "geodesic"])
@pytest.mark.parametrize("distance_method", ["auto", "planar", "geodesic"])
def test_with_allow_non_polygons_they_enclose_nothing(
    distance_method: DistanceMethod,
    crs: str | None,
    coords: pl.DataFrame,
    line_coords: pl.DataFrame,
    ring_coords: pl.DataFrame,
) -> None:
    if distance_method == "geodesic" and crs is None:
        pytest.skip("refused for having no CRS instead")
    fn = _area_by(distance_method)

    for df in _non_polygons(crs, coords, line_coords, ring_coords):
        (name,) = df.columns
        got = df.select(fn(name, allow_non_polygons=True)).to_series()

        assert got.dtype == pl.Float64
        assert got.to_list() == [0.0] * (len(df) - 1) + [None]


def test_with_allow_non_polygons_geodesic_still_needs_a_crs(
    line_coords: pl.DataFrame,
) -> None:
    lf = XY.lines(line_coords).lazy()

    with pytest.raises(TypeError, match="declares none"):
        lf.select(
            geo.area("line", distance_method="geodesic", allow_non_polygons=True)
        ).collect_schema()


def test_allow_non_polygons_leaves_polygons_alone(planar_area: Area) -> None:
    df = _rings([_SQUARE])

    assert df.select(planar_area("polygon", allow_non_polygons=True)).equals(
        df.select(planar_area("polygon"))
    )


@pytest.mark.parametrize("distance_method", ["auto", "planar"])
def test_the_namespace_passes_allow_non_polygons_on(
    distance_method: DistanceMethod, line_coords: pl.DataFrame
) -> None:
    df = XY.lines(line_coords)
    kwargs = {"distance_method": distance_method, "allow_non_polygons": True}

    assert df.select(pl.col("line").geo.area(**kwargs)).equals(
        df.select(geo.area("line", **kwargs))
    )


def test_an_unknown_distance_method_is_refused() -> None:
    with pytest.raises(ValueError, match="`distance_method` must be one of"):
        geo.area("polygon", distance_method="Geodesic")  # type: ignore[arg-type]
