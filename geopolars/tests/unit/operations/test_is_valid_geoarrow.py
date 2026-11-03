"""Whether a geometry is valid by GeoArrow: whole and closed, and if asked, finite and inside its CRS."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

import geopolars as gpl
from geopolars import geo
from geopolars.datatypes import (
    BoxType,
    BoxXY,
    BoxXYZ,
    GeoArrowType,
    LineStringXY,
    LineStringXYZM,
    MultiLineStringXY,
    MultiPointXY,
    MultiPolygonXY,
    PointXY,
    PointXYZM,
    PolygonXY,
)

if TYPE_CHECKING:
    from geopolars.geo.fix_out_of_bounds import CrsBounds

WGS84 = "EPSG:4326"
UTM31N = "EPSG:32631"

_XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})
_XYZM = pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64, "m": pl.Float64})

Coord = dict[str, float]
Line = list[Coord]

NAN = math.nan
INF = math.inf


def _line(*xy: tuple[float, float]) -> Line:
    return [{"x": x, "y": y} for x, y in xy]


_SQUARE = _line((0, 0), (4, 0), (4, 4), (0, 4), (0, 0))
_HOLE = _line((1, 1), (2, 1), (2, 2), (1, 1))


def _storage(dtype: GeoArrowType) -> pl.DataType:
    """The storage of `dtype`, with every coordinate as `_XY` or `_XYZM`."""
    coords = _XYZM if isinstance(dtype, (PointXYZM, LineStringXYZM)) else _XY
    for _ in range(dtype._nesting):
        coords = pl.List(coords)
    return coords


def _is_valid_geoarrow(
    dtype: GeoArrowType, *rows: object, **kwargs: object
) -> list[bool | None]:
    """`is_valid_geoarrow` over `rows`, relabelled as `dtype` without any checks."""
    storage = pl.Series("g", list(rows), dtype=_storage(dtype))
    df = pl.DataFrame(storage.ext.to(dtype))
    return df.select(geo.is_valid_geoarrow("g", **kwargs))["g"].to_list()


def _is_finite(
    dtype: GeoArrowType, *rows: object, **kwargs: object
) -> list[bool | None]:
    return _is_valid_geoarrow(dtype, *rows, check_finite_coordinates=True, **kwargs)


# Null coordinates.


def test_a_null_axis_is_invalid() -> None:
    assert _is_valid_geoarrow(PointXY(), {"x": 1.0, "y": None}) == [False]


def test_a_null_vertex_is_invalid() -> None:
    assert _is_valid_geoarrow(LineStringXY(), [{"x": 0.0, "y": 0.0}, None]) == [False]


@pytest.mark.parametrize(
    ("dtype", "valid", "null_vertex"),
    [
        (LineStringXY(), _SQUARE, [_SQUARE[0], None]),
        (PolygonXY(), [_SQUARE, _HOLE], [[_SQUARE[0], None, *_SQUARE[2:]]]),
    ],
    ids=["line", "polygon"],
)
def test_a_null_vertex_only_makes_its_own_row_invalid(
    dtype: GeoArrowType, valid: object, null_vertex: object
) -> None:
    assert _is_valid_geoarrow(dtype, valid, null_vertex, valid) == [
        True,
        False,
        True,
    ]


def test_a_null_ring_is_invalid() -> None:
    assert _is_valid_geoarrow(PolygonXY(), [_SQUARE, None]) == [False]


@pytest.mark.parametrize(
    "dtype", [MultiPointXY(), MultiLineStringXY(), MultiPolygonXY()]
)
def test_a_null_part_is_invalid(dtype: GeoArrowType) -> None:
    assert _is_valid_geoarrow(dtype, [None]) == [False]


@pytest.mark.parametrize(
    "dtype", [PointXY(), LineStringXY(), PolygonXY(), MultiPolygonXY()]
)
def test_a_null_geometry_gives_a_null_result(dtype: GeoArrowType) -> None:
    assert _is_valid_geoarrow(dtype, None) == [None]


# Shapes: only the rings have a rule.


@pytest.mark.parametrize(
    "line", [[], _line((0, 0)), _line((0, 0), (1, 1))], ids=["empty", "one", "two"]
)
def test_any_line_is_valid(line: Line) -> None:
    assert _is_valid_geoarrow(LineStringXY(), line) == [True]


def test_a_polygon_with_a_hole_is_valid() -> None:
    assert _is_valid_geoarrow(PolygonXY(), [_SQUARE, _HOLE]) == [True]


def test_an_open_ring_is_invalid() -> None:
    assert _is_valid_geoarrow(PolygonXY(), [_SQUARE[:-1]]) == [False]


def test_an_open_hole_makes_the_polygon_invalid() -> None:
    assert _is_valid_geoarrow(PolygonXY(), [_SQUARE, _HOLE[:-1]]) == [False]


@pytest.mark.parametrize(
    "ring", [_line((0, 0)), _line((0, 0), (1, 1), (0, 0))], ids=["one", "three"]
)
def test_a_short_closed_ring_is_valid(ring: Line) -> None:
    assert _is_valid_geoarrow(PolygonXY(), [ring]) == [True]


@pytest.mark.parametrize("rings", [[], [[]], [[], []], [[], _HOLE]])
def test_an_empty_ring_is_closed(rings: list[Line]) -> None:
    assert _is_valid_geoarrow(PolygonXY(), rings) == [True]


def test_a_multipolygon_with_one_open_ring_is_invalid() -> None:
    good, other = [_SQUARE, _HOLE], [_SQUARE]
    bad = [_SQUARE[:-1]]

    assert _is_valid_geoarrow(MultiPolygonXY(), [good, other], [good, bad]) == [
        True,
        False,
    ]


@pytest.mark.parametrize(
    "dtype", [MultiPointXY(), MultiLineStringXY(), MultiPolygonXY()]
)
def test_an_empty_multi_geometry_is_valid(dtype: GeoArrowType) -> None:
    assert _is_valid_geoarrow(dtype, []) == [True]


# Finite coordinates.


@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_an_axis_may_be_non_finite_by_default(value: float) -> None:
    assert _is_valid_geoarrow(LineStringXY(), _line((0, 0), (value, 1))) == [True]


@pytest.mark.parametrize("axis", ["x", "y", "z", "m"])
@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_a_non_finite_axis_can_be_refused(axis: str, value: float) -> None:
    point = {"x": 1.0, "y": 2.0, "z": 3.0, "m": 4.0} | {axis: value}

    assert _is_finite(PointXYZM(), point) == [False]


def test_a_point_that_is_nan_on_every_axis_is_the_empty_point() -> None:
    assert _is_finite(PointXYZM(), dict.fromkeys("xyzm", NAN)) == [True]


def test_a_point_that_is_nan_only_in_x_and_y_is_not_finite() -> None:
    point = {"x": NAN, "y": NAN, "z": 3.0, "m": 4.0}

    assert _is_finite(PointXYZM(), point) == [False]


def test_a_point_that_is_only_partly_nan_is_not_finite() -> None:
    assert _is_finite(PointXY(), {"x": NAN, "y": 1.0}) == [False]


def test_an_empty_point_may_be_a_vertex() -> None:
    assert _is_finite(LineStringXY(), _line((0, 0), (NAN, NAN))) == [True]


def test_a_multipoint_with_one_non_finite_point_is_invalid() -> None:
    good, bad = {"x": 0.0, "y": 0.0}, {"x": INF, "y": 0.0}

    assert _is_finite(MultiPointXY(), [good, good], [good, bad]) == [True, False]


def test_a_multilinestring_with_one_non_finite_line_is_invalid() -> None:
    good, bad = _line((0, 0), (1, 1)), _line((0, 0), (NAN, 1))

    assert _is_finite(MultiLineStringXY(), [good, good], [good, bad]) == [True, False]


def test_without_the_null_check_a_coordinate_can_still_be_held_finite() -> None:
    line = _line((0, 0), (NAN, 1))

    assert _is_finite(LineStringXY(), line, check_null_coordinates=False) == [False]


def test_the_finite_check_still_needs_every_coordinate_there() -> None:
    line = [{"x": 0.0, "y": 0.0}, {"x": 1.0}]

    assert _is_finite(LineStringXY(), line) == [False]


# CRS bounds.


def _point(x: float, y: float) -> Coord:
    return {"x": x, "y": y}


def _within_crs(
    dtype: GeoArrowType, *rows: object, crs_bounds: CrsBounds = "crs_wrapped_longitude"
) -> list[bool | None]:
    out_of_bounds = geo.OutOfBounds(crs_bounds=crs_bounds)
    return _is_valid_geoarrow(dtype, *rows, out_of_bounds=out_of_bounds)


def test_there_are_no_crs_bounds_by_default() -> None:
    assert _is_valid_geoarrow(PointXY(crs=WGS84), _point(0, 91)) == [True]


def test_by_default_proj_is_not_asked() -> None:
    """An opaque SRID would raise if PROJ were asked for its area of use."""
    assert _is_valid_geoarrow(_srid(), _point(0, 0)) == [True]


def test_a_latitude_past_the_pole_is_invalid() -> None:
    points = [_point(0, 90), _point(0, 91), _point(0, -91)]

    assert _within_crs(PointXY(crs=WGS84), *points) == [True, False, False]


def test_a_longitude_up_to_a_full_turn_east_can_be_valid() -> None:
    points = [_point(-180, 0), _point(270, 0), _point(360, 0)]

    assert _within_crs(PointXY(crs=WGS84), *points) == [True, True, True]


def test_a_longitude_past_a_full_turn_or_half_a_turn_west_is_invalid() -> None:
    points = [_point(361, 0), _point(-181, 0)]

    assert _within_crs(PointXY(crs=WGS84), *points) == [False, False]


def test_a_longitude_past_the_antimeridian_can_be_refused() -> None:
    points = [_point(180, 0), _point(181, 0)]
    out = _within_crs(PointXY(crs=WGS84), *points, crs_bounds="crs")

    assert out == [True, False]


def test_every_vertex_is_held_to_the_bounds() -> None:
    line = _line((0, 0), (0, 95))

    assert _within_crs(LineStringXY(crs=WGS84), line) == [False]


def test_the_empty_point_is_exempt_from_the_bounds() -> None:
    assert _within_crs(PointXY(crs=WGS84), _point(NAN, NAN)) == [True]


def test_without_a_crs_there_are_no_bounds() -> None:
    out = _within_crs(PointXY(), _point(1000, 1000), crs_bounds="area_of_use")

    assert out == [True]


def test_a_projected_crs_is_not_bounded_by_the_crs() -> None:
    assert _within_crs(PointXY(crs=UTM31N), _point(-1e7, 1e8)) == [True]


def test_a_point_outside_the_area_of_use_can_be_refused() -> None:
    """UTM 31N is meant for 0 to 6 degrees east, which lies around easting 500 km."""
    points = [_point(500_000, 5_000_000), _point(-5_000_000, 5_000_000)]
    out = _within_crs(PointXY(crs=UTM31N), *points, crs_bounds="area_of_use")

    assert out == [True, False]


def test_the_area_of_use_narrows_a_geographic_crs() -> None:
    """ETRS89 is meant for Europe only."""
    points = [_point(5, 52), _point(-100, 40)]
    out = _within_crs(PointXY(crs="EPSG:4258"), *points, crs_bounds="area_of_use")

    assert out == [True, False]


def test_the_area_of_use_can_allow_a_wrapped_longitude() -> None:
    """NZGD2000's area of use crosses the antimeridian, so it only bounds latitude."""
    points = [_point(185, -44)]
    nzgd2000 = PointXY(crs="EPSG:4167")

    assert _within_crs(nzgd2000, *points, crs_bounds="area_of_use") == [False]
    assert _within_crs(
        nzgd2000, *points, crs_bounds="area_of_use_wrapped_longitude"
    ) == [True]


def _srid() -> PointXY:
    metadata = '{"crs":"1234","crs_type":"srid"}'
    return PointXY.ext_from_params(PointXY._extension_name, _XY, metadata)


def test_an_opaque_srid_is_not_bounded() -> None:
    assert _within_crs(_srid(), _point(1000, 1000)) == [True]


def test_an_opaque_srid_has_no_area_of_use() -> None:
    with pytest.raises(ComputeError, match="opaque SRID"):
        _within_crs(_srid(), _point(0, 0), crs_bounds="area_of_use")


def test_rejects_an_unknown_crs_bounds() -> None:
    with pytest.raises(ValueError, match="`crs_bounds` must be one of"):
        geo.OutOfBounds(crs_bounds="everywhere")  # type: ignore[arg-type]


def test_the_bounds_do_not_need_the_finite_check() -> None:
    out = _within_crs(PointXY(crs=WGS84), {"x": INF, "y": 0.0}, crs_bounds="crs")

    assert out == [False]


def test_without_the_coordinate_checks_the_rings_still_count() -> None:
    out = _is_valid_geoarrow(PolygonXY(), [_SQUARE[:-1]], check_null_coordinates=False)

    assert out == [False]


# Boxes.

_BOX = {"xmin": 1.0, "ymin": 2.0, "xmax": 3.0, "ymax": 4.0}


def _boxes_are_valid(
    *rows: object, box: BoxType | None = None, **kwargs: object
) -> list[bool | None]:
    box = BoxXY() if box is None else box
    storage = pl.Series("g", list(rows), dtype=box.ext_storage())
    df = pl.DataFrame(storage.ext.to(box))
    return df.select(geo.is_valid_geoarrow("g", **kwargs))["g"].to_list()


def test_a_box_is_valid() -> None:
    assert _boxes_are_valid(_BOX) == [True]


def test_a_null_box_gives_a_null_result() -> None:
    assert _boxes_are_valid(None, _BOX) == [None, True]


def test_a_box_without_any_bound_is_invalid() -> None:
    """The way geopandas writes the box of a null geometry."""
    assert _boxes_are_valid(dict.fromkeys(_BOX)) == [False]


def test_a_box_with_a_null_bound_is_invalid() -> None:
    assert _boxes_are_valid(_BOX | {"ymax": None}, _BOX) == [False, True]


def test_without_the_null_check_a_null_bound_is_not_looked_for() -> None:
    out = _boxes_are_valid(_BOX | {"ymax": None}, check_null_coordinates=False)

    assert out == [True]


@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_a_box_may_be_non_finite_by_default(value: float) -> None:
    assert _boxes_are_valid(_BOX | {"xmin": value}) == [True]


@pytest.mark.parametrize("bound", ["xmin", "ymax"])
@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_a_non_finite_bound_can_be_refused(bound: str, value: float) -> None:
    out = _boxes_are_valid(_BOX | {bound: value}, check_finite_coordinates=True)

    assert out == [False]


@pytest.mark.parametrize("axis", ["x", "y"])
def test_the_empty_range_is_finite(axis: str) -> None:
    box = _BOX | {f"{axis}min": INF, f"{axis}max": -INF}

    assert _boxes_are_valid(box, check_finite_coordinates=True) == [True]


def test_the_empty_range_is_finite_in_z() -> None:
    box = _BOX | {"zmin": INF, "zmax": -INF}

    assert _boxes_are_valid(box, box=BoxXYZ(), check_finite_coordinates=True) == [True]


@pytest.mark.parametrize(("low", "high"), [(-INF, INF), (INF, 3.0), (1.0, -INF)])
def test_only_inf_to_minus_inf_is_the_empty_range(low: float, high: float) -> None:
    box = _BOX | {"xmin": low, "xmax": high}

    assert _boxes_are_valid(box, check_finite_coordinates=True) == [False]


def test_a_box_crossing_the_antimeridian_is_finite() -> None:
    box = _BOX | {"xmin": 170.0, "xmax": -170.0}

    assert _boxes_are_valid(box, check_finite_coordinates=True) == [True]


def test_a_box_has_no_crs_bounds_to_check() -> None:
    with pytest.raises(TypeError, match="`out_of_bounds` of a box"):
        _boxes_are_valid(_BOX, box=BoxXY(crs=WGS84), out_of_bounds=geo.OutOfBounds())


# The expression itself.


def test_it_runs_on_the_streaming_engine() -> None:
    lines = [_SQUARE, [_SQUARE[0], None], None, []] * 100
    lf = (
        pl.LazyFrame({"g": lines}, schema={"g": pl.List(_XY)})
        .select(pl.col("g").ext.to(LineStringXY()))
        .select(geo.is_valid_geoarrow("g"))
    )

    assert_frame_equal(lf.collect(engine="streaming"), lf.collect())


def test_a_slice_is_checked_on_its_own_rows() -> None:
    rings = [[_SQUARE[:-1]], [_SQUARE], [_SQUARE, None], [_SQUARE, _HOLE]]
    storage = pl.Series("g", rings, dtype=pl.List(pl.List(_XY)))
    df = pl.DataFrame(storage.ext.to(PolygonXY())).slice(1, 2)

    assert df.select(geo.is_valid_geoarrow("g"))["g"].to_list() == [True, False]


def test_namespace_matches_the_functional_api() -> None:
    storage = pl.Series("g", [_point(0, 0), _point(0, 95)], dtype=_XY)
    df = pl.DataFrame(storage.ext.to(PointXY(crs=WGS84)))

    assert_frame_equal(
        df.select(gpl.col("g").geo.is_valid_geoarrow(out_of_bounds=geo.OutOfBounds())),
        df.select(geo.is_valid_geoarrow("g", out_of_bounds=geo.OutOfBounds())),
    )


def test_namespace_passes_the_checks_on() -> None:
    rows = [[_SQUARE], [[_point(0, 0), _point(INF, 0), _point(4, 4), _point(0, 0)]]]
    df = pl.DataFrame(
        pl.Series("g", rows, dtype=_storage(PolygonXY())).ext.to(PolygonXY())
    )
    checks = {"check_null_coordinates": False, "check_finite_coordinates": True}

    out = df.select(gpl.col("g").geo.is_valid_geoarrow(**checks))

    assert_frame_equal(out, df.select(geo.is_valid_geoarrow("g", **checks)))
    assert out["g"].to_list() == [True, False]


def test_rejects_a_non_geometry_while_building_the_plan() -> None:
    lf = pl.LazyFrame({"g": [1.0]}).select(geo.is_valid_geoarrow("g"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()


def test_is_valid_ogc_still_rejects_a_box() -> None:
    lf = pl.LazyFrame(schema={"g": BoxXY()}).select(geo.is_valid_ogc("g"))

    with pytest.raises(TypeError, match="expected a `geoarrow.point`"):
        lf.collect_schema()
