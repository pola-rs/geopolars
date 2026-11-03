"""Whether a geometry is valid, by GeoArrow or by OGC, and nulling out the ones that are not.

`is_valid_geoarrow` is a native kernel (A-tier). It checks every coordinate first,
then every row's rings.
`is_valid_ogc` adds the shape checks natively, and the topology checks through GEOS (B-tier).
The CRS bounds are looked up once, while the plan is built.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.datatypes import BoxType
from geopolars.geo._dispatch import on_geometry, on_geometry_or_box
from geopolars.geo.fix_out_of_bounds import _bounds

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType
    from geopolars.geo.fix_out_of_bounds import OutOfBounds


def _kernel(
    function_name: str, *, out_of_bounds: OutOfBounds | None, **checks: bool
) -> Callable[[pl.Expr, GeoArrowType | BoxType], pl.Expr]:
    def build(column: pl.Expr, geometry: GeoArrowType | BoxType) -> pl.Expr:
        if isinstance(geometry, BoxType) and out_of_bounds is not None:
            msg = f"cannot check `out_of_bounds` of a box: {geometry!r}"
            raise TypeError(msg)
        # PROJ is only asked when there are bounds to check.
        bounds = None if out_of_bounds is None else _bounds(geometry, out_of_bounds)
        return register_plugin_function(
            plugin_path=LIB,
            args=[column],
            function_name=function_name,
            is_elementwise=True,
            kwargs={"bounds": bounds, **checks},
        )

    return build


@doc()
def is_valid_geoarrow(
    geometry: IntoExprColumn,
    *,
    out_of_bounds: OutOfBounds | None = None,
    check_null_coordinates: bool = True,
    check_finite_coordinates: bool = False,
) -> pl.Expr:
    """Whether each geometry is valid by the GeoArrow spec: one `bool` per row, a multi-geometry included.
    A null geometry gives a null result.

    GeoArrow asks two things (<https://geoarrow.org/format.html>):

    - only the whole geometry may be null.
      A part, a ring or a coordinate nested in it may not;
    - every ring is closed: its last vertex lies at exactly the same place as its first,
      in x, y and z. An empty ring is closed too.

    This is all the other operations need to read a geometry.
    For the stricter rules of OGC Simple Features, see `is_valid_ogc`.

    It can check two more things about every coordinate,
    in the same pass over the coordinates.

    - `check_finite_coordinates=True`: every axis (x, y, z and m) has to be finite:
      no NaN, no infinity.
      The exception is the empty point, which may stand anywhere, also as a vertex.
    - `out_of_bounds`: no coordinate may lie outside the bounds `fix_out_of_bounds` would fix.
      Its `crs_bounds` says which bounds those are. Its `action` and `too_short` are ignored here.
      With `out_of_bounds=None`, the default, bounds are not checked, and PROJ is not asked for any.

    {empty_point}

    `check_null_coordinates=False` skips looking for null values below the row,
    and trusts the column to hold to GeoArrow's rule on them.
    This can save computation time.
    A null coordinate is then read as whatever values lie under it.
    Do not use this unless you know for sure the inner data is valid!
    The rings still have to be closed.

    A `geoarrow.box` is checked too.
    It is invalid if any of its bounds is null.
    With `check_finite_coordinates=True`, every range has to be finite,
    or the empty range `inf` to `-inf`.
    A finite `xmin > xmax` crosses the antimeridian, and is valid.
    `out_of_bounds` cannot be given for a box.

    ```python
    df.filter(geo.is_valid_geoarrow("parcel"))
    ```
    """
    return on_geometry_or_box(
        geometry,
        _kernel(
            "is_valid_geoarrow",
            out_of_bounds=out_of_bounds,
            check_null_coordinates=check_null_coordinates,
            check_finite_coordinates=check_finite_coordinates,
        ),
    )


def null_out_invalid_geoarrow(
    geometry: IntoExprColumn,
    *,
    out_of_bounds: OutOfBounds | None = None,
    check_null_coordinates: bool = True,
    check_finite_coordinates: bool = False,
) -> pl.Expr:
    """Null out every geometry that `is_valid_geoarrow` rejects, keeping the rest as they are.
    The arguments are those of `is_valid_geoarrow`.

    Every constructor in `geo` already holds to GeoArrow's rules,
    but nothing checks a column that arrives another way,
    such as from a file or relabelled with `ext.to`:

    ```python
    pl.scan_parquet("routes.parquet").select(geo.null_out_invalid_geoarrow("route"))
    ```

    It takes a `geoarrow.box` too.
    geopandas writes the box of a null geometry with every bound null, but the box itself not null.
    This makes that box null:

    ```python
    pl.scan_parquet("places.parquet").with_columns(geo.null_out_invalid_geoarrow("bbox"))
    ```

    On a column that is entirely valid, it copies nothing.
    """
    return on_geometry_or_box(
        geometry,
        _kernel(
            "null_out_invalid_geoarrow",
            out_of_bounds=out_of_bounds,
            check_null_coordinates=check_null_coordinates,
            check_finite_coordinates=check_finite_coordinates,
        ),
    )


def is_valid_ogc(
    geometry: IntoExprColumn,
    *,
    out_of_bounds: OutOfBounds | None = None,
    check_null_coordinates: bool = True,
    check_ring_self_intersect: bool = True,
    check_polygon_self_overlap: bool = True,
    check_multi_polygon_self_overlap: bool = True,
) -> pl.Expr:
    """Whether each geometry is valid by OGC Simple Features
    (which GEOS, and therefore shapely and GeoPandas also uses).
    One `bool` per row, a multi-geometry included. A null geometry gives a null result.

    A geometry valid by OGC is valid by `is_valid_geoarrow` as well,
    with every coordinate finite on every axis.
    Only a point, or a point of a multipoint, may be the empty point (see `is_valid_geoarrow`).
    `out_of_bounds` and `check_null_coordinates` are as for `is_valid_geoarrow`.

    On top of that, per geometry
    (<https://docs.ogc.org/is/06-103r4/06-103r4.pdf>, sections 6.1.11.1 and 6.1.14):

    | geometry | valid when |
    | - | - |
    | `PointXY[ZM]`, `MultiPointXY[ZM]` | its coordinates are |
    | `LineStringXY[ZM]` | it has no vertices, or at least 2 distinct ones |
    | `PolygonXY[ZM]` | all rings empty, or every ring with at least 4 distinct vertices, fitting together (see below) |
    | `MultiLineStringXY[ZM]` | every part valid as a linestring |
    | `MultiPolygonXY[ZM]` | every part valid as a polygon, and the parts fit together (see below) |

    A vertex is distinct when it lies elsewhere in x and y than the one before it.
    So the same vertex twice in a row is fine, as long as there are enough others.
    A linestring may cross itself (see `is_simple`).

    A polygon's rings also have to fit together, in x and y:

    - no ring crosses or touches itself (`check_ring_self_intersect`);
    - every hole lies inside the exterior, no two holes share any area, nor does one lie inside another,
      and rings touch each other only at points, never along a line (`check_polygon_self_overlap`).
      Whether the interior stays in one piece is only checked when every topology check is on.

    A multipolygon's parts fit together, in x and y, when
    no two parts share any area (one may lie inside a hole of another),
    and they touch each other only at points, never along a line (`check_multi_polygon_self_overlap`).

    Each of these three topology checks can be turned off on its own, which can save computation time.
    `check_polygon_self_overlap` has the biggest impact on polygons.
    `check_multi_polygon_self_overlap` compares every pair of parts,
    so its cost grows quickly with their number.

    An empty geometry is valid by OGC, as it is for GEOS.
    Use `is_empty` to find those, and `has_repeats_in_succession` to find the same vertex twice in a row.

    ```python
    df.filter(geo.is_valid_ogc("parcel"))
    ```
    """
    return on_geometry(
        geometry,
        _kernel(
            "is_valid_ogc",
            out_of_bounds=out_of_bounds,
            check_null_coordinates=check_null_coordinates,
            check_ring_self_intersect=check_ring_self_intersect,
            check_polygon_self_overlap=check_polygon_self_overlap,
            check_multi_polygon_self_overlap=check_multi_polygon_self_overlap,
        ),
    )


def null_out_invalid_ogc(
    geometry: IntoExprColumn,
    *,
    out_of_bounds: OutOfBounds | None = None,
    check_null_coordinates: bool = True,
    check_ring_self_intersect: bool = True,
    check_polygon_self_overlap: bool = True,
    check_multi_polygon_self_overlap: bool = True,
) -> pl.Expr:
    """Null out every geometry that `is_valid_ogc` rejects, keeping the rest as they are.
    The arguments are those of `is_valid_ogc`.

    ```python
    df.select(geo.null_out_invalid_ogc("parcel"))
    ```

    On a column that is entirely valid, it copies nothing.
    """
    return on_geometry(
        geometry,
        _kernel(
            "null_out_invalid_ogc",
            out_of_bounds=out_of_bounds,
            check_null_coordinates=check_null_coordinates,
            check_ring_self_intersect=check_ring_self_intersect,
            check_polygon_self_overlap=check_polygon_self_overlap,
            check_multi_polygon_self_overlap=check_multi_polygon_self_overlap,
        ),
    )
