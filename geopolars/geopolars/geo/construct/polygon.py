"""Building polygons and multipolygons."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.geo.construct._common import (
    _from_columns,
    _gather,
    _polygon_kwargs,
)

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn
    from geopolars.geo.close_rings import CloseRings


@doc()
def polygons_from_rings_in_single_column(
    rings: IntoExprColumn,
    *,
    crs: str | None = None,
    close: CloseRings | None = None,
) -> pl.Expr:
    """Build a `geoarrow.polygon` column out of lists of rings.

    `rings` is a list column holding one list per polygon, of either
    `geoarrow.linestring`s or the bare vertex lists a linestring wraps. The
    first ring of a polygon is its exterior ring; the rest are its holes.

    Every ring has to be a ring (see [`is_ring`][geopolars.geo.is_ring]):
    at least 4 vertices, the last one repeating the first.
    A polygon with a ring that is not one is null in the result.
    The exception is a polygon whose rings are all empty, which is the empty polygon.
    With `close=CloseRings(...)`, an open ring is first closed as `close_rings` does.
    A ring that is still too short after that stays refused.

    {closing_rings}
    {flag_for_absent_axis}

    Rings normally arrive grouped, one linestring at a time:

    ```python
    (
        df.group_by("plot", "ring", maintain_order=True)
        .agg(
            geo.line_string_from_vertices(
                geo.point("lon", "lat").implode()
            ).alias("ring")
        )
        .group_by("plot", maintain_order=True)
        .agg(
            geo.polygons_from_rings_in_single_column(
                pl.col("ring").implode()
            ).alias("plot")
        )
    )
    ```
    """
    return _gather("polygon", rings, _polygon_kwargs(crs, close))


@doc(parts="holes", geometry="exterior")
def polygons_from_rings_in_separate_columns(
    exterior: IntoExprColumn,
    holes: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
    close: CloseRings | None = None,
) -> pl.Expr:
    """Build a `geoarrow.polygon` column out of an exterior ring and its holes.

    `exterior` holds one ring per polygon,
    as a `geoarrow.linestring` or as a bare vertex list.
    `holes` holds one ring per polygon, a list of them, or a multilinestring.
    Without `holes`, no polygon has holes.

    A null exterior gives a null polygon.
    A null `holes` value means the polygon has no holes.
    A null ring among the holes, or a null vertex, makes the polygon null.
    The CRS is the exterior's.
    {same_crs}
    Rings are checked (and closed with `close`)
    as in [`polygons_from_rings_in_single_column`][geopolars.geo.polygons_from_rings_in_single_column].

    ```python
    outer = ways.filter(role="outer").group_by("relation").agg(
        geo.line_string_from_vertices(
            geo.point("lon", "lat").implode()
        ).alias("outer")
    )
    inner = (
        ways.filter(role="inner")
        .group_by("relation", "way")
        .agg(
            geo.line_string_from_vertices(
                geo.point("lon", "lat").implode()
            ).alias("inner")
        )
        .group_by("relation")
        .agg("inner")
    )
    outer.join(inner, on="relation", how="left").select(
        geo.polygons_from_rings_in_separate_columns("outer", "inner")
    )
    ```
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[exterior] if holes is None else [exterior, holes],
        function_name="polygon_rings_in_separate_columns",
        is_elementwise=True,
        kwargs=_polygon_kwargs(crs, close),
    )


def polygons_from_coordinates(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
    close: CloseRings | None = None,
) -> pl.Expr:
    """Build a `geoarrow.polygon` column from one coordinate column per axis.
    Each is a `List[Float64]`, holding the exterior ring of one polygon per row.
    Polygons created by this expression can therefore not have any holes.
    For those, use [`polygons_from_nested_coordinates`][geopolars.geo.polygons_from_nested_coordinates]
    or [`polygons_from_rings_in_separate_columns`][geopolars.geo.polygons_from_rings_in_separate_columns].

    ```python
    df.select(geo.polygons_from_coordinates("lon", "lat"))
    ```

    Rings are checked (and closed with `close`)
    as in [`polygons_from_rings_in_single_column`][geopolars.geo.polygons_from_rings_in_single_column].
    """
    return _from_columns("polygon_coords", x, y, z, m, _polygon_kwargs(crs, close))


def polygons_from_nested_coordinates(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
    close: CloseRings | None = None,
) -> pl.Expr:
    """Build a `geoarrow.polygon` column from one coordinate column per axis.
    Each is a `List[List[Float64]]`: one list per polygon, of one list per ring.
    The first ring of a polygon is its exterior ring, and the rest are its holes.
    See [`polygons_from_coordinates`][geopolars.geo.polygons_from_coordinates]
    for polygons without holes, one ring per row.

    ```python
    df.select(geo.polygons_from_nested_coordinates("lon", "lat"))
    ```

    Rings are checked (and closed with `close`)
    as in [`polygons_from_rings_in_single_column`][geopolars.geo.polygons_from_rings_in_single_column].
    """
    return _from_columns(
        "polygon_nested_coords", x, y, z, m, _polygon_kwargs(crs, close)
    )


@doc()
def multi_polygons_from_polygons(
    polygons: IntoExprColumn,
    *,
    crs: str | None = None,
    close: CloseRings | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multipolygon` column out of lists of polygons.
    Takes either a list of polygons, or bare lists of rings.

    ```python
    (
        df.group_by("country", "island", "ring", maintain_order=True)
        .agg(
            geo.line_string_from_vertices(
                geo.point("lon", "lat").implode()
            ).alias("ring")
        )
        .group_by("country", "island", maintain_order=True)
        .agg(
            geo.polygons_from_rings_in_single_column(
                pl.col("ring").implode()
            ).alias("island")
        )
        .group_by("country", maintain_order=True)
        .agg(
            geo.multi_polygons_from_polygons(
                pl.col("island").implode()
            ).alias("country")
        )
    )
    ```

    Rings are checked (and closed with `close`)
    as in [`polygons_from_rings_in_single_column`][geopolars.geo.polygons_from_rings_in_single_column].
    A multipolygon with a polygon that fails this is null in the result.
    """
    return _gather("multipolygon", polygons, _polygon_kwargs(crs, close))


def multi_polygons_from_coordinates(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
    close: CloseRings | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multipolygon` column from one coordinate column per axis.
    Each is a `List[List[Float64]]`: one list per multipolygon,
    of the exterior ring of each of its polygons.
    Multipolygons created by this expression can therefore not have any holes.
    For those, use [`multi_polygons_from_nested_coordinates`][geopolars.geo.multi_polygons_from_nested_coordinates]
    or [`multi_polygons_from_polygons`][geopolars.geo.multi_polygons_from_polygons].

    ```python
    df.select(geo.multi_polygons_from_coordinates("lon", "lat"))
    ```

    Rings are checked (and closed with `close`)
    as in [`polygons_from_rings_in_single_column`][geopolars.geo.polygons_from_rings_in_single_column].
    A multipolygon with a polygon that fails this is null in the result.
    """
    return _from_columns("multipolygon_coords", x, y, z, m, _polygon_kwargs(crs, close))


def multi_polygons_from_nested_coordinates(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
    close: CloseRings | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multipolygon` column from one coordinate column per axis.
    Each is a `List[List[List[Float64]]]`: polygons, then rings, then coordinates.
    The first ring of every polygon is its exterior ring, and the rest are its holes.
    See [`multi_polygons_from_coordinates`][geopolars.geo.multi_polygons_from_coordinates]
    for multipolygons without holes.

    ```python
    df.select(geo.multi_polygons_from_nested_coordinates("lon", "lat"))
    ```

    Rings are checked (and closed with `close`)
    as in [`polygons_from_rings_in_single_column`][geopolars.geo.polygons_from_rings_in_single_column].
    A multipolygon with a polygon that fails this is null in the result.
    """
    return _from_columns(
        "multipolygon_nested_coords", x, y, z, m, _polygon_kwargs(crs, close)
    )
