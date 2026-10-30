"""Building linestrings and multilinestrings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from geopolars._docs import doc
from geopolars.geo.construct._common import _from_columns, _gather, _metadata

if TYPE_CHECKING:
    import polars as pl

    from geopolars._typing import IntoExprColumn


@doc()
def line_string_from_vertices(
    vertices: IntoExprColumn, *, crs: str | None = None
) -> pl.Expr:
    """Build a `geoarrow.linestring` column out of lists of vertices.

    `vertices` is a list column holding one list per linestring,
    of either `geoarrow.point`s or the bare coordinate structs a point wraps.
    A `geoarrow.multipoint` column works too: its points become the vertices, in order.

    Vertices normally arrive grouped:

    ```python
    df.group_by("route").agg(
        geo.line_string_from_vertices(
            geo.point("lon", "lat").implode()
        ).alias("route")
    )
    ```

    {constructed_geoarrow_valid}
    """
    return _gather("linestring", vertices, _metadata(crs))


@doc()
def line_string_from_columns(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.linestring` column from one coordinate column per axis.
    Each is a `List[Float64]` holding one list of coordinates per linestring

    ```python
    df.select(geo.line_string_from_columns("lon", "lat", z="elevation"))
    ```

    {constructed_geoarrow_valid}
    """
    return _from_columns("linestring_coords", x, y, z, m, _metadata(crs))


def line_string(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.linestring` column, from vertices or from coordinates.
    Dispatches to either:
    - [`line_string_from_vertices`][geopolars.geo.line_string_from_vertices]
    - [`line_string_from_columns`][geopolars.geo.line_string_from_columns]
    """
    if y is None:
        if z is not None or m is not None:
            msg = (
                "line_string() got a z or m coordinate without a y coordinate; "
                "pass x and y as columns, or call line_string_from_vertices()"
            )
            raise TypeError(msg)
        return line_string_from_vertices(x, crs=crs)

    return line_string_from_columns(x, y, z, m, crs=crs)


@doc()
def multi_line_string_from_line_strings(
    line_strings: IntoExprColumn, *, crs: str | None = None
) -> pl.Expr:
    """Build a `geoarrow.multilinestring` column out of lists of linestrings.
    Takes either a list of linestrings, or bare vertex lists.

    ```python
    (
        df.group_by("river", "branch", maintain_order=True)
        .agg(
            geo.line_string_from_vertices(
                geo.point("lon", "lat").implode()
            ).alias("branch")
        )
        .group_by("river", maintain_order=True)
        .agg(
            geo.multi_line_string_from_line_strings(
                pl.col("branch").implode()
            ).alias("river")
        )
    )
    ```

    {constructed_geoarrow_valid}
    """
    return _gather("multilinestring", line_strings, _metadata(crs))


@doc()
def multi_line_string_from_columns(
    x: IntoExprColumn,
    y: IntoExprColumn,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multilinestring` column from one column per axis.
    Each is a `List[List[Float64]]`.

    ```python
    df.select(geo.multi_line_string_from_columns("lon", "lat"))
    ```

    {constructed_geoarrow_valid}
    """
    return _from_columns("multilinestring_coords", x, y, z, m, _metadata(crs))


def multi_line_string(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multilinestring` column, from linestrings or coordinates.
    Dispatches to either:
    - [`multi_line_string_from_line_strings`][geopolars.geo.multi_line_string_from_line_strings]
    - [`multi_line_string_from_columns`][geopolars.geo.multi_line_string_from_columns]
    """
    if y is None:
        if z is not None or m is not None:
            msg = (
                "multi_line_string() got a z or m coordinate without a y "
                "coordinate; pass x and y as columns, or call "
                "multi_line_string_from_line_strings()"
            )
            raise TypeError(msg)
        return multi_line_string_from_line_strings(x, crs=crs)

    return multi_line_string_from_columns(x, y, z, m, crs=crs)
