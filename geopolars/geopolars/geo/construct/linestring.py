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

    {no_validity_required}

    `vertices` is a list column holding one list per linestring,
    of either `geoarrow.point` values or the bare coordinate structs a point wraps.
    A `geoarrow.multipoint` column works too: its points become the vertices, in order.

    Vertices normally arrive grouped:

    .. code-block:: python

        df.group_by("route").agg(
            geo.line_string_from_vertices(
                geo.point("lon", "lat").implode()
            ).alias("route")
        )

    {null_inside_row}

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

    {no_validity_required}

    Each is a `List[Float64]` holding one list of coordinates per linestring

    .. code-block:: python

        df.select(geo.line_string_from_columns("lon", "lat", z="elevation"))

    {null_inside_row}

    {constructed_geoarrow_valid}
    """
    return _from_columns("linestring_coords", x, y, z, m, _metadata(crs))


@doc()
def line_string(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.linestring` column, from vertices or from coordinates.

    {no_validity_required}

    Dispatches to either:

    - :func:`~geopolars.geo.line_string_from_vertices`
    - :func:`~geopolars.geo.line_string_from_columns`
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

    {no_validity_required}

    Takes either a list of linestrings, or bare vertex lists.

    .. code-block:: python

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

    {null_inside_row}

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

    {no_validity_required}

    Each is a `List[List[Float64]]`.

    .. code-block:: python

        df.select(geo.multi_line_string_from_columns("lon", "lat"))

    {null_inside_row}

    {constructed_geoarrow_valid}
    """
    return _from_columns("multilinestring_coords", x, y, z, m, _metadata(crs))


@doc()
def multi_line_string(
    x: IntoExprColumn,
    y: IntoExprColumn | None = None,
    z: IntoExprColumn | None = None,
    m: IntoExprColumn | None = None,
    *,
    crs: str | None = None,
) -> pl.Expr:
    """Build a `geoarrow.multilinestring` column, from linestrings or coordinates.

    {no_validity_required}

    Dispatches to either:

    - :func:`~geopolars.geo.multi_line_string_from_line_strings`
    - :func:`~geopolars.geo.multi_line_string_from_columns`
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
