"""Labelling a geometry with a coordinate reference system, or moving it to another."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from geopolars._docs import doc
from geopolars._utils import LIB
from geopolars.datatypes import BoxType, Wkb, Wkt
from geopolars.geo._dispatch import on_geometry

if TYPE_CHECKING:
    from geopolars._typing import IntoExprColumn
    from geopolars.datatypes import GeoArrowType
    from geopolars.datatypes.encoded import EncodedGeometry


@doc()
def set_crs(geometry: IntoExprColumn, crs: str, *, force: bool = False) -> pl.Expr:
    """Declare the CRS the coordinates are in, without moving them.

    {crs_forms}
    It is only a label: nothing is checked or reprojected (that is `to_crs`).
    A geometry that already declares a CRS is refused while the plan is built,
    unless `force` is set, in which case its label is overwritten.
    Any other metadata is kept.

    .. code-block:: python

        df.select(geo.set_crs("parcel", "EPSG:28992"))
    """

    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        if geometry._declares_crs() and not force:
            msg = (
                f"{geometry!r} already declares a CRS; use `to_crs` to reproject it, "
                "or `force=True` to overwrite the label"
            )
            raise ValueError(msg)
        return column.ext.storage().ext.to(geometry._with_crs(crs))

    return on_geometry(geometry, build)


def crs_to_projjson(geometry: IntoExprColumn) -> pl.Expr:
    """Write the column's CRS as PROJJSON, which is the same CRS in another form.

    Nothing is reprojected, and any other metadata is kept.
    A column without a CRS, or with one that already is PROJJSON, is left as it is.
    A CRS that cannot be read is refused while the plan is built.
    Besides the geometries, this also takes a box (such as the covering of `with_covering`),
    `Wkb` and `Wkt`.

    It is mainly needed to write GeoParquet,
    as described under "CRS" in `docs/source/user-guide/geo/io.md`.

    .. code-block:: python

        df.with_columns(geo.crs_to_projjson("parcel")).write_parquet("parcels.parquet")
    """

    def build(
        column: pl.Expr, geometry: GeoArrowType | BoxType | EncodedGeometry
    ) -> pl.Expr:
        return column.ext.storage().ext.to(geometry._with_projjson_crs())

    return on_geometry(geometry, build, also=(BoxType, Wkb, Wkt))


def is_geographic(geometry: IntoExprColumn, *, ignore_errors: bool = False) -> pl.Expr:
    """Whether the column's CRS is geographic: its `x` and `y` are longitude and latitude.

    False if the column has no CRS, or with a projected one.
    The CRS is the same for every row, so this is decided while the plan is built
    and gives a single bool (broadcast by `with_columns`), not one per row.
    A CRS that cannot be read is refused while the plan is built,
    unless `ignore_errors` is set, in which case it gives False.

    .. code-block:: python

        df.with_columns(geo.is_geographic("parcel"))
    """

    def build(column: pl.Expr, geometry: GeoArrowType) -> pl.Expr:
        try:
            turn = geometry._longitude_turn()
        except (pl.exceptions.ComputeError, ValueError):
            if not ignore_errors:
                raise
            turn = None
        return pl.lit(turn is not None)

    return on_geometry(geometry, build)


@doc()
def to_crs(
    geometry: IntoExprColumn,
    crs: str,
    *,
    allow_projection_with_unscaled_z: bool = False,
    assume_z_in_target_crs: bool = False,
) -> pl.Expr:
    """Reproject every coordinate from the column's CRS to `crs`.

    {crs_forms}
    The source CRS is read from the column's metadata.
    The result declares `crs` instead, and keeps the rest of the metadata.
    `m` is carried through untouched.

    A CRS specifies `z` when it is compound, or when its coordinate system has three axes
    (e.g. `"EPSG:4979"`, but not `"EPSG:4326"`).
    What happens to `z` depends on which CRSs specify it.
    If a flag is needed but not specified, it fails at plan time.

    =============  =============  =======================================  ===========
    from           to             needs                                    `z`
    =============  =============  =======================================  ===========
    specifies `z`  specifies `z`  nothing                                  reprojected
    specifies `z`  no `z`         `allow_projection_with_unscaled_z=True`  kept as is
    no `z`         no `z`         `allow_projection_with_unscaled_z=True`  kept as is
    no `z`         specifies `z`  `assume_z_in_target_crs=True`            kept as is
    =============  =============  =======================================  ===========

    A geometry without `z` needs neither flag.
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[geometry],
        function_name="to_crs",
        is_elementwise=True,
        kwargs={
            "to": crs,
            "allow_projection_with_unscaled_z": allow_projection_with_unscaled_z,
            "assume_z_in_target_crs": assume_z_in_target_crs,
        },
    )
