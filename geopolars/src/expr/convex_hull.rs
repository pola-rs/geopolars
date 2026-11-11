//! The convex hull of a linestring or polygon, or their multi-variants, forwarded to GEOS.

use geopolars_compute::geos::{
    line_string_convex_hulls, multi_line_string_convex_hulls, multi_point_convex_hulls,
    multi_polygon_convex_hulls, polygon_convex_hulls,
};
use geopolars_core::{
    Kind, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::ListArray;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use geopolars_core::GeoDimension;

use super::IntoPolarsResult;
use crate::geoarrow::{describe, Geo};

/// A point encloses nothing, so it is refused.
fn encloses_nothing(dtype: &DataType) -> PolarsError {
    polars_err!(
        SchemaMismatch:
        "convex_hull expects a linestring, polygon, multipoint, multilinestring or multipolygon column, got: {}",
        dtype
    )
}

/// A polygon per geometry, with the input's metadata and z, but without m.
/// Also the kind of the input.
fn hull_type(dtype: &DataType) -> PolarsResult<(DataType, Kind)> {
    let geo = describe(dtype)?;
    match geo.kind {
        Kind::LineString
        | Kind::MultiPoint
        | Kind::MultiLineString
        | Kind::Polygon
        | Kind::MultiPolygon => {}
        Kind::Point => return Err(encloses_nothing(dtype)),
    }
    let dim = if geo.dim.has_z() {
        GeoDimension::XYZ
    } else {
        GeoDimension::XY
    };
    let dtype = Geo::new(Kind::Polygon, dim, geo.metadata.clone()).dtype();
    Ok((dtype, geo.kind))
}

/// For [`convex_hull`].
fn kernel_output_type_convex_hull(input_fields: &[Field]) -> PolarsResult<Field> {
    let (dtype, _) = hull_type(input_fields[0].dtype())?;
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

/// See python docs for details.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConvexHullKwargs {
    ignore_per_row_errors: bool,
    ccw_outer: bool,
    early_finite_check: bool,
}

/// The convex hulls of one chunk of `kind` geometries, as polygons.
fn chunk_hulls(
    rows: &ListArray<i64>,
    kind: Kind,
    dtype: &DataType,
    kwargs: &ConvexHullKwargs,
) -> PolarsResult<ListArray<i64>> {
    let (ignore_errors, ccw_outer, early) = (
        kwargs.ignore_per_row_errors,
        kwargs.ccw_outer,
        kwargs.early_finite_check,
    );
    match kind {
        Kind::LineString => line_string_convex_hulls(
            &LineStringArray::new(rows)?,
            ignore_errors,
            ccw_outer,
            early,
        ),
        Kind::MultiPoint => multi_point_convex_hulls(
            &MultiPointArray::new(rows)?,
            ignore_errors,
            ccw_outer,
            early,
        ),
        Kind::MultiLineString => multi_line_string_convex_hulls(
            &MultiLineStringArray::new(rows)?,
            ignore_errors,
            ccw_outer,
            early,
        ),
        Kind::Polygon => {
            polygon_convex_hulls(&PolygonArray::new(rows)?, ignore_errors, ccw_outer, early)
        }
        Kind::MultiPolygon => multi_polygon_convex_hulls(
            &MultiPolygonArray::new(rows)?,
            ignore_errors,
            ccw_outer,
            early,
        ),
        Kind::Point => return Err(encloses_nothing(dtype)),
    }
    .map(PolygonArray::into_list_array)
    .into_polars()
}

/// See `convex_hull`.
#[polars_expr(output_type_func=kernel_output_type_convex_hull)]
fn convex_hull(inputs: &[Series], kwargs: ConvexHullKwargs) -> PolarsResult<Series> {
    let (dtype, kind) = hull_type(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .list()?
        .downcast_iter()
        .map(|rows| Ok(chunk_hulls(rows, kind, inputs[0].dtype(), &kwargs)?.boxed()))
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the hulls are polygons of `dtype`'s dimension, its storage.
    Ok(
        unsafe {
            Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &dtype)
        },
    )
}
