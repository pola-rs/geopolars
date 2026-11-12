//! The convex hull of a linestring or polygon, or their multi-variants, forwarded to GEOS.

use geopolars_core::convex_hulls;
use geopolars_views::{GeoDimension, Kind};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

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

/// See `convex_hull`.
#[polars_expr(output_type_func=kernel_output_type_convex_hull)]
fn convex_hull(inputs: &[Series], kwargs: ConvexHullKwargs) -> PolarsResult<Series> {
    let (dtype, kind) = hull_type(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| {
            let hulls = convex_hulls(
                chunk.as_ref(),
                kind,
                kwargs.ignore_per_row_errors,
                kwargs.ccw_outer,
                kwargs.early_finite_check,
            )?;
            PolarsResult::Ok(hulls.into_list_array().boxed())
        })
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the hulls are polygons of `dtype`'s dimension, its storage.
    Ok(
        unsafe {
            Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &dtype)
        },
    )
}
