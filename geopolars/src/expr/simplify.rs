//! The simplification of a linestring or polygon, or their multi-variants, forwarded to GEOS.

use geopolars_views::{GeoDimension, Kind};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, Geo};

/// Python returns a point or multipoint as it is, so it never gets here.
fn already_simple(dtype: &DataType) -> PolarsError {
    polars_err!(
        SchemaMismatch:
        "simplify expects a linestring, polygon, multilinestring or multipolygon column, got: {}",
        dtype
    )
}

/// The input's kind and metadata, with its z, but without m.
/// Also the kind of the input.
fn simplified_type(dtype: &DataType) -> PolarsResult<(DataType, Kind)> {
    let geo = describe(dtype)?;
    match geo.kind {
        Kind::LineString | Kind::MultiLineString | Kind::Polygon | Kind::MultiPolygon => {}
        Kind::Point | Kind::MultiPoint => return Err(already_simple(dtype)),
    }
    let dim = if geo.dim.has_z() {
        GeoDimension::XYZ
    } else {
        GeoDimension::XY
    };
    let dtype = Geo::new(geo.kind, dim, geo.metadata.clone()).dtype();
    Ok((dtype, geo.kind))
}

/// For [`simplify`].
fn kernel_output_type_simplify(input_fields: &[Field]) -> PolarsResult<Field> {
    let (dtype, _) = simplified_type(input_fields[0].dtype())?;
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

/// See python docs for details.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimplifyKwargs {
    tolerance: f64,
    preserve_topology: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

/// See `simplify`.
#[polars_expr(output_type_func=kernel_output_type_simplify)]
fn simplify(inputs: &[Series], kwargs: SimplifyKwargs) -> PolarsResult<Series> {
    let (dtype, kind) = simplified_type(inputs[0].dtype())?;
    let SimplifyKwargs {
        tolerance,
        preserve_topology,
        ignore_per_row_errors,
        early_finite_check,
    } = kwargs;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| {
            let simplified = geopolars_core::simplify(
                chunk.as_ref(),
                kind,
                tolerance,
                preserve_topology,
                ignore_per_row_errors,
                early_finite_check,
            )?;
            Ok(simplified.boxed())
        })
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the simplified geometries are of the input's kind, in `dtype`'s dimension: its storage.
    Ok(
        unsafe {
            Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &dtype)
        },
    )
}
