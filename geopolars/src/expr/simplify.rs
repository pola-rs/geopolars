//! The simplification of a linestring or polygon, or their multi-variants, forwarded to GEOS.

use geopolars_compute::geos::{
    simplify_line_strings, simplify_multi_line_strings, simplify_multi_polygons, simplify_polygons,
};
use geopolars_views::{
    GeoDimension, Kind, LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::ListArray;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::IntoPolarsResult;
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

/// One chunk of `kind` geometries, simplified.
fn chunk_simplified(
    rows: &ListArray<i64>,
    kind: Kind,
    dtype: &DataType,
    kwargs: &SimplifyKwargs,
) -> PolarsResult<ListArray<i64>> {
    let SimplifyKwargs {
        tolerance,
        preserve_topology,
        ignore_per_row_errors,
        early_finite_check,
    } = *kwargs;
    match kind {
        Kind::LineString => simplify_line_strings(
            &LineStringArray::new(rows)?,
            tolerance,
            preserve_topology,
            ignore_per_row_errors,
            early_finite_check,
        )
        .map(LineStringArray::into_list_array),
        Kind::MultiLineString => simplify_multi_line_strings(
            &MultiLineStringArray::new(rows)?,
            tolerance,
            preserve_topology,
            ignore_per_row_errors,
            early_finite_check,
        )
        .map(MultiLineStringArray::into_list_array),
        Kind::Polygon => simplify_polygons(
            &PolygonArray::new(rows)?,
            tolerance,
            preserve_topology,
            ignore_per_row_errors,
            early_finite_check,
        )
        .map(PolygonArray::into_list_array),
        Kind::MultiPolygon => simplify_multi_polygons(
            &MultiPolygonArray::new(rows)?,
            tolerance,
            preserve_topology,
            ignore_per_row_errors,
            early_finite_check,
        )
        .map(MultiPolygonArray::into_list_array),
        Kind::Point | Kind::MultiPoint => return Err(already_simple(dtype)),
    }
    .into_polars()
}

/// See `simplify`.
#[polars_expr(output_type_func=kernel_output_type_simplify)]
fn simplify(inputs: &[Series], kwargs: SimplifyKwargs) -> PolarsResult<Series> {
    let (dtype, kind) = simplified_type(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .list()?
        .downcast_iter()
        .map(|rows| Ok(chunk_simplified(rows, kind, inputs[0].dtype(), &kwargs)?.boxed()))
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the simplified geometries are of the input's kind, in `dtype`'s dimension: its storage.
    Ok(
        unsafe {
            Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &dtype)
        },
    )
}
