//! The bounding box of a geometry, or of a whole column.
//! Implements the Python `bounds` and `total_bounds`.

use geopolars_compute::native::BoxOptions;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::bbox::GeoBox;
use crate::geoarrow::{describe, storage_series};

/// The keyword arguments of the Python `bounds`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundsKwargs {
    margin_x: f64,
    margin_y: f64,
    /// The CRS's longitude turn, asked of PROJ at plan time.
    turn: Option<f64>,
}

/// The box of `dtype`'s dimension and metadata.
fn bounds_dtype(dtype: &DataType) -> PolarsResult<DataType> {
    let geo = describe(dtype)?;
    Ok(GeoBox::new(geo.dim, geo.metadata.clone()).dtype())
}

fn kernel_output_type_bounds(input_fields: &[Field], _kwargs: BoundsKwargs) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let dtype = bounds_dtype(field.dtype())?;
    Ok(Field::new(field.name().clone(), dtype))
}

#[polars_expr(output_type_func_with_kwargs=kernel_output_type_bounds)]
fn bounds(inputs: &[Series], kwargs: BoundsKwargs) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let dtype = bounds_dtype(inputs[0].dtype())?;
    let options = BoxOptions {
        margin_x: kwargs.margin_x,
        margin_y: kwargs.margin_y,
        turn: kwargs.turn,
    };
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| geopolars_core::bounds(chunk.as_ref(), geo.kind, geo.dim, &options))
        .collect::<PolarsResult<Vec<_>>>()?;
    storage_series(inputs[0].name().clone(), chunks, &dtype)
}

/// The keyword arguments of the Python `total_bounds`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TotalBoundsKwargs {
    /// See [`BoundsKwargs`].
    turn: Option<f64>,
    ignore_nulls: bool,
}

fn kernel_output_type_total_bounds(
    input_fields: &[Field],
    _kwargs: TotalBoundsKwargs,
) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let dtype = bounds_dtype(field.dtype())?;
    Ok(Field::new(field.name().clone(), dtype))
}

#[polars_expr(output_type_func_with_kwargs=kernel_output_type_total_bounds)]
fn total_bounds(inputs: &[Series], kwargs: TotalBoundsKwargs) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let (kind, dim) = (geo.kind, geo.dim);
    let dtype = bounds_dtype(inputs[0].dtype())?;
    let name = inputs[0].name().clone();
    // No rows, or only null ones: nothing to bound.
    if inputs[0].null_count() == inputs[0].len() {
        return Ok(Series::full_null(name, 1, &dtype));
    }
    let options = BoxOptions {
        margin_x: 0.0,
        margin_y: 0.0,
        turn: kwargs.turn,
    };
    let storage = inputs[0].ext()?.storage();
    let chunk =
        geopolars_core::total_bounds(storage.chunks(), kind, dim, kwargs.ignore_nulls, &options)?;
    storage_series(name, vec![chunk], &dtype)
}
