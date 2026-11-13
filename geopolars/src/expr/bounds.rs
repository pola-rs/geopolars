//! The bounding box of a geometry.

use geopolars_compute::native::BoxOptions;
use geopolars_views::Kind;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::bbox::GeoBox;
use crate::geoarrow::{describe, storage_series};

/// See python `bounds`.
/// `turn` is the CRS's longitude turn, asked of PROJ at plan time.
/// If `per_part=true`, a multi-geometry gets a list of boxes, one per part.
/// If `per_part=false`, it gets one box around the whole row.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundsKwargs {
    margin_x: f64,
    margin_y: f64,
    turn: Option<f64>,
    per_part: bool,
}

fn boxes_in_list(kind: Kind, per_part: bool) -> bool {
    per_part && kind.is_multi()
}

/// The box of `dtype`'s dimension and metadata, in a list if each part gets one.
fn bounds_dtype(dtype: &DataType, per_part: bool) -> PolarsResult<DataType> {
    let geo = describe(dtype)?;
    let bbox = GeoBox::new(geo.dim, geo.metadata.clone()).dtype();
    Ok(if boxes_in_list(geo.kind, per_part) {
        DataType::List(Box::new(bbox))
    } else {
        bbox
    })
}

/// For [`bounds`].
fn kernel_output_type_bounds(input_fields: &[Field], kwargs: BoundsKwargs) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let dtype = bounds_dtype(field.dtype(), kwargs.per_part)?;
    Ok(Field::new(field.name().clone(), dtype))
}

/// See python `bounds`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_bounds)]
fn bounds(inputs: &[Series], kwargs: BoundsKwargs) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let dtype = bounds_dtype(inputs[0].dtype(), kwargs.per_part)?;
    let options = BoxOptions {
        margin_x: kwargs.margin_x,
        margin_y: kwargs.margin_y,
        turn: kwargs.turn,
    };
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| {
            geopolars_core::bounds(chunk.as_ref(), geo.kind, geo.dim, kwargs.per_part, &options)
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    storage_series(inputs[0].name().clone(), chunks, &dtype)
}

/// See python `total_bounds`.
/// `turn` is the CRS's longitude turn, asked of PROJ at plan time.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TotalBoundsKwargs {
    turn: Option<f64>,
    ignore_nulls: bool,
}

/// For [`total_bounds`].
fn kernel_output_type_total_bounds(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let dtype = bounds_dtype(field.dtype(), false)?;
    Ok(Field::new(field.name().clone(), dtype))
}

/// See python `total_bounds`.
#[polars_expr(output_type_func=kernel_output_type_total_bounds)]
fn total_bounds(inputs: &[Series], kwargs: TotalBoundsKwargs) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let (kind, dim) = (geo.kind, geo.dim);
    let dtype = bounds_dtype(inputs[0].dtype(), false)?;
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
