//! The bounding box of a geometry.

use geopolars_compute::native::{box_array, BoxOptions, Extent};
use geopolars_compute::with_const_flags;
use geopolars_core::column_extent;
use geopolars_views::{GeoDimension, Kind};
use polars::prelude::*;
use polars_arrow::array::Array;
use polars_arrow::bitmap::Bitmap;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::bbox::GeoBox;
use crate::geoarrow::describe;

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

/// The box of `dtype`'s dimension and metadata, in a list if each part gets one.
fn bounds_dtype(dtype: &DataType, per_part: bool) -> PolarsResult<DataType> {
    let geo = describe(dtype)?;
    let bbox = GeoBox::new(geo.dim, geo.metadata.clone()).dtype();
    Ok(if per_part && geo.kind.is_multi() {
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
    // SAFETY: every chunk is the storage of `dtype`.
    Ok(
        unsafe {
            Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &dtype)
        },
    )
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

/// The box around every chunk of `storage`, as a column of one row.
fn total_box<const Z: bool, const M: bool, const IGNORE_NULLS: bool>(
    storage: &Series,
    kind: Kind,
    dim: GeoDimension,
    options: &BoxOptions,
) -> PolarsResult<Box<dyn Array>> {
    let mut total = Some(Extent::<Z, M>::EMPTY);
    for chunk in storage.chunks() {
        let extent = column_extent::<Z, M, IGNORE_NULLS>(chunk.as_ref(), kind)?;
        total = total.zip(extent).map(|(mut total, extent)| {
            total.merge(&extent);
            total
        });
    }
    let validity = total.is_none().then(|| Bitmap::new_zeroed(1));
    let total = options.finish(&total.unwrap_or(Extent::EMPTY));
    Ok(box_array(dim, &[total], validity).boxed())
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
    let chunk = with_const_flags!(
        Z = dim.has_z(), M = dim.has_m(), IGNORE_NULLS = kwargs.ignore_nulls;
        total_box::<Z, M, IGNORE_NULLS>(storage, kind, dim, &options)
    )?;
    // SAFETY: the chunk is the storage of `dtype`.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(name, vec![chunk], &dtype) })
}
