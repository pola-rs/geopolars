//! The rings of a polygon (see the `rings` module of `geopolars_compute::native`).

use geopolars_core::{exteriors, interiors, PolygonKind};
use geopolars_views::Kind;
use polars::prelude::*;
use polars_arrow::array::Array;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::{describe, Geo};

/// A `kind` per polygon, or a list of them per multipolygon,
/// with the input's dimension and metadata.
/// Also the kind of the input.
fn rings_type(dtype: &DataType, name: &str, kind: Kind) -> PolarsResult<(DataType, PolygonKind)> {
    let geo = describe(dtype)?;
    let polygon_kind = PolygonKind::new(name, geo.kind)?;
    let part = Geo::new(kind, geo.dim, geo.metadata.clone()).dtype();
    let dtype = match polygon_kind {
        PolygonKind::Polygon => part,
        PolygonKind::MultiPolygon => DataType::List(Box::new(part)),
    };
    Ok((dtype, polygon_kind))
}

/// For [`exterior`].
fn kernel_output_type_exterior(input_fields: &[Field]) -> PolarsResult<Field> {
    let (dtype, _) = rings_type(input_fields[0].dtype(), "exterior", Kind::LineString)?;
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

/// For [`interior`].
fn kernel_output_type_interior(input_fields: &[Field]) -> PolarsResult<Field> {
    let (dtype, _) = rings_type(input_fields[0].dtype(), "interior", Kind::MultiLineString)?;
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

/// Every chunk of the `kind` geometries `inputs[0]` put through `rings`, as a Series of `dtype`.
fn per_chunk(
    inputs: &[Series],
    dtype: &DataType,
    kind: PolygonKind,
    rings: fn(&dyn Array, PolygonKind) -> PolarsResult<Box<dyn Array>>,
) -> PolarsResult<Series> {
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| rings(chunk.as_ref(), kind))
        .collect::<PolarsResult<Vec<_>>>()?;
    // SAFETY: `exteriors` and `interiors` document the storage they give,
    // which `rings_type` describes in the same way.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, dtype) })
}

/// See `exterior`.
#[polars_expr(output_type_func=kernel_output_type_exterior)]
fn exterior(inputs: &[Series]) -> PolarsResult<Series> {
    let (dtype, kind) = rings_type(inputs[0].dtype(), "exterior", Kind::LineString)?;
    per_chunk(inputs, &dtype, kind, exteriors)
}

/// See `interior`.
#[polars_expr(output_type_func=kernel_output_type_interior)]
fn interior(inputs: &[Series]) -> PolarsResult<Series> {
    let (dtype, kind) = rings_type(inputs[0].dtype(), "interior", Kind::MultiLineString)?;
    per_chunk(inputs, &dtype, kind, interiors)
}
