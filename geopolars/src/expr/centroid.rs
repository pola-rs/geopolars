//! The centroid of a geometry.

use geopolars_core::{centroids, SegmentKind};
use geopolars_views::Kind;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::{describe, Geo};

/// The point a `dtype` column's centroids are, and the kind of geometry they are of.
/// The centroid keeps the geometry's dimension.
fn centroid_of(dtype: &DataType) -> PolarsResult<(Geo, SegmentKind)> {
    let geo = describe(dtype)?;
    let kind = SegmentKind::new("centroid", geo.kind)?;
    Ok((Geo::new(Kind::Point, geo.dim, geo.metadata.clone()), kind))
}

/// For [`centroid`].
fn kernel_output_type_centroid(input_fields: &[Field]) -> PolarsResult<Field> {
    let (point, _) = centroid_of(input_fields[0].dtype())?;
    Ok(Field::new(input_fields[0].name().clone(), point.dtype()))
}

/// See python `centroid`.
#[polars_expr(output_type_func=kernel_output_type_centroid)]
fn centroid(inputs: &[Series]) -> PolarsResult<Series> {
    let (point, kind) = centroid_of(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| {
            let points = centroids(chunk.as_ref(), kind, point.dim())?;
            PolarsResult::Ok(points.into_struct_array().boxed())
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    // SAFETY: every chunk is the storage `point` describes.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &point.dtype())
    })
}
