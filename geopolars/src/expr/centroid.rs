//! The centroid of a geometry.
//! Implements the Python `centroid` for (multi)linestrings and (multi)polygons,
//! as the Python side takes the mean coordinate of a (multi)point itself.

use geopolars_core::{centroids, SegmentKind};
use geopolars_views::Kind;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::{describe, geometry_series, Geo};

/// The point a `dtype` column's centroids are, and the kind of geometry they are of.
fn centroid_of(dtype: &DataType) -> PolarsResult<(Geo, SegmentKind)> {
    let geo = describe(dtype)?;
    let kind = SegmentKind::new("centroid", geo.kind)?;
    Ok((Geo::new(Kind::Point, geo.dim, geo.metadata.clone()), kind))
}

fn kernel_output_type_centroid(input_fields: &[Field]) -> PolarsResult<Field> {
    let (point, _) = centroid_of(input_fields[0].dtype())?;
    Ok(Field::new(input_fields[0].name().clone(), point.dtype()))
}

#[polars_expr(output_type_func=kernel_output_type_centroid)]
fn centroid(inputs: &[Series]) -> PolarsResult<Series> {
    let (point, kind) = centroid_of(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| centroids(chunk.as_ref(), kind, point.dim()))
        .collect::<PolarsResult<Vec<_>>>()?;
    geometry_series(inputs[0].name().clone(), point, chunks)
}
