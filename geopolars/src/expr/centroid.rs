//! The centroid of a geometry.
//! Implements the Python `centroid` for (multi)linestrings and (multi)polygons,
//! as the Python side takes the mean coordinate of a (multi)point itself.

use geopolars_compute::native::{CentroidAnchor, CentroidMethod};
use geopolars_core::{centroids, SegmentKind};
use geopolars_views::Kind;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, geometry_series, Geo};

#[derive(Deserialize)]
#[serde(remote = "CentroidMethod", rename_all = "snake_case")]
enum CentroidMethodDef {
    Footprint,
    #[serde(rename = "3d")]
    ThreeD,
}

#[derive(Deserialize)]
#[serde(remote = "CentroidAnchor", rename_all = "snake_case")]
enum CentroidAnchorDef {
    LengthWeightedMean,
    VertexMean,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CentroidKwargs {
    #[serde(with = "CentroidMethodDef")]
    method: CentroidMethod,
    #[serde(with = "CentroidAnchorDef")]
    anchor: CentroidAnchor,
}

/// The point a `dtype` column's centroids are, and the kind of geometry they are of.
fn centroid_of(dtype: &DataType) -> PolarsResult<(Geo, SegmentKind)> {
    let geo = describe(dtype)?;
    let kind = SegmentKind::new("centroid", geo.kind)?;
    Ok((Geo::new(Kind::Point, geo.dim, geo.metadata.clone()), kind))
}

fn kernel_output_type_centroid(
    input_fields: &[Field],
    _kwargs: CentroidKwargs,
) -> PolarsResult<Field> {
    let (point, _) = centroid_of(input_fields[0].dtype())?;
    Ok(Field::new(input_fields[0].name().clone(), point.dtype()))
}

#[polars_expr(output_type_func_with_kwargs=kernel_output_type_centroid)]
fn centroid(inputs: &[Series], kwargs: CentroidKwargs) -> PolarsResult<Series> {
    let (point, kind) = centroid_of(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| {
            centroids(
                chunk.as_ref(),
                kind,
                point.dim(),
                kwargs.method,
                kwargs.anchor,
            )
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    geometry_series(inputs[0].name().clone(), point, chunks)
}
