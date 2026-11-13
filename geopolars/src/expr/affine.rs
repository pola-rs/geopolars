//! Geometric transformations that preserve lines and parallelism,
//! but not necessarily Euclidean distances and angles.
//! The kernels live in `geopolars_compute::native`.

use geopolars_compute::native::{self, RowShift};
use geopolars_core::affine_transform;
use geopolars_views::{Coords, Kind, PointArray};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use crate::geoarrow::{describe, same_geometry_series};

// TODO: (Blocked until z-centroid is implemented): allow rotation around x/y for polygon
// TODO: (Blocked until z-centroid is implemented): allow scaling around z for polygon(?)

/// The kwargs of the affine kernels: see [`native::AffineMatrix`].
#[derive(Deserialize)]
struct AffineMatrix {
    coefficients: [f64; 12],
}

impl From<&AffineMatrix> for native::AffineMatrix {
    fn from(matrix: &AffineMatrix) -> Self {
        Self {
            coefficients: matrix.coefficients,
        }
    }
}

/// The kernel behind affine ops like `translate` (see `geo/affine.py`).
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn affine(inputs: &[Series], kwargs: AffineMatrix) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let matrix = native::AffineMatrix::from(&kwargs);
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| affine_transform(chunk.as_ref(), kind, &matrix, RowShift::None))
        .collect::<PolarsResult<_>>()?;
    let geo = describe(inputs[0].dtype())?.geo();
    Ok(same_geometry_series(inputs[0].name().clone(), geo, chunks))
}

/// For an operation by the points of a second column,
/// one per geometry: it hands back the geometry it got.
fn kernel_output_type_same_geometry_by_points(input_fields: &[Field]) -> PolarsResult<Field> {
    let [geometry, points] = input_fields else {
        polars_bail!(ComputeError: "expected a geometry and its points, got {} inputs", input_fields.len());
    };
    let metadata = describe(geometry.dtype())?.metadata;
    let points = describe(points.dtype())?;
    polars_ensure!(
        points.kind == Kind::Point,
        SchemaMismatch: "expected a `{}` column of points to transform by, got: {}",
        Kind::Point.name(), points.typ
    );
    // Points without a CRS are plain positions, like ones given as numbers.
    if metadata.declares_crs() && points.metadata.declares_crs() {
        let (crs, points_crs) = (metadata.crs()?, points.metadata.crs()?);
        polars_ensure!(
            crs == points_crs,
            SchemaMismatch: "the geometry is in {crs}, but its points are in {points_crs}; \
            use `to_crs` to bring one onto the other first"
        );
    }
    Ok(geometry.clone())
}

/// The kernel behind affine ops about a point per row (see `geo/affine.py`).
/// See [`RowShift::About`].
#[polars_expr(output_type_func=kernel_output_type_same_geometry_by_points)]
fn affine_about(inputs: &[Series], kwargs: AffineMatrix) -> PolarsResult<Series> {
    affine_by_points(inputs, &kwargs, |origins| RowShift::About(origins))
}

/// The kernel behind affine ops that move every geometry by a vector of its own
/// (see `geo/affine.py`). See [`RowShift::Then`].
#[polars_expr(output_type_func=kernel_output_type_same_geometry_by_points)]
fn affine_then_shift(inputs: &[Series], kwargs: AffineMatrix) -> PolarsResult<Series> {
    affine_by_points(inputs, &kwargs, |vectors| RowShift::Then(vectors))
}

/// Transform the geometries of `inputs[0]` by `matrix`,
/// and by `shift` of the points of `inputs[1]`: one per geometry, or one for all of them.
fn affine_by_points(
    inputs: &[Series],
    matrix: &AffineMatrix,
    shift: for<'a> fn(&'a PointArray) -> RowShift<'a>,
) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let storage = inputs[0].ext()?.storage().rechunk();
    let points = inputs[1].ext()?.storage();
    let points = match (storage.len(), points.len()) {
        (n, m) if n == m => points.rechunk(),
        (n, 1) => points.new_from_index(0, n).rechunk(),
        (n, m) => polars_bail!(ShapeMismatch: "cannot transform {n} geometries by {m} points"),
    };
    let points = PointArray::new(Coords::new(points.chunks()[0].as_ref())?);

    let chunk = affine_transform(
        storage.chunks()[0].as_ref(),
        kind,
        &matrix.into(),
        shift(&points),
    )?;
    let geo = describe(inputs[0].dtype())?.geo();
    Ok(same_geometry_series(
        inputs[0].name().clone(),
        geo,
        vec![chunk],
    ))
}
