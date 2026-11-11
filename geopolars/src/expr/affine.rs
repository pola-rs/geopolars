//! Geometric transformations that preserve lines and parallelism,
//! but not necessarily Euclidean distances and angles.
//! The kernels live in `geopolars_compute::native`.

use geopolars_compute::native::{
    self, affine_line_strings, affine_multi_line_strings, affine_multi_points,
    affine_multi_polygons, affine_points, affine_polygons, RowShift,
};
use geopolars_core::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::Array;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use crate::geoarrow::describe;

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

/// One chunk of `kind` geometries, transformed by `matrix` and `shift`.
fn chunk_transformed(
    chunk: &dyn Array,
    kind: Kind,
    matrix: &native::AffineMatrix,
    shift: RowShift,
) -> PolarsResult<Box<dyn Array>> {
    Ok(match kind {
        Kind::Point => {
            let array = PointArray::new(Coords::new(chunk)?);
            affine_points(matrix, &array, shift)
                .into_struct_array()
                .boxed()
        }
        Kind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            affine_line_strings(matrix, &array, shift)
                .into_list_array()
                .boxed()
        }
        Kind::MultiPoint => {
            let array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            affine_multi_points(matrix, &array, shift)
                .into_list_array()
                .boxed()
        }
        Kind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            affine_polygons(matrix, &array, shift)
                .into_list_array()
                .boxed()
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            affine_multi_line_strings(matrix, &array, shift)
                .into_list_array()
                .boxed()
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            affine_multi_polygons(matrix, &array, shift)
                .into_list_array()
                .boxed()
        }
    })
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
        .map(|chunk| chunk_transformed(chunk.as_ref(), kind, &matrix, RowShift::None))
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the geometries keep their kind and dimension, so this is the input's storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, inputs[0].dtype())
    })
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

    let chunk = chunk_transformed(
        storage.chunks()[0].as_ref(),
        kind,
        &matrix.into(),
        shift(&points),
    )?;
    // SAFETY: the geometries keep their kind and dimension, so this is the input's storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(
            inputs[0].name().clone(),
            vec![chunk],
            inputs[0].dtype(),
        )
    })
}
