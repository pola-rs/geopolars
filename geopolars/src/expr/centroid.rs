//! The centroid of a geometry.

use geopolars_compute::native::{
    line_string_centroids, multi_line_string_centroids, multi_polygon_centroids, polygon_centroids,
};
use geopolars_compute::with_const_flags;
use geopolars_core::{
    Kind, LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray,
};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::{describe, Geo};

/// The point a `dtype` column's centroids are, and the kind of geometry they are of.
/// The centroid keeps the geometry's dimension.
fn centroid_of(dtype: &DataType) -> PolarsResult<(Geo, Kind)> {
    let geo = describe(dtype)?;
    if let Kind::Point | Kind::MultiPoint = geo.kind {
        polars_bail!(
            SchemaMismatch: "centroid expects a (multi)linestring or (multi)polygon column, got: {}",
            dtype
        )
    }
    Ok((
        Geo::new(Kind::Point, geo.dim, geo.metadata.clone()),
        geo.kind,
    ))
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
    let (has_z, has_m) = (point.dim().has_z(), point.dim().has_m());
    let storage = inputs[0].ext()?.storage();

    let chunks = storage
        .list()?
        .downcast_iter()
        .map(|rows| {
            let points = match kind {
                Kind::LineString => {
                    let array = LineStringArray::new_untrimmed(rows)?;
                    with_const_flags!(Z = has_z, M = has_m; line_string_centroids::<Z, M>(&array))
                }
                Kind::Polygon => {
                    let array = PolygonArray::new_untrimmed(rows)?;
                    with_const_flags!(Z = has_z, M = has_m; polygon_centroids::<Z, M>(&array))
                }
                Kind::MultiLineString => {
                    let array = MultiLineStringArray::new_untrimmed(rows)?;
                    with_const_flags!(
                        Z = has_z, M = has_m;
                        multi_line_string_centroids::<Z, M>(&array)
                    )
                }
                Kind::MultiPolygon => {
                    let array = MultiPolygonArray::new_untrimmed(rows)?;
                    with_const_flags!(Z = has_z, M = has_m; multi_polygon_centroids::<Z, M>(&array))
                }
                Kind::Point | Kind::MultiPoint => unreachable!("refused by `centroid_of`"),
            };
            PolarsResult::Ok(points.into_struct_array().boxed())
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    // SAFETY: every chunk is the storage `point` describes.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &point.dtype())
    })
}
