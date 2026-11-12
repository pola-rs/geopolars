use geopolars_compute::native::{
    line_string_centroids, multi_line_string_centroids, multi_polygon_centroids, polygon_centroids,
};
use geopolars_compute::with_const_flags;
use geopolars_views::{
    downcast, GeoDimension, Kind, LineStringArray, MultiLineStringArray, MultiPolygonArray,
    PointArray, PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::refused;

/// Finds the centroid of every (multi)linestring or (multi)polygon
/// in one chunk of a `kind` column's storage of dimension `dim`.
/// Implements the Python `centroid`.
pub fn centroids(chunk: &dyn Array, kind: Kind, dim: GeoDimension) -> PolarsResult<PointArray> {
    let (has_z, has_m) = (dim.has_z(), dim.has_m());
    Ok(match kind {
        Kind::LineString => {
            let array = LineStringArray::new_untrimmed(downcast(chunk, "a list of coordinates")?)?;
            with_const_flags!(Z = has_z, M = has_m; line_string_centroids::<Z, M>(&array))
        }
        Kind::Polygon => {
            let array = PolygonArray::new_untrimmed(downcast(chunk, "a list of rings")?)?;
            with_const_flags!(Z = has_z, M = has_m; polygon_centroids::<Z, M>(&array))
        }
        Kind::MultiLineString => {
            let array =
                MultiLineStringArray::new_untrimmed(downcast(chunk, "a list of linestrings")?)?;
            with_const_flags!(
                Z = has_z, M = has_m;
                multi_line_string_centroids::<Z, M>(&array)
            )
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new_untrimmed(downcast(chunk, "a list of polygons")?)?;
            with_const_flags!(Z = has_z, M = has_m; multi_polygon_centroids::<Z, M>(&array))
        }
        Kind::Point | Kind::MultiPoint => return Err(refused("centroid", kind)),
    })
}
