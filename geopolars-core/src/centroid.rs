use geopolars_compute::native::{
    line_string_centroids, multi_line_string_centroids, multi_polygon_centroids, polygon_centroids,
    CentroidAnchor, CentroidWeighing,
};
use geopolars_compute::OnRowError;
use geopolars_views::{
    downcast, GeometryArray, LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray,
};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::{IntoPolarsResult, SegmentKind};

/// Finds the centroid of every (multi)linestring or (multi)polygon in one storage chunk.
/// Implements the Python `centroid`.
/// `kind` describes the column that `chunk` comes from.
pub fn centroids(
    chunk: &dyn Array,
    kind: SegmentKind,
    weighing_method: CentroidWeighing,
    anchor: CentroidAnchor,
    on_row_error: OnRowError,
) -> PolarsResult<GeometryArray> {
    let centroids = match kind {
        SegmentKind::LineString => {
            let array = LineStringArray::new_untrimmed(downcast(chunk, "a list of coordinates")?)?;
            line_string_centroids(&array, weighing_method, on_row_error)
        }
        SegmentKind::Polygon => {
            let array = PolygonArray::new_untrimmed(downcast(chunk, "a list of rings")?)?;
            polygon_centroids(&array, weighing_method, anchor, on_row_error)
        }
        SegmentKind::MultiLineString => {
            let array =
                MultiLineStringArray::new_untrimmed(downcast(chunk, "a list of linestrings")?)?;
            multi_line_string_centroids(&array, weighing_method, on_row_error)
        }
        SegmentKind::MultiPolygon => {
            let array = MultiPolygonArray::new_untrimmed(downcast(chunk, "a list of polygons")?)?;
            multi_polygon_centroids(&array, weighing_method, anchor, on_row_error)
        }
    };
    centroids.map(GeometryArray::Point).into_polars()
}
