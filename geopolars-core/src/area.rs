use std::fmt::Display;

use geopolars_compute::native::{multi_polygon_areas, polygon_areas, AreaMetric};
use geopolars_views::{downcast, Kind, MultiPolygonArray, PolygonArray};
use polars_arrow::array::{Array, PrimitiveArray};
use polars_core::prelude::{polars_err, PolarsResult};

use crate::refused;

/// Measures the area of every polygon or multipolygon in one chunk of a `kind` column's storage,
/// as `metric` measures it.
/// Implements the Python `area_planar` and `area_geodesic`.
pub fn areas(
    chunk: &dyn Array,
    kind: Kind,
    metric: &impl AreaMetric<Error: Display>,
) -> PolarsResult<PrimitiveArray<f64>> {
    match kind {
        Kind::Polygon => {
            let array = PolygonArray::new_untrimmed(downcast(chunk, "a list of rings")?)?;
            polygon_areas(metric, &array)
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new_untrimmed(downcast(chunk, "a list of polygons")?)?;
            multi_polygon_areas(metric, &array)
        }
        Kind::Point | Kind::LineString | Kind::MultiPoint | Kind::MultiLineString => {
            return Err(refused("area", kind))
        }
    }
    .map_err(|e| polars_err!(ComputeError: "{e}"))
}
