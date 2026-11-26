use geopolars_compute::native::{multi_polygon_areas, polygon_areas, AreaMetric, Bounds};
use geopolars_compute::{EarlyCoordCheck, GeoComputeError, OnOutOfBounds, OnRowError};
use geopolars_views::{downcast, MultiPolygonArray, PolygonArray};
use polars_arrow::array::{Array, PrimitiveArray};
use polars_core::prelude::{polars_err, PolarsResult};

use crate::PolygonKind;

/// Measures the area of every polygon or multipolygon in `chunk`, the storage of a `kind` column.
/// How it is measured (geodesic, planar, etc) is determined by `metric`.
/// Implements the Python `area`.
pub fn areas(
    chunk: &dyn Array,
    kind: PolygonKind,
    metric: &impl AreaMetric<Error: Into<GeoComputeError>>,
    on_row_error: OnRowError,
    early_coord_check: EarlyCoordCheck,
    bounds: Option<Bounds>,
    on_out_of_bounds: OnOutOfBounds,
) -> PolarsResult<PrimitiveArray<f64>> {
    match kind {
        PolygonKind::Polygon => {
            let array = PolygonArray::new_untrimmed(downcast(chunk, "a list of rings")?)?;
            polygon_areas(
                metric,
                &array,
                on_row_error,
                early_coord_check,
                bounds,
                on_out_of_bounds,
            )
        }
        PolygonKind::MultiPolygon => {
            let array = MultiPolygonArray::new_untrimmed(downcast(chunk, "a list of polygons")?)?;
            multi_polygon_areas(
                metric,
                &array,
                on_row_error,
                early_coord_check,
                bounds,
                on_out_of_bounds,
            )
        }
    }
    .map_err(|e| polars_err!(ComputeError: "{e}"))
}
