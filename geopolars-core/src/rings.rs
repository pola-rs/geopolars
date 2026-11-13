use geopolars_compute::native::{
    multi_polygon_exteriors, multi_polygon_interiors, polygon_exteriors, polygon_interiors,
};
use geopolars_views::{downcast, MultiPolygonArray, PolygonArray};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::PolygonKind;

/// Finds the exterior of every polygon or multipolygon in one chunk of a `kind` column's storage.
/// Implements the Python `exterior`.
/// The result is the storage of a linestring per polygon, or a list of them per multipolygon.
pub fn exteriors(chunk: &dyn Array, kind: PolygonKind) -> PolarsResult<Box<dyn Array>> {
    Ok(match kind {
        PolygonKind::Polygon => {
            let array = PolygonArray::new_untrimmed(downcast(chunk, "a list of rings")?)?;
            polygon_exteriors(&array).into_list_array().boxed()
        }
        PolygonKind::MultiPolygon => {
            let array = MultiPolygonArray::new_untrimmed(downcast(chunk, "a list of polygons")?)?;
            multi_polygon_exteriors(&array).boxed()
        }
    })
}

/// Finds the holes of every polygon or multipolygon in one chunk of a `kind` column's storage.
/// Implements the Python `interior`.
/// The result is the storage of a multilinestring per polygon,
/// or a list of them per multipolygon.
pub fn interiors(chunk: &dyn Array, kind: PolygonKind) -> PolarsResult<Box<dyn Array>> {
    Ok(match kind {
        PolygonKind::Polygon => {
            let array = PolygonArray::new_untrimmed(downcast(chunk, "a list of rings")?)?;
            polygon_interiors(&array).into_list_array().boxed()
        }
        PolygonKind::MultiPolygon => {
            let array = MultiPolygonArray::new_untrimmed(downcast(chunk, "a list of polygons")?)?;
            multi_polygon_interiors(&array).boxed()
        }
    })
}
