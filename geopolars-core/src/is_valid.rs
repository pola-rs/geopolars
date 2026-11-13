use geopolars_compute::geoarrow_validity::{
    line_strings_are_valid_geoarrow, multi_line_strings_are_valid_geoarrow,
    multi_points_are_valid_geoarrow, multi_polygons_are_valid_geoarrow, points_are_valid_geoarrow,
    polygons_are_valid_geoarrow,
};
use geopolars_compute::inner_nulls::{
    propagate_inner_nulls_of_line_strings, propagate_inner_nulls_of_multi_line_strings,
    propagate_inner_nulls_of_multi_points, propagate_inner_nulls_of_multi_polygons,
    propagate_inner_nulls_of_points, propagate_inner_nulls_of_polygons,
};
use geopolars_compute::native::{Bounds, CoordinateSelection};
use geopolars_compute::ogc_validity::{
    line_strings_are_valid_ogc, multi_line_strings_are_valid_ogc, multi_points_are_valid_ogc,
    multi_polygons_are_valid_ogc, points_are_valid_ogc, polygons_are_valid_ogc,
    MultiPolygonTopologyChecks,
};
use geopolars_views::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray, UncheckedStorage,
};
use polars_arrow::array::{Array, BooleanArray, ListArray};
use polars_core::prelude::{polars_err, PolarsError, PolarsResult};

/// Which validity to check, with its checks.
pub enum Validity {
    /// See the Python `is_valid_geoarrow`.
    GeoArrow(CoordinateSelection),
    /// See the Python `is_valid_ogc`.
    Ogc {
        bounds: Option<Bounds>,
        topology: MultiPolygonTopologyChecks,
    },
}

fn compute_err(e: impl std::fmt::Display) -> PolarsError {
    polars_err!(ComputeError: "is_valid_ogc {e}")
}

/// Checks whether every geometry in one chunk of a `kind` column's storage is valid.
/// Implements the Python `is_valid_geoarrow` and `is_valid_ogc`.
pub fn are_valid(
    chunk: &dyn Array,
    kind: Kind,
    check_null_coordinates: bool,
    validity: &Validity,
) -> PolarsResult<BooleanArray> {
    let propagate = check_null_coordinates;
    let unchecked = || UncheckedStorage::new(chunk);
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");

    let valid = match kind {
        Kind::Point => {
            let array = if propagate {
                propagate_inner_nulls_of_points(unchecked()?)
            } else {
                PointArray::new(Coords::new(chunk)?)
            };
            match validity {
                Validity::GeoArrow(selection) => points_are_valid_geoarrow(&array, selection),
                Validity::Ogc { bounds, .. } => points_are_valid_ogc(&array, *bounds),
            }
        }
        Kind::LineString => {
            let array = if propagate {
                propagate_inner_nulls_of_line_strings(unchecked()?)
            } else {
                LineStringArray::new(rows()?)?
            };
            match validity {
                Validity::GeoArrow(selection) => line_strings_are_valid_geoarrow(&array, selection),
                Validity::Ogc { bounds, .. } => line_strings_are_valid_ogc(&array, *bounds),
            }
        }
        Kind::MultiPoint => {
            let array = if propagate {
                propagate_inner_nulls_of_multi_points(unchecked()?)
            } else {
                MultiPointArray::new(rows()?)?
            };
            match validity {
                Validity::GeoArrow(selection) => multi_points_are_valid_geoarrow(&array, selection),
                Validity::Ogc { bounds, .. } => multi_points_are_valid_ogc(&array, *bounds),
            }
        }
        Kind::MultiLineString => {
            let array = if propagate {
                propagate_inner_nulls_of_multi_line_strings(unchecked()?)
            } else {
                MultiLineStringArray::new(rows()?)?
            };
            match validity {
                Validity::GeoArrow(selection) => {
                    multi_line_strings_are_valid_geoarrow(&array, selection)
                }
                Validity::Ogc { bounds, .. } => multi_line_strings_are_valid_ogc(&array, *bounds),
            }
        }
        Kind::Polygon => {
            let array = if propagate {
                propagate_inner_nulls_of_polygons(unchecked()?)
            } else {
                PolygonArray::new(rows()?)?
            };
            match validity {
                Validity::GeoArrow(selection) => polygons_are_valid_geoarrow(&array, selection),
                Validity::Ogc { bounds, topology } => {
                    polygons_are_valid_ogc(&array, *bounds, &topology.polygons)
                        .map_err(compute_err)?
                }
            }
        }
        Kind::MultiPolygon => {
            let array = if propagate {
                propagate_inner_nulls_of_multi_polygons(unchecked()?)
            } else {
                MultiPolygonArray::new(rows()?)?
            };
            match validity {
                Validity::GeoArrow(selection) => {
                    multi_polygons_are_valid_geoarrow(&array, selection)
                }
                Validity::Ogc { bounds, topology } => {
                    multi_polygons_are_valid_ogc(&array, *bounds, topology).map_err(compute_err)?
                }
            }
        }
    };
    // A null row stays null, and an incomplete one is invalid:
    // both are `false` in `valid`'s values.
    Ok(valid.with_validity(chunk.validity().cloned()))
}
