//! Whether a geometry is valid, by GeoArrow (through `geopolars_compute::geoarrow_validity`)
//! or by OGC Simple Features (through `geopolars_compute::ogc_validity`).
//! See `is_valid_geoarrow` and `is_valid_ogc` in Python for the rules.

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
use geopolars_compute::native::{Bounds, CoordinateSelection, NonFinite};
use geopolars_compute::ogc_validity::{
    line_strings_are_valid_ogc, multi_line_strings_are_valid_ogc, multi_points_are_valid_ogc,
    multi_polygons_are_valid_ogc, points_are_valid_ogc, polygons_are_valid_ogc,
    MultiPolygonTopologyChecks, PolygonTopologyChecks,
};
use geopolars_views::{
    downcast, Coords, Kind, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray, UncheckedStorage,
};
use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use crate::geoarrow::describe;

/// See python
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IsValidGeoArrowKwargs {
    bounds: Option<Bounds>,
    check_null_coordinates: bool,
    check_finite_coordinates: bool,
}

impl Default for IsValidGeoArrowKwargs {
    fn default() -> Self {
        Self {
            bounds: None,
            check_null_coordinates: true,
            check_finite_coordinates: false,
        }
    }
}

/// See python
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IsValidOgcKwargs {
    bounds: Option<Bounds>,
    check_null_coordinates: bool,
    check_ring_self_intersect: bool,
    check_polygon_self_overlap: bool,
    check_multi_polygon_self_overlap: bool,
}

impl Default for IsValidOgcKwargs {
    fn default() -> Self {
        Self {
            bounds: None,
            check_null_coordinates: true,
            check_ring_self_intersect: true,
            check_polygon_self_overlap: true,
            check_multi_polygon_self_overlap: true,
        }
    }
}

impl IsValidOgcKwargs {
    fn topology(&self) -> MultiPolygonTopologyChecks {
        MultiPolygonTopologyChecks {
            polygons: PolygonTopologyChecks {
                check_ring_self_intersect: self.check_ring_self_intersect,
                check_polygon_self_overlap: self.check_polygon_self_overlap,
            },
            check_multi_polygon_self_overlap: self.check_multi_polygon_self_overlap,
        }
    }
}

/// Which validity to check, with its checks.
enum Validity {
    GeoArrow(CoordinateSelection),
    Ogc {
        bounds: Option<Bounds>,
        topology: MultiPolygonTopologyChecks,
    },
}

fn compute_err(e: impl std::fmt::Display) -> PolarsError {
    polars_err!(ComputeError: "is_valid_ogc {e}")
}

/// [`are_valid`] for one chunk of the storage of a `kind`.
///
/// If `check_null_coordinates=true`, every null value below a row is first moved out to the row
/// (see [geopolars_compute::inner_nulls]), so that the views can trust the rest.
fn chunk_is_valid(
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

fn are_valid(
    geometries: &Series,
    check_null_coordinates: bool,
    validity: &Validity,
) -> PolarsResult<BooleanChunked> {
    let kind = describe(geometries.dtype())?.kind;
    let storage = geometries.ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_is_valid(chunk.as_ref(), kind, check_null_coordinates, validity));
    BooleanChunked::try_from_chunk_iter(geometries.name().clone(), chunks)
}

fn are_valid_geoarrow(
    geometries: &Series,
    kwargs: &IsValidGeoArrowKwargs,
) -> PolarsResult<BooleanChunked> {
    let non_finite = NonFinite {
        select_non_finite: true,
        select_empty: false,
        ignore_z: false,
        ignore_m: false,
    };
    let selection = CoordinateSelection {
        non_finite: kwargs.check_finite_coordinates.then_some(non_finite),
        bounds: kwargs.bounds,
    };
    are_valid(
        geometries,
        kwargs.check_null_coordinates,
        &Validity::GeoArrow(selection),
    )
}

fn are_valid_ogc(geometries: &Series, kwargs: &IsValidOgcKwargs) -> PolarsResult<BooleanChunked> {
    let validity = Validity::Ogc {
        bounds: kwargs.bounds,
        topology: kwargs.topology(),
    };
    are_valid(geometries, kwargs.check_null_coordinates, &validity)
}

/// For [`is_valid_geoarrow`] and [`is_valid_ogc`]: a `bool` per geometry.
fn kernel_output_type_is_valid(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(Field::new(field.name().clone(), DataType::Boolean))
}

/// See `is_valid_geoarrow`.
#[polars_expr(output_type_func=kernel_output_type_is_valid)]
fn is_valid_geoarrow(inputs: &[Series], kwargs: IsValidGeoArrowKwargs) -> PolarsResult<Series> {
    Ok(are_valid_geoarrow(&inputs[0], &kwargs)?.into_series())
}

/// See `is_valid_ogc`.
#[polars_expr(output_type_func=kernel_output_type_is_valid)]
fn is_valid_ogc(inputs: &[Series], kwargs: IsValidOgcKwargs) -> PolarsResult<Series> {
    Ok(are_valid_ogc(&inputs[0], &kwargs)?.into_series())
}

/// `geometries`, null where `keep` is `false`.
/// `keep`'s values alone say what stays: a null row has to be `false` in them.
/// On a column where every row stays, it copies nothing.
fn keep_rows(geometries: &Series, keep: &BooleanChunked) -> PolarsResult<Series> {
    if keep.all() {
        return Ok(geometries.clone());
    }
    let geo = describe(geometries.dtype())?;
    let storage = geometries.ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .zip(keep.downcast_iter())
        .map(|(chunk, keep)| chunk.with_validity(Some(keep.values().clone())))
        .collect();
    // SAFETY: only the outer validity changed, so every chunk is still the storage.
    let storage = unsafe {
        Series::from_chunks_and_dtype_unchecked(storage.name().clone(), chunks, storage.dtype())
    };
    Ok(storage.into_extension(geo.typ.clone()))
}

/// See `null_out_invalid_geoarrow`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn null_out_invalid_geoarrow(
    inputs: &[Series],
    kwargs: IsValidGeoArrowKwargs,
) -> PolarsResult<Series> {
    // A null row is `false` in the values.
    keep_rows(&inputs[0], &are_valid_geoarrow(&inputs[0], &kwargs)?)
}

/// See `null_out_invalid_ogc`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn null_out_invalid_ogc(inputs: &[Series], kwargs: IsValidOgcKwargs) -> PolarsResult<Series> {
    // A null row is `false` in the values.
    keep_rows(&inputs[0], &are_valid_ogc(&inputs[0], &kwargs)?)
}
