//! Whether a geometry is valid, by GeoArrow or by OGC Simple Features,
//! through [`geopolars_core::are_valid`].
//! See `is_valid_geoarrow` and `is_valid_ogc` in Python for the rules.

use geopolars_compute::native::{Bounds, CoordinateSelection, NonFinite};
use geopolars_compute::ogc_validity::{MultiPolygonTopologyChecks, PolygonTopologyChecks};
use geopolars_core::{are_valid, ValidityCheck};
use geopolars_views::GeometryArray;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use crate::geoarrow::{describe, geometry_series};

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

fn are_valid_per_chunk(
    geometries: &Series,
    check_null_coordinates: bool,
    check: &ValidityCheck,
) -> PolarsResult<BooleanChunked> {
    let kind = describe(geometries.dtype())?.kind;
    let storage = geometries.ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| are_valid(chunk.as_ref(), kind, check_null_coordinates, check));
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
    are_valid_per_chunk(
        geometries,
        kwargs.check_null_coordinates,
        &ValidityCheck::GeoArrow(selection),
    )
}

fn are_valid_ogc(geometries: &Series, kwargs: &IsValidOgcKwargs) -> PolarsResult<BooleanChunked> {
    let check = ValidityCheck::Ogc {
        bounds: kwargs.bounds,
        topology: kwargs.topology(),
    };
    are_valid_per_chunk(geometries, kwargs.check_null_coordinates, &check)
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
        .map(|(chunk, keep)| {
            let geometries = GeometryArray::new(chunk.as_ref(), geo.kind)?;
            Ok(geometries.with_validity(Some(keep.values().clone())))
        })
        .collect::<PolarsResult<_>>()?;
    geometry_series(storage.name().clone(), geo.geo(), chunks)
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
