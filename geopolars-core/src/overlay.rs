use geopolars_compute::geos::{Overlay, OverlayPairs};
use geopolars_views::{GeometryArray, Kind};
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

use crate::IntoPolarsResult;

/// See the Python `intersection` and `difference`.
#[derive(Debug, Clone, Copy)]
pub struct OverlayOptions {
    pub ccw_outer: bool,
    pub ignore_per_row_errors: bool,
    pub early_finite_check: bool,
}

/// Overlays the geometries on the same row of a chunk of an `a_kind` column's storage
/// and one of a `b_kind` column's, through GEOS.
/// Implements the Python `intersection` and `difference`.
/// The result is of `output_kind`, with z exactly if `z=true`.
/// See [`OverlayPairs::new`] for the lengths `a` and `b` may have.
#[allow(clippy::too_many_arguments)]
pub fn overlay(
    a: &dyn Array,
    a_kind: Kind,
    b: &dyn Array,
    b_kind: Kind,
    overlay: Overlay,
    output_kind: Kind,
    z: bool,
    options: OverlayOptions,
) -> PolarsResult<GeometryArray> {
    let (a, b) = (
        GeometryArray::new(a, a_kind)?,
        GeometryArray::new(b, b_kind)?,
    );
    let OverlayOptions {
        ccw_outer,
        ignore_per_row_errors,
        early_finite_check,
    } = options;
    let pairs = OverlayPairs::new(&a, &b, overlay, ignore_per_row_errors, early_finite_check)
        .into_polars()?;
    match output_kind {
        Kind::Point => pairs.into_points(z).map(GeometryArray::Point),
        Kind::MultiPoint => pairs.into_multi_points(z).map(GeometryArray::MultiPoint),
        Kind::LineString => pairs.into_line_strings(z).map(GeometryArray::LineString),
        Kind::MultiLineString => pairs
            .into_multi_line_strings(z)
            .map(GeometryArray::MultiLineString),
        Kind::Polygon => pairs
            .into_polygons(z, ccw_outer)
            .map(GeometryArray::Polygon),
        Kind::MultiPolygon => pairs
            .into_multi_polygons(z, ccw_outer)
            .map(GeometryArray::MultiPolygon),
    }
    .into_polars()
}
