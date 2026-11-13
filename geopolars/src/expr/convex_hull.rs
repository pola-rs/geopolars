//! The convex hull of a linestring or polygon, or their multi-variants, forwarded to GEOS.

use geopolars_core::{convex_hulls, NestedKind};
use geopolars_views::{GeoDimension, Kind};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, geometry_series, Geo};

/// A point encloses nothing, so it is refused.
/// A polygon per geometry, with the input's metadata and z, but without m.
/// Also the kind of the input.
fn hull_type(dtype: &DataType) -> PolarsResult<(Geo, NestedKind)> {
    let geo = describe(dtype)?;
    let kind = NestedKind::new("convex_hull", geo.kind)?;
    let dim = if geo.dim.has_z() {
        GeoDimension::XYZ
    } else {
        GeoDimension::XY
    };
    let hull = Geo::new(Kind::Polygon, dim, geo.metadata.clone());
    Ok((hull, kind))
}

/// For [`convex_hull`].
fn kernel_output_type_convex_hull(input_fields: &[Field]) -> PolarsResult<Field> {
    let (hull, _) = hull_type(input_fields[0].dtype())?;
    Ok(Field::new(input_fields[0].name().clone(), hull.dtype()))
}

/// See python docs for details.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConvexHullKwargs {
    ignore_per_row_errors: bool,
    ccw_outer: bool,
    early_finite_check: bool,
}

/// See `convex_hull`.
#[polars_expr(output_type_func=kernel_output_type_convex_hull)]
fn convex_hull(inputs: &[Series], kwargs: ConvexHullKwargs) -> PolarsResult<Series> {
    let (hull, kind) = hull_type(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| {
            convex_hulls(
                chunk.as_ref(),
                kind,
                kwargs.ignore_per_row_errors,
                kwargs.ccw_outer,
                kwargs.early_finite_check,
            )
        })
        .collect::<PolarsResult<_>>()?;
    geometry_series(inputs[0].name().clone(), hull, chunks)
}
