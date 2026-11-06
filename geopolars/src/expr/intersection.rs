//! The intersection of two geometries
//! (see [`geopolars_compute::geos::IntersectionPairs`]).

use geopolars_compute::geos::IntersectionPairs;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::IntoPolarsResult;
use crate::geoarrow::{
    describe, ensure_same_crs, geometry_array, kind_of, single_chunk_masked_pairs, Geo,
    GeoDimension, GeometryPairs, Kind,
};

/// The dimension of the parts an intersection keeps.
/// Ordered by dimension, from points up to polygons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Parts {
    Points,
    Lines,
    Polygons,
}

impl Parts {
    fn name(self) -> &'static str {
        match self {
            Parts::Points => "points",
            Parts::Lines => "lines",
            Parts::Polygons => "polygons",
        }
    }
}

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntersectionKwargs {
    keep: Option<Parts>,
    allow_splitting: bool,
    ccw_outer: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

fn dimension(kind: Kind) -> Parts {
    match kind {
        Kind::Point | Kind::MultiPoint => Parts::Points,
        Kind::LineString | Kind::MultiLineString => Parts::Lines,
        Kind::Polygon | Kind::MultiPolygon => Parts::Polygons,
    }
}

fn output_kind(parts: Parts, multi: bool) -> Kind {
    match (parts, multi) {
        (Parts::Points, false) => Kind::Point,
        (Parts::Points, true) => Kind::MultiPoint,
        (Parts::Lines, false) => Kind::LineString,
        (Parts::Lines, true) => Kind::MultiLineString,
        (Parts::Polygons, false) => Kind::Polygon,
        (Parts::Polygons, true) => Kind::MultiPolygon,
    }
}

/// Works out the output type of the Python `intersection`.
fn intersection_type(a: &DataType, b: &DataType, kwargs: &IntersectionKwargs) -> PolarsResult<Geo> {
    let (a, b) = (describe(a)?, describe(b)?);
    ensure_same_crs(&a, &b)?;
    let lowest = dimension(a.kind).min(dimension(b.kind));
    let parts = match kwargs.keep {
        None => lowest,
        Some(keep) => {
            // Python refuses a higher `keep` itself, so it never gets here.
            polars_ensure!(
                keep <= lowest,
                InvalidOperation: "the intersection of a {} and a {} cannot keep {}",
                a.kind.display(), b.kind.display(), keep.name()
            );
            keep
        }
    };
    // A single lowest-dimensional input stays single, unless splitting is allowed.
    // A point cannot split, so it stays single regardless.
    let has_point = a.kind == Kind::Point || b.kind == Kind::Point;
    let single_lowest = [a.kind, b.kind]
        .into_iter()
        .any(|kind| dimension(kind) == lowest && !kind.is_multi());
    let multi = !has_point && (kwargs.allow_splitting || !single_lowest);
    // m is dropped.
    let dim = if a.dim.has_z() || b.dim.has_z() {
        GeoDimension::XYZ
    } else {
        GeoDimension::XY
    };
    Ok(Geo::new(output_kind(parts, multi), dim, a.metadata.clone()))
}

/// For [`intersection`]: a geometry (see [`intersection_type`]), named after the first one.
fn kernel_output_type_intersection(
    input_fields: &[Field],
    kwargs: IntersectionKwargs,
) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "intersection takes two geometries, got {} inputs", input_fields.len());
    };
    let geo = intersection_type(a.dtype(), b.dtype(), &kwargs)?;
    Ok(Field::new(a.name().clone(), geo.dtype()))
}

/// The intersections of `pairs`, as the storage of `geo`'s kind and dimension.
fn intersected_storage(
    pairs: IntersectionPairs,
    geo: &Geo,
    ccw_outer: bool,
) -> PolarsResult<ArrayRef> {
    let z = geo.dim().has_z();
    match geo.kind() {
        Kind::Point => pairs
            .into_points(z)
            .map(|array| array.into_struct_array().boxed()),
        Kind::MultiPoint => pairs
            .into_multi_points(z)
            .map(|array| array.into_list_array().boxed()),
        Kind::LineString => pairs
            .into_line_strings(z)
            .map(|array| array.into_list_array().boxed()),
        Kind::MultiLineString => pairs
            .into_multi_line_strings(z)
            .map(|array| array.into_list_array().boxed()),
        Kind::Polygon => pairs
            .into_polygons(z, ccw_outer)
            .map(|array| array.into_list_array().boxed()),
        Kind::MultiPolygon => pairs
            .into_multi_polygons(z, ccw_outer)
            .map(|array| array.into_list_array().boxed()),
    }
    .into_polars()
}

/// See `intersection`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_intersection)]
fn intersection(inputs: &[Series], kwargs: IntersectionKwargs) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let geo = intersection_type(a.dtype(), b.dtype(), &kwargs)?;
    let chunk_pairs = match single_chunk_masked_pairs(a, b)? {
        GeometryPairs::Pairwise(pairs) => pairs,
        // `IntersectionPairs` pairs a single row with every row itself,
        // so the single geometry stays on its own side.
        GeometryPairs::AgainstOne {
            one,
            many,
            one_is_a,
        } => many
            .into_iter()
            .map(|many| {
                if one_is_a {
                    (one.clone(), many)
                } else {
                    (many, one.clone())
                }
            })
            .collect(),
        GeometryPairs::AllNull(lengths) => {
            let rows = lengths.iter().sum();
            return Ok(Series::full_null(a.name().clone(), rows, &geo.dtype()));
        }
    };
    let (a_kind, b_kind) = (kind_of(a)?, kind_of(b)?);
    let (ignore, early) = (kwargs.ignore_per_row_errors, kwargs.early_finite_check);
    let chunks = chunk_pairs
        .iter()
        .map(|(a, b)| {
            let (a, b) = (
                geometry_array(a.as_ref(), a_kind)?,
                geometry_array(b.as_ref(), b_kind)?,
            );
            let pairs = IntersectionPairs::new(&a, &b, ignore, early).into_polars()?;
            intersected_storage(pairs, &geo, kwargs.ccw_outer)
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    // SAFETY: `intersected_storage` picks the conversion from `geo`'s kind and dimension.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(a.name().clone(), chunks, &geo.dtype()) })
}
