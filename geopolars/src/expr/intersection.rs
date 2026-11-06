//! The intersection of two geometries
//! (see [`geopolars_compute::geos::intersection_pairwise`]).

use geopolars_compute::geos::{
    intersection_against_one, intersection_pairwise, IntersectionOutput, IntersectionParts,
};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::IntoPolarsResult;
use crate::geoarrow::{
    describe, ensure_same_crs, map_geometry_pairs, storage_as_single_chunk, Geo, GeoDimension, Kind,
};

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Keep {
    Points,
    Lines,
    Polygons,
}

impl Keep {
    fn name(self) -> &'static str {
        match self {
            Keep::Points => "points",
            Keep::Lines => "lines",
            Keep::Polygons => "polygons",
        }
    }
}

impl From<Keep> for IntersectionParts {
    fn from(keep: Keep) -> Self {
        match keep {
            Keep::Points => Self::Points,
            Keep::Lines => Self::Lines,
            Keep::Polygons => Self::Polygons,
        }
    }
}

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntersectionKwargs {
    keep: Option<Keep>,
    allow_splitting: bool,
    ccw_outer: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

fn dimension(kind: Kind) -> IntersectionParts {
    match kind {
        Kind::Point | Kind::MultiPoint => IntersectionParts::Points,
        Kind::LineString | Kind::MultiLineString => IntersectionParts::Lines,
        Kind::Polygon | Kind::MultiPolygon => IntersectionParts::Polygons,
    }
}

fn output_kind(parts: IntersectionParts, multi: bool) -> Kind {
    match (parts, multi) {
        (IntersectionParts::Points, false) => Kind::Point,
        (IntersectionParts::Points, true) => Kind::MultiPoint,
        (IntersectionParts::Lines, false) => Kind::LineString,
        (IntersectionParts::Lines, true) => Kind::MultiLineString,
        (IntersectionParts::Polygons, false) => Kind::Polygon,
        (IntersectionParts::Polygons, true) => Kind::MultiPolygon,
    }
}

/// Works out the output type of the Python `intersection`,
/// and the [`IntersectionOutput`] the kernel has to give for it.
fn intersection_type(
    a: &DataType,
    b: &DataType,
    kwargs: &IntersectionKwargs,
) -> PolarsResult<(DataType, IntersectionOutput)> {
    let (a, b) = (describe(a)?, describe(b)?);
    ensure_same_crs(&a, &b)?;
    let lowest = dimension(a.kind).min(dimension(b.kind));
    let parts = match kwargs.keep {
        None => lowest,
        Some(keep) => {
            let parts = keep.into();
            // Python refuses a higher `keep` itself, so it never gets here.
            polars_ensure!(
                parts <= lowest,
                InvalidOperation: "the intersection of a {} and a {} cannot keep {}",
                a.kind.display(), b.kind.display(), keep.name()
            );
            parts
        }
    };
    // A single lowest-dimensional input stays single, unless splitting is allowed.
    // A point cannot split, so it stays single regardless.
    let has_point = a.kind == Kind::Point || b.kind == Kind::Point;
    let single_lowest = [a.kind, b.kind]
        .into_iter()
        .any(|kind| dimension(kind) == lowest && !kind.is_multi());
    let multi = !has_point && (kwargs.allow_splitting || !single_lowest);
    let z = a.dim.has_z() || b.dim.has_z();
    let dim = if z {
        GeoDimension::XYZ
    } else {
        GeoDimension::XY
    };
    let dtype = Geo::new(output_kind(parts, multi), dim, a.metadata.clone()).dtype();
    let output = IntersectionOutput {
        parts,
        multi,
        z,
        ccw_outer: kwargs.ccw_outer,
    };
    Ok((dtype, output))
}

/// For [`intersection`]: a geometry (see [`intersection_type`]), named after the first one.
fn kernel_output_type_intersection(
    input_fields: &[Field],
    kwargs: IntersectionKwargs,
) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "intersection takes two geometries, got {} inputs", input_fields.len());
    };
    let (dtype, _) = intersection_type(a.dtype(), b.dtype(), &kwargs)?;
    Ok(Field::new(a.name().clone(), dtype))
}

/// See `intersection`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_intersection)]
fn intersection(inputs: &[Series], kwargs: IntersectionKwargs) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let (dtype, output) = intersection_type(a.dtype(), b.dtype(), &kwargs)?;
    let (ignore, early) = (kwargs.ignore_per_row_errors, kwargs.early_finite_check);
    let chunks = map_geometry_pairs(
        a,
        b,
        |a, b| intersection_pairwise(a, b, output, ignore, early).into_polars(),
        |one, many, one_is_a| {
            intersection_against_one(one, many, one_is_a, output, ignore, early).into_polars()
        },
        |rows| storage_as_single_chunk(&Series::full_null(PlSmallStr::EMPTY, rows, &dtype)),
    )?;
    // SAFETY: the kernel gives the storage of the kind and dimension `dtype` says.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(a.name().clone(), chunks, &dtype) })
}
