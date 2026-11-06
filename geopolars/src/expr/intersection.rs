//! The intersection of two geometries
//! (see [`geopolars_compute::geos::intersection_pairwise`]).

use geopolars_compute::geos::{
    intersection_against_one, intersection_pairwise, IntersectionOutput, IntersectionParts,
};
use polars::prelude::*;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::IntoPolarsResult;
use crate::geoarrow::{
    describe, ensure_same_crs, geometry_array, kind_of, storage_as_single_chunk, Geo, GeoDimension,
    Kind,
};

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Keep {
    Points,
    Lines,
    Polygons,
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

fn is_multi(kind: Kind) -> bool {
    matches!(
        kind,
        Kind::MultiPoint | Kind::MultiLineString | Kind::MultiPolygon
    )
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
            let keep = keep.into();
            polars_ensure!(
                keep <= lowest,
                InvalidOperation: "the intersection of a {} and a {} cannot keep {keep:?}",
                a.kind.display(), b.kind.display()
            );
            keep
        }
    };
    // A single lowest-dimensional input stays single, unless splitting is allowed.
    // A point cannot split, so it stays single regardless.
    let has_point = a.kind == Kind::Point || b.kind == Kind::Point;
    let single_lowest = [a.kind, b.kind]
        .into_iter()
        .any(|kind| dimension(kind) == lowest && !is_multi(kind));
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

/// Runs [intersection_against_one] on every chunk of `many`, against the single row of `one`.
fn every_intersection_with_one(
    one: &Series,
    many: &Series,
    one_is_a: bool,
    output: IntersectionOutput,
    kwargs: &IntersectionKwargs,
) -> PolarsResult<Vec<ArrayRef>> {
    let (ignore, early) = (kwargs.ignore_per_row_errors, kwargs.early_finite_check);
    let one_chunk = storage_as_single_chunk(one)?;
    let one_is_null = one_chunk.is_null(0);
    let one = geometry_array(one_chunk.as_ref(), kind_of(one)?)?;
    let kind = kind_of(many)?;
    many.ext()?
        .storage()
        .chunks()
        .iter()
        .map(|many| {
            // A null `one` makes every row null, so `many` is not sent to GEOS, where a row could fail.
            let many = if one_is_null {
                many.with_validity(Some(Bitmap::new_zeroed(many.len())))
            } else {
                many.clone()
            };
            let many = geometry_array(many.as_ref(), kind)?;
            intersection_against_one(&one, &many, one_is_a, output, ignore, early).into_polars()
        })
        .collect()
}

/// See `intersection`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_intersection)]
fn intersection(inputs: &[Series], kwargs: IntersectionKwargs) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let (dtype, output) = intersection_type(a.dtype(), b.dtype(), &kwargs)?;
    let chunks = match (a.len(), b.len()) {
        // The kinds can differ, but `aligned_chunks` needs two storages of one physical type.
        // So both sides are made a single chunk instead.
        (n, m) if n == m => {
            let (a_chunk, b_chunk) = (storage_as_single_chunk(a)?, storage_as_single_chunk(b)?);
            // A row null on one side is made null on the other too,
            // so that it is not sent to GEOS, where it could fail.
            let validity = combine_validities_and(a_chunk.validity(), b_chunk.validity());
            let (a, b) = (
                geometry_array(
                    a_chunk.with_validity(validity.clone()).as_ref(),
                    kind_of(a)?,
                )?,
                geometry_array(b_chunk.with_validity(validity).as_ref(), kind_of(b)?)?,
            );
            let (ignore, early) = (kwargs.ignore_per_row_errors, kwargs.early_finite_check);
            vec![intersection_pairwise(&a, &b, output, ignore, early).into_polars()?]
        }
        (1, _) => every_intersection_with_one(a, b, true, output, &kwargs)?,
        (_, 1) => every_intersection_with_one(b, a, false, output, &kwargs)?,
        (n, m) => polars_bail!(ShapeMismatch: "cannot intersect {n} geometries with {m}"),
    };
    // SAFETY: the kernel gives the storage of the kind and dimension `dtype` says.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(a.name().clone(), chunks, &dtype) })
}
