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
    describe, ensure_same_crs, geometry_array, kind_of, single_chunk_masked_pairs,
    storage_as_single_chunk, Geo, GeoDimension, GeometryPairs, Kind,
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
    let (a_kind, b_kind) = (kind_of(a)?, kind_of(b)?);
    let chunks = match single_chunk_masked_pairs(a, b)? {
        GeometryPairs::Pairwise(pairs) => pairs
            .iter()
            .map(|(a, b)| {
                let (a, b) = (
                    geometry_array(a.as_ref(), a_kind)?,
                    geometry_array(b.as_ref(), b_kind)?,
                );
                intersection_pairwise(&a, &b, output, ignore, early).into_polars()
            })
            .collect::<PolarsResult<Vec<_>>>()?,
        GeometryPairs::AgainstOne {
            one,
            many,
            one_is_a,
        } => {
            let (one_kind, many_kind) = if one_is_a {
                (a_kind, b_kind)
            } else {
                (b_kind, a_kind)
            };
            let one = geometry_array(one.as_ref(), one_kind)?;
            many.iter()
                .map(|many| {
                    let many = geometry_array(many.as_ref(), many_kind)?;
                    intersection_against_one(&one, &many, one_is_a, output, ignore, early)
                        .into_polars()
                })
                .collect::<PolarsResult<Vec<_>>>()?
        }
        GeometryPairs::AllNull(lengths) => lengths
            .iter()
            .map(|&rows| {
                storage_as_single_chunk(&Series::full_null(PlSmallStr::EMPTY, rows, &dtype))
            })
            .collect::<PolarsResult<Vec<_>>>()?,
    };
    // `intersection_type` and `intersections_into_array` pick the kind separately.
    #[cfg(debug_assertions)]
    {
        let DataType::Extension(_, storage) = &dtype else {
            unreachable!("`intersection_type` gives a geometry dtype");
        };
        let storage = storage.to_arrow(CompatLevel::newest());
        for chunk in &chunks {
            debug_assert_eq!(chunk.dtype(), &storage, "the kernel gave another storage");
        }
    }
    // SAFETY: the kernel gives the storage of the kind and dimension `dtype` says.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(a.name().clone(), chunks, &dtype) })
}
