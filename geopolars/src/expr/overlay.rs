//! The intersection and difference of two geometries
//! (see [`geopolars_compute::geos::OverlayPairs`]).

use geopolars_compute::geos::Overlay;
use geopolars_core::OverlayOptions;
use geopolars_views::Kind;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{
    describe, ensure_same_crs, kind_of, single_chunk_masked_pairs, Geo, GeoColumn, GeoDimension,
    GeometryPairs,
};

/// The topological dimension of an input, or of the parts an intersection keeps.
/// Ordered from points up to polygons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
enum TopologicalDimension {
    Points,
    Lines,
    Polygons,
}

impl TopologicalDimension {
    fn name(self) -> &'static str {
        match self {
            TopologicalDimension::Points => "points",
            TopologicalDimension::Lines => "lines",
            TopologicalDimension::Polygons => "polygons",
        }
    }
}

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntersectionKwargs {
    keep: Option<TopologicalDimension>,
    allow_splitting: bool,
    ccw_outer: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifferenceKwargs {
    allow_splitting: bool,
    ccw_outer: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

fn topological_dimension(kind: Kind) -> TopologicalDimension {
    match kind {
        Kind::Point | Kind::MultiPoint => TopologicalDimension::Points,
        Kind::LineString | Kind::MultiLineString => TopologicalDimension::Lines,
        Kind::Polygon | Kind::MultiPolygon => TopologicalDimension::Polygons,
    }
}

fn output_kind(kept_dimension: TopologicalDimension, multi: bool) -> Kind {
    match (kept_dimension, multi) {
        (TopologicalDimension::Points, false) => Kind::Point,
        (TopologicalDimension::Points, true) => Kind::MultiPoint,
        (TopologicalDimension::Lines, false) => Kind::LineString,
        (TopologicalDimension::Lines, true) => Kind::MultiLineString,
        (TopologicalDimension::Polygons, false) => Kind::Polygon,
        (TopologicalDimension::Polygons, true) => Kind::MultiPolygon,
    }
}

/// Works out the output type of the Python `intersection`.
fn intersection_type(a: &DataType, b: &DataType, kwargs: &IntersectionKwargs) -> PolarsResult<Geo> {
    let (a, b) = (describe(a)?, describe(b)?);
    ensure_same_crs(&a, &b)?;
    let lowest = topological_dimension(a.kind).min(topological_dimension(b.kind));
    let kept_dimension = match kwargs.keep {
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
        .any(|kind| topological_dimension(kind) == lowest && !kind.is_multi());
    let multi = !has_point && (kwargs.allow_splitting || !single_lowest);
    Ok(Geo::new(
        output_kind(kept_dimension, multi),
        overlay_dimension(&a, &b),
        a.metadata.clone(),
    ))
}

/// Works out the output type of the Python `difference`.
fn difference_type(a: &DataType, b: &DataType, kwargs: &DifferenceKwargs) -> PolarsResult<Geo> {
    let (a, b) = (describe(a)?, describe(b)?);
    ensure_same_crs(&a, &b)?;
    // A point cannot split, so it stays single regardless.
    let multi = a.kind.is_multi() || (kwargs.allow_splitting && a.kind != Kind::Point);
    Ok(Geo::new(
        output_kind(topological_dimension(a.kind), multi),
        overlay_dimension(&a, &b),
        a.metadata.clone(),
    ))
}

fn overlay_dimension(a: &GeoColumn, b: &GeoColumn) -> GeoDimension {
    if a.dim.has_z() || b.dim.has_z() {
        GeoDimension::XYZ
    } else {
        GeoDimension::XY
    }
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

fn kernel_output_type_difference(
    input_fields: &[Field],
    kwargs: DifferenceKwargs,
) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "difference takes two geometries, got {} inputs", input_fields.len());
    };
    let geo = difference_type(a.dtype(), b.dtype(), &kwargs)?;
    Ok(Field::new(a.name().clone(), geo.dtype()))
}

/// Overlays the two geometries of `inputs` into `geo`, which the output type function gave.
fn overlaid(
    inputs: &[Series],
    geo: Geo,
    overlay: Overlay,
    options: OverlayOptions,
) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let chunk_pairs = match single_chunk_masked_pairs(a, b)? {
        GeometryPairs::Pairwise(pairs) => pairs,
        // `OverlayPairs` pairs a single row with every row itself,
        // so the single geometry stays on its own side.
        GeometryPairs::OneAgainstMany { a, b } => b.into_iter().map(|b| (a.clone(), b)).collect(),
        GeometryPairs::ManyAgainstOne { a, b } => a.into_iter().map(|a| (a, b.clone())).collect(),
        GeometryPairs::AllNull(lengths) => {
            let rows = lengths.iter().sum();
            return Ok(Series::full_null(a.name().clone(), rows, &geo.dtype()));
        }
    };
    let (a_kind, b_kind) = (kind_of(a)?, kind_of(b)?);
    let (kind, z) = (geo.kind(), geo.dim().has_z());
    let chunks = chunk_pairs
        .iter()
        .map(|(a, b)| {
            geopolars_core::overlay(
                a.as_ref(),
                a_kind,
                b.as_ref(),
                b_kind,
                overlay,
                kind,
                z,
                options,
            )
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    #[cfg(debug_assertions)]
    {
        let storage = geo.dim().storage(geo.kind().nesting());
        let storage = storage.to_arrow(CompatLevel::newest());
        for chunk in &chunks {
            debug_assert_eq!(chunk.dtype(), &storage, "the kernel gave another storage");
        }
    }
    // SAFETY: every chunk is the storage of `geo`'s kind, with z exactly when `geo` has it.
    // `overlay_dimension` never gives m, so z alone fixes the dimension.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(a.name().clone(), chunks, &geo.dtype()) })
}

/// See `intersection`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_intersection)]
fn intersection(inputs: &[Series], kwargs: IntersectionKwargs) -> PolarsResult<Series> {
    let geo = intersection_type(inputs[0].dtype(), inputs[1].dtype(), &kwargs)?;
    let options = OverlayOptions {
        ccw_outer: kwargs.ccw_outer,
        ignore_per_row_errors: kwargs.ignore_per_row_errors,
        early_finite_check: kwargs.early_finite_check,
    };
    overlaid(inputs, geo, Overlay::Intersection, options)
}

/// See `difference`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_difference)]
fn difference(inputs: &[Series], kwargs: DifferenceKwargs) -> PolarsResult<Series> {
    let geo = difference_type(inputs[0].dtype(), inputs[1].dtype(), &kwargs)?;
    let options = OverlayOptions {
        ccw_outer: kwargs.ccw_outer,
        ignore_per_row_errors: kwargs.ignore_per_row_errors,
        early_finite_check: kwargs.early_finite_check,
    };
    overlaid(inputs, geo, Overlay::Difference, options)
}
