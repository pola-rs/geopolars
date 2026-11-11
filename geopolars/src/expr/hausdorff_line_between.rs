//! The line spanning the Hausdorff distance between two geometries
//! (see [`geopolars_compute::native::hausdorff_lines_pairwise`]).

use geopolars_compute::native::{hausdorff_lines_against_one, hausdorff_lines_pairwise, NonFinite};
use geopolars_compute::GeoComputeResult;
use geopolars_core::{GeometryArray, Kind, LineStringArray};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::IntoPolarsResult;
use crate::geoarrow::{
    describe, ensure_same_crs, kind_of, single_chunk_masked_pairs, Geo, GeometryPairs,
};

/// See python. Its `non_finite` arrives as `select_non_finite` and `select_empty`,
/// which select the coordinates that fail their row.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HausdorffLineBetweenKwargs {
    select_non_finite: bool,
    select_empty: bool,
    ignore_z: bool,
    ignore_m: bool,
    ignore_per_row_errors: bool,
}

impl HausdorffLineBetweenKwargs {
    fn failing(&self) -> NonFinite {
        NonFinite {
            select_non_finite: self.select_non_finite,
            select_empty: self.select_empty,
            ignore_z: self.ignore_z,
            ignore_m: self.ignore_m,
        }
    }
}

fn hausdorff_line_between_type(a: &DataType, b: &DataType) -> PolarsResult<DataType> {
    let (a, b) = (describe(a)?, describe(b)?);
    ensure_same_crs(&a, &b)?;
    let dim = a.dim.shared_with(b.dim);
    Ok(Geo::new(Kind::LineString, dim, a.metadata.clone()).dtype())
}

/// For [`hausdorff_line_between`], named after the first geometry.
fn kernel_output_type_hausdorff_line_between(input_fields: &[Field]) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "hausdorff_line_between takes two geometries, got {} inputs", input_fields.len());
    };
    Ok(Field::new(
        a.name().clone(),
        hausdorff_line_between_type(a.dtype(), b.dtype())?,
    ))
}

/// See `hausdorff_line_between`.
#[polars_expr(output_type_func=kernel_output_type_hausdorff_line_between)]
fn hausdorff_line_between(
    inputs: &[Series],
    kwargs: HausdorffLineBetweenKwargs,
) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let dtype = hausdorff_line_between_type(a.dtype(), b.dtype())?;
    let name = a.name().clone();
    let (a_kind, b_kind) = (kind_of(a)?, kind_of(b)?);
    let (failing, ignore) = (kwargs.failing(), kwargs.ignore_per_row_errors);
    let against_one =
        |one: ArrayRef, one_kind: Kind, many: Vec<ArrayRef>, many_kind: Kind, one_is_a| {
            let one = GeometryArray::new(one.as_ref(), one_kind)?;
            many.iter()
                .map(|many| {
                    let many = GeometryArray::new(many.as_ref(), many_kind)?;
                    lines_chunk(hausdorff_lines_against_one(
                        &one, &many, one_is_a, failing, ignore,
                    ))
                })
                .collect::<PolarsResult<Vec<_>>>()
        };
    let chunks = match single_chunk_masked_pairs(a, b)? {
        GeometryPairs::Pairwise(pairs) => pairs
            .iter()
            .map(|(a, b)| {
                let a = GeometryArray::new(a.as_ref(), a_kind)?;
                let b = GeometryArray::new(b.as_ref(), b_kind)?;
                lines_chunk(hausdorff_lines_pairwise(&a, &b, failing, ignore))
            })
            .collect::<PolarsResult<Vec<_>>>()?,
        GeometryPairs::OneAgainstMany { a: one, b: many } => {
            against_one(one, a_kind, many, b_kind, true)?
        }
        GeometryPairs::ManyAgainstOne { a: many, b: one } => {
            against_one(one, b_kind, many, a_kind, false)?
        }
        GeometryPairs::AllNull(lengths) => {
            return Ok(Series::full_null(name, lengths.iter().sum(), &dtype));
        }
    };
    // SAFETY: the lines are linestrings with the axes both sides have, as `dtype` says.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(name, chunks, &dtype) })
}

fn lines_chunk(lines: GeoComputeResult<LineStringArray>) -> PolarsResult<ArrayRef> {
    Ok(lines.into_polars()?.into_list_array().boxed())
}
