//! The line spanning the Hausdorff distance between two geometries.
//! Implements the Python `hausdorff_line_between`.

use crate::common_arguments::OnRowErrorDef;
use geopolars_compute::native::NonFinite;
use geopolars_compute::OnRowError;
use geopolars_core::{hausdorff_lines_to_one, pairwise_hausdorff_lines};
use geopolars_views::Kind;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{
    describe, ensure_same_crs, geometry_series, kind_of, single_chunk_pairs, Geo,
    GeometryPairs,
};

/// The keyword arguments of the Python `hausdorff_line_between`.
/// Its `non_finite` arrives as `select_non_finite` and `select_empty`,
/// which select the coordinates that fail their row.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HausdorffLineBetweenKwargs {
    select_non_finite: bool,
    select_empty: bool,
    ignore_z: bool,
    ignore_m: bool,
    #[serde(with = "OnRowErrorDef")]
    on_row_error: OnRowError,
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

fn hausdorff_line_between_type(a: &DataType, b: &DataType) -> PolarsResult<Geo> {
    let (a, b) = (describe(a)?, describe(b)?);
    ensure_same_crs(&a, &b)?;
    let dim = a.dim.shared_with(b.dim);
    Ok(Geo::new(Kind::LineString, dim, a.metadata.clone()))
}

fn kernel_output_type_hausdorff_line_between(
    input_fields: &[Field],
    _kwargs: HausdorffLineBetweenKwargs,
) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "hausdorff_line_between takes two geometries, got {} inputs", input_fields.len());
    };
    Ok(Field::new(
        a.name().clone(),
        hausdorff_line_between_type(a.dtype(), b.dtype())?.dtype(),
    ))
}

#[polars_expr(output_type_func_with_kwargs=kernel_output_type_hausdorff_line_between)]
fn hausdorff_line_between(
    inputs: &[Series],
    kwargs: HausdorffLineBetweenKwargs,
) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let lines = hausdorff_line_between_type(a.dtype(), b.dtype())?;
    let name = a.name().clone();
    let (a_kind, b_kind) = (kind_of(a)?, kind_of(b)?);
    let (failing, on_row_error) = (kwargs.failing(), kwargs.on_row_error);
    let against_one =
        |one: ArrayRef, one_kind: Kind, many: Vec<ArrayRef>, many_kind: Kind, one_is_a| {
            many.iter()
                .map(|many| {
                    hausdorff_lines_to_one(
                        one.as_ref(),
                        one_kind,
                        many.as_ref(),
                        many_kind,
                        one_is_a,
                        failing,
                        on_row_error,
                    )
                })
                .collect::<PolarsResult<Vec<_>>>()
        };
    let chunks = match single_chunk_pairs(a, b)? {
        GeometryPairs::Pairwise(pairs) => pairs
            .iter()
            .map(|(a, b)| {
                pairwise_hausdorff_lines(
                    a.as_ref(),
                    a_kind,
                    b.as_ref(),
                    b_kind,
                    failing,
                    on_row_error,
                )
            })
            .collect::<PolarsResult<Vec<_>>>()?,
        GeometryPairs::OneAgainstMany { a: one, b: many } => {
            against_one(one, a_kind, many, b_kind, true)?
        }
        GeometryPairs::ManyAgainstOne { a: many, b: one } => {
            against_one(one, b_kind, many, a_kind, false)?
        }
        GeometryPairs::AllNull(lengths) => {
            return Ok(Series::full_null(
                name,
                lengths.iter().sum(),
                &lines.dtype(),
            ));
        }
    };
    geometry_series(name, lines, chunks)
}
