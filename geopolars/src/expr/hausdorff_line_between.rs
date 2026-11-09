//! The line spanning the Hausdorff distance between two geometries
//! (see [`geopolars_compute::native::hausdorff_lines_pairwise`]).

use geopolars_compute::native::{
    hausdorff_lines_against_one, hausdorff_lines_pairwise, HausdorffOptions,
};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::IntoPolarsResult;
use crate::geoarrow::{
    describe, ensure_same_crs, geometry_array, kind_of, storage_as_single_chunk, Geo, Kind,
};

/// See python. Its `non_finite` arrives as `skip_empty` and `skip_non_finite`.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HausdorffLineBetweenKwargs {
    skip_empty: bool,
    skip_non_finite: bool,
    ignore_z: bool,
    ignore_m: bool,
    ignore_per_row_errors: bool,
}

impl HausdorffLineBetweenKwargs {
    fn options(&self) -> HausdorffOptions {
        HausdorffOptions {
            skip_empty: self.skip_empty,
            skip_non_finite: self.skip_non_finite,
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

/// Runs [hausdorff_lines_against_one] on every chunk of `many`.
fn every_hausdorff_line_against_one(
    one: &Series,
    many: &Series,
    lines_start_on_one: bool,
    kwargs: &HausdorffLineBetweenKwargs,
) -> PolarsResult<Vec<ArrayRef>> {
    let one = geometry_array(storage_as_single_chunk(one)?.as_ref(), kind_of(one)?)?;
    let kind = kind_of(many)?;
    many.ext()?
        .storage()
        .chunks()
        .iter()
        .map(|many| {
            let many = geometry_array(many.as_ref(), kind)?;
            let lines = hausdorff_lines_against_one(
                &one,
                &many,
                lines_start_on_one,
                kwargs.options(),
                kwargs.ignore_per_row_errors,
            )
            .into_polars()?;
            Ok(lines.into_list_array().boxed())
        })
        .collect()
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
    let chunks = match (a.len(), b.len()) {
        // The kinds can differ, but `aligned_chunks` needs two storages of one physical type,
        // so both sides are made a single chunk instead.
        (n, m) if n == m => {
            let a = geometry_array(storage_as_single_chunk(a)?.as_ref(), kind_of(a)?)?;
            let b = geometry_array(storage_as_single_chunk(b)?.as_ref(), kind_of(b)?)?;
            let lines =
                hausdorff_lines_pairwise(&a, &b, kwargs.options(), kwargs.ignore_per_row_errors)
                    .into_polars()?;
            vec![lines.into_list_array().boxed()]
        }
        (1, _) => every_hausdorff_line_against_one(a, b, true, &kwargs)?,
        (_, 1) => every_hausdorff_line_against_one(b, a, false, &kwargs)?,
        (n, m) => polars_bail!(ShapeMismatch: "cannot pair {n} geometries with {m}"),
    };
    // SAFETY: the lines are linestrings with the axes both sides have, as `dtype` says.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(name, chunks, &dtype) })
}
