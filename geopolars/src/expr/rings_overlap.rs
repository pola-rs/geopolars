//! Whether two rings overlap
//! (see [`geopolars_compute::geos::rings_overlap_pairwise`]).

use geopolars_compute::geos::{rings_overlap_against_one, rings_overlap_pairwise};
use geopolars_compute::with_const_flags;
use geopolars_views::{downcast, Kind, LineStringArray};
use polars::prelude::*;
use polars_arrow::array::BooleanArray;
use polars_arrow::datatypes::ArrowDataType;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{aligned_pairs, describe, ensure_same_crs, GeometryPairs};

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RingsOverlapKwargs {
    /// Whether rings that only touch overlap too.
    include_touching: bool,
    ignore_per_row_errors: bool,
}

/// For [`rings_overlap`]: a `bool`, named after the first ring.
///
/// Both have to be linestrings, in the same CRS (or neither declaring one).
fn kernel_output_type_rings_overlap(input_fields: &[Field]) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "rings_overlap takes two rings, got {} inputs", input_fields.len());
    };
    let (geo_a, geo_b) = (describe(a.dtype())?, describe(b.dtype())?);
    for (field, geo) in [(a, &geo_a), (b, &geo_b)] {
        polars_ensure!(
            geo.kind == Kind::LineString,
            SchemaMismatch: "rings_overlap expects two `{}` columns, got: {}",
            Kind::LineString.name(), field.dtype()
        );
    }
    ensure_same_crs(&geo_a, &geo_b)?;
    Ok(Field::new(a.name().clone(), DataType::Boolean))
}

fn compute_err(e: impl std::fmt::Display) -> PolarsError {
    polars_err!(ComputeError: "rings_overlap {e}")
}

fn rings(chunk: &ArrayRef) -> PolarsResult<LineStringArray> {
    LineStringArray::new(downcast(chunk.as_ref(), "a list of vertices")?)
}

fn rings_overlap_chunks<const TOUCHING: bool>(
    a: &Series,
    b: &Series,
    ignore_per_row_errors: bool,
) -> PolarsResult<Vec<BooleanArray>> {
    Ok(match aligned_pairs(a, b)? {
        GeometryPairs::Pairwise(pairs) => pairs
            .iter()
            .map(|(a, b)| {
                rings_overlap_pairwise::<TOUCHING>(&rings(a)?, &rings(b)?, ignore_per_row_errors)
                    .map_err(compute_err)
            })
            .collect::<PolarsResult<Vec<_>>>()?,
        // Overlapping goes both ways,
        // so it does not matter which side `one` came from.
        GeometryPairs::OneAgainstMany { a: one, b: many }
        | GeometryPairs::ManyAgainstOne { a: many, b: one } => {
            let one = rings(&one)?;
            many.iter()
                .map(|many| {
                    rings_overlap_against_one::<TOUCHING>(
                        &one.get(0),
                        &rings(many)?,
                        ignore_per_row_errors,
                    )
                    .map_err(compute_err)
                })
                .collect::<PolarsResult<Vec<_>>>()?
        }
        GeometryPairs::AllNull(lengths) => lengths
            .iter()
            .map(|&rows| BooleanArray::new_null(ArrowDataType::Boolean, rows))
            .collect(),
    })
}

/// See `rings_overlap`.
#[polars_expr(output_type_func=kernel_output_type_rings_overlap)]
fn rings_overlap(inputs: &[Series], kwargs: RingsOverlapKwargs) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let chunks = with_const_flags!(
        TOUCHING = kwargs.include_touching;
        rings_overlap_chunks::<TOUCHING>(a, b, kwargs.ignore_per_row_errors)
    )?;
    Ok(BooleanChunked::from_chunk_iter(a.name().clone(), chunks).into_series())
}
