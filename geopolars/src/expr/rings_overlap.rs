//! Whether two rings overlap
//! (see [`geopolars_compute::geos::rings_overlap_pairwise`]).

use geopolars_core::{pairwise_rings_overlap, rings_overlap_one};
use geopolars_views::Kind;
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

/// See `rings_overlap`.
#[polars_expr(output_type_func=kernel_output_type_rings_overlap)]
fn rings_overlap(inputs: &[Series], kwargs: RingsOverlapKwargs) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let (touching, ignore) = (kwargs.include_touching, kwargs.ignore_per_row_errors);
    let chunks = match aligned_pairs(a, b)? {
        GeometryPairs::Pairwise(pairs) => pairs
            .iter()
            .map(|(a, b)| pairwise_rings_overlap(a.as_ref(), b.as_ref(), touching, ignore))
            .collect::<PolarsResult<Vec<_>>>()?,
        // Overlapping goes both ways,
        // so it does not matter which side `one` came from.
        GeometryPairs::OneAgainstMany { a: one, b: many }
        | GeometryPairs::ManyAgainstOne { a: many, b: one } => many
            .iter()
            .map(|many| rings_overlap_one(one.as_ref(), many.as_ref(), touching, ignore))
            .collect::<PolarsResult<Vec<_>>>()?,
        GeometryPairs::AllNull(lengths) => lengths
            .iter()
            .map(|&rows| BooleanArray::new_null(ArrowDataType::Boolean, rows))
            .collect(),
    };
    Ok(BooleanChunked::from_chunk_iter(a.name().clone(), chunks).into_series())
}
