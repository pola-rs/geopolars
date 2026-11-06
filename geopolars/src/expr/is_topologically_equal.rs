//! Whether two geometries are topologically equal
//! (see [`geopolars_compute::geos::topologically_equal_pairwise`]).

use geopolars_compute::geos::{
    topologically_equal_against_one, topologically_equal_pairwise, ToGeosRows,
};
use geopolars_core::{
    downcast, Coords, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::datatypes::ArrowDataType;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::IntoPolarsResult;
use crate::geoarrow::{describe, ensure_same_crs, Kind};

/// See python
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IsTopologicallyEqualKwargs {
    ignore_per_row_errors: bool,
    early_finite_check: bool,
}

/// For [`is_topologically_equal`]: a `bool`, named after the first geometry.
///
/// Both can be of any kind and dimension, in the same CRS (or neither declaring one).
fn kernel_output_type_is_topologically_equal(input_fields: &[Field]) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "is_topologically_equal takes two geometries, got {} inputs", input_fields.len());
    };
    ensure_same_crs(&describe(a.dtype())?, &describe(b.dtype())?)?;
    Ok(Field::new(a.name().clone(), DataType::Boolean))
}

/// Wraps a storage chunk of `kind` for the GEOS kernels.
fn geos_view(chunk: &dyn Array, kind: Kind) -> PolarsResult<Box<dyn ToGeosRows>> {
    let rows = |chunk| downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => Box::new(PointArray::new(Coords::new(chunk)?)),
        Kind::MultiPoint => Box::new(MultiPointArray::new(rows(chunk)?)?),
        Kind::LineString => Box::new(LineStringArray::new(rows(chunk)?)?),
        Kind::MultiLineString => Box::new(MultiLineStringArray::new(rows(chunk)?)?),
        Kind::Polygon => Box::new(PolygonArray::new(rows(chunk)?)?),
        Kind::MultiPolygon => Box::new(MultiPolygonArray::new(rows(chunk)?)?),
    })
}

fn kind_of(geometries: &Series) -> PolarsResult<Kind> {
    Ok(describe(geometries.dtype())?.kind)
}

fn storage_as_single_chunk(geometries: &Series) -> PolarsResult<ArrayRef> {
    Ok(geometries.ext()?.storage().rechunk().chunks()[0].clone())
}

/// Runs [topologically_equal_against_one] on every chunk of `many`, against the single row of `one`.
fn every_topologically_equal_to_one(
    one: &Series,
    many: &Series,
    kwargs: &IsTopologicallyEqualKwargs,
) -> PolarsResult<Vec<BooleanArray>> {
    let (ignore, early) = (kwargs.ignore_per_row_errors, kwargs.early_finite_check);
    let one_chunk = storage_as_single_chunk(one)?;
    let many_chunks = many.ext()?.storage().chunks();
    // Every result is null, so `many` is not sent to GEOS, where a row could fail.
    if one_chunk.is_null(0) {
        return Ok(many_chunks
            .iter()
            .map(|many| BooleanArray::new_null(ArrowDataType::Boolean, many.len()))
            .collect());
    }
    let one = geos_view(one_chunk.as_ref(), kind_of(one)?)?;
    let kind = kind_of(many)?;
    many_chunks
        .iter()
        .map(|many| {
            let many = geos_view(many.as_ref(), kind)?;
            topologically_equal_against_one(one.as_ref(), many.as_ref(), ignore, early)
                .into_polars()
        })
        .collect()
}

/// See `is_topologically_equal`.
#[polars_expr(output_type_func=kernel_output_type_is_topologically_equal)]
fn is_topologically_equal(
    inputs: &[Series],
    kwargs: IsTopologicallyEqualKwargs,
) -> PolarsResult<Series> {
    let (a, b) = (&inputs[0], &inputs[1]);
    let chunks = match (a.len(), b.len()) {
        // The kinds can differ, but `aligned_chunks` needs two storages of one physical type.
        // So both sides are made a single chunk instead.
        (n, m) if n == m => {
            let (a_chunk, b_chunk) = (storage_as_single_chunk(a)?, storage_as_single_chunk(b)?);
            // A row null on one side is made null on the other too,
            // so that it is not sent to GEOS, where it could fail.
            let validity = combine_validities_and(a_chunk.validity(), b_chunk.validity());
            let (a, b) = (
                geos_view(
                    a_chunk.with_validity(validity.clone()).as_ref(),
                    kind_of(a)?,
                )?,
                geos_view(b_chunk.with_validity(validity).as_ref(), kind_of(b)?)?,
            );
            let (ignore, early) = (kwargs.ignore_per_row_errors, kwargs.early_finite_check);
            vec![
                topologically_equal_pairwise(a.as_ref(), b.as_ref(), ignore, early)
                    .into_polars()?,
            ]
        }
        (1, _) => every_topologically_equal_to_one(a, b, &kwargs)?,
        (_, 1) => every_topologically_equal_to_one(b, a, &kwargs)?,
        (n, m) => polars_bail!(ShapeMismatch: "cannot compare {n} geometries against {m}"),
    };
    Ok(BooleanChunked::from_chunk_iter(a.name().clone(), chunks).into_series())
}
