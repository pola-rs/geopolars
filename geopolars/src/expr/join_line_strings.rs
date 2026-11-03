//! Joining a list of linestrings, or the parts of a multilinestring, into one linestring,
//! through `geopolars_compute::native::joined_line_strings`
//! and `geopolars_compute::native::line_strings_joined_where_coinciding`.
//! See `join_line_strings` in Python for the rules.

use geopolars_compute::inner_nulls::propagate_inner_nulls_of_multi_line_strings;
use geopolars_compute::native::{joined_line_strings, line_strings_joined_where_coinciding, Gaps};
use geopolars_core::UncheckedStorage;
use polars::prelude::*;
use polars_arrow::array::Array;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::construct::storage_of;
use super::has_repeats_in_succession::tolerance;
use crate::geoarrow::{describe, Geo, Kind};

/// `gaps` of `join_line_strings`.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum GapsKwarg {
    Null,
    Bridge,
    Split,
}

/// See python
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JoinLineStringsKwargs {
    abs_tol: Option<f64>,
    rel_tol: Option<f64>,
    unique_z: bool,
    unique_m: bool,
    gaps: GapsKwarg,
    drop_repeated_joins: bool,
}

/// The geometry `join_line_strings` gives back:
/// a linestring, or a multilinestring with `gaps="split"`.
fn joined_type(dtype: &DataType, kwargs: &JoinLineStringsKwargs) -> PolarsResult<Geo> {
    let parts = match dtype {
        DataType::List(inner) => describe(inner)
            .ok()
            .filter(|geo| geo.kind == Kind::LineString),
        dtype => describe(dtype)
            .ok()
            .filter(|geo| geo.kind == Kind::MultiLineString),
    };
    let Some(parts) = parts else {
        polars_bail!(
            SchemaMismatch: "join_line_strings expects a list of `linestring`s or a `multilinestring`, got: {}",
            dtype
        );
    };
    let kind = match kwargs.gaps {
        GapsKwarg::Split => Kind::MultiLineString,
        GapsKwarg::Null | GapsKwarg::Bridge => Kind::LineString,
    };
    Ok(Geo::new(kind, parts.dim, parts.metadata.clone()))
}

fn kernel_output_type_join_line_strings(
    input_fields: &[Field],
    kwargs: JoinLineStringsKwargs,
) -> PolarsResult<Field> {
    let geo = joined_type(input_fields[0].dtype(), &kwargs)?;
    Ok(Field::new(input_fields[0].name().clone(), geo.dtype()))
}

/// [`join_line_strings`] for one chunk of the storage of a list of linestrings.
fn chunk_join_line_strings(
    chunk: &dyn Array,
    kwargs: &JoinLineStringsKwargs,
) -> PolarsResult<Box<dyn Array>> {
    let array = propagate_inner_nulls_of_multi_line_strings(UncheckedStorage::new(chunk)?);
    let tolerance = tolerance(kwargs.abs_tol, kwargs.rel_tol);
    let (unique_z, unique_m, drop) = (kwargs.unique_z, kwargs.unique_m, kwargs.drop_repeated_joins);
    let joined = |gaps| joined_line_strings(&array, tolerance, unique_z, unique_m, gaps, drop);
    Ok(match kwargs.gaps {
        GapsKwarg::Null => joined(Gaps::Null).into_list_array().boxed(),
        GapsKwarg::Bridge => joined(Gaps::Bridge).into_list_array().boxed(),
        GapsKwarg::Split => {
            line_strings_joined_where_coinciding(&array, tolerance, unique_z, unique_m, drop)
                .into_list_array()
                .boxed()
        }
    })
}

/// See `join_line_strings`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_join_line_strings)]
fn join_line_strings(inputs: &[Series], kwargs: JoinLineStringsKwargs) -> PolarsResult<Series> {
    let geo = joined_type(inputs[0].dtype(), &kwargs)?;
    let chunks = storage_of(&inputs[0])?
        .chunks()
        .iter()
        .map(|chunk| chunk_join_line_strings(chunk.as_ref(), &kwargs))
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the chunks are the storage of `geo`, of the input's dimension.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &geo.dtype())
    })
}
