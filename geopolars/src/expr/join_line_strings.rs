//! Joining the linestrings of each row into one.
//! Implements the Python `join_line_strings`.

use geopolars_compute::native::Coincidence;
use geopolars_core::JoinGaps;
use geopolars_views::Kind;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::construct::storage_of;
use crate::geoarrow::{describe, geometry_series, Geo};

#[derive(Deserialize)]
#[serde(remote = "JoinGaps", rename_all = "snake_case")]
enum JoinGapsDef {
    Null,
    Bridge,
    Split,
}

/// The keyword arguments of the Python `join_line_strings`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JoinLineStringsKwargs {
    abs_tol: Option<f64>,
    rel_tol: Option<f64>,
    unique_z: bool,
    unique_m: bool,
    #[serde(with = "JoinGapsDef")]
    gaps: JoinGaps,
    drop_repeated_joins: bool,
}

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
        JoinGaps::Split => Kind::MultiLineString,
        JoinGaps::Null | JoinGaps::Bridge => Kind::LineString,
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

#[polars_expr(output_type_func_with_kwargs=kernel_output_type_join_line_strings)]
fn join_line_strings(inputs: &[Series], kwargs: JoinLineStringsKwargs) -> PolarsResult<Series> {
    let geo = joined_type(inputs[0].dtype(), &kwargs)?;
    let coincidence = Coincidence::new(kwargs.abs_tol, kwargs.rel_tol);
    let chunks = storage_of(&inputs[0])?
        .chunks()
        .iter()
        .map(|chunk| {
            geopolars_core::join_line_strings(
                chunk.as_ref(),
                coincidence,
                kwargs.unique_z,
                kwargs.unique_m,
                kwargs.gaps,
                kwargs.drop_repeated_joins,
            )
        })
        .collect::<PolarsResult<_>>()?;
    geometry_series(inputs[0].name().clone(), geo, chunks)
}
