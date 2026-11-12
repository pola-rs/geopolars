//! Adding parts to geometries, from another column.
//! See `add_parts` in Python for the rules.

use geopolars_core::{EmptyParts, NullParts};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::construct::{
    fields_of, listed_parts, paired_rows, part_kinds, parts_storage, NullPartsDef,
};
use crate::geoarrow::{describe, ensure_same_crs_unless_bare, Geo};

#[derive(Deserialize)]
#[serde(remote = "EmptyParts", rename_all = "snake_case")]
enum EmptyPartsDef {
    Keep,
    Drop,
}

/// The keyword arguments of `add_parts`.
/// These have to be mirrored on the Python side.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AddPartsKwargs {
    #[serde(with = "NullPartsDef")]
    null_parts: NullParts,
    #[serde(with = "EmptyPartsDef")]
    empty_parts: EmptyParts,
}

/// Checks the input fields of `add_parts`.
/// Returns the type of the result, and whether `parts` holds a list of parts per row.
fn parts_added(input_fields: &[Field]) -> PolarsResult<(Geo, bool)> {
    let geo = describe(input_fields[0].dtype())?;
    let (part, multi) = part_kinds(geo.kind);
    let names = ("geometry", "parts");
    let (listed, metadata) = listed_parts(&input_fields[1], part, geo.dim, names)?;
    ensure_same_crs_unless_bare(Some(geo.metadata), metadata.as_deref(), names)?;
    Ok((Geo::new(multi, geo.dim, geo.metadata.clone()), listed))
}

fn kernel_output_type_add_parts(input_fields: &[Field]) -> PolarsResult<Field> {
    let (geo, _) = parts_added(input_fields)?;
    Ok(Field::new(input_fields[0].name().clone(), geo.dtype()))
}

/// See `add_parts`.
#[polars_expr(output_type_func=kernel_output_type_add_parts)]
fn add_parts(inputs: &[Series], kwargs: AddPartsKwargs) -> PolarsResult<Series> {
    let (geo, listed) = parts_added(&fields_of(inputs))?;
    let kind = describe(inputs[0].dtype())?.kind;
    let base = inputs[0].ext()?.storage();
    let parts = parts_storage(&inputs[1], geo.kind(), geo.dim())?;
    let (base, parts) = paired_rows(base, &parts, ("geometries", "parts"))?;

    let chunk = geopolars_core::add_parts(
        base.chunks()[0].as_ref(),
        kind,
        parts.chunks()[0].as_ref(),
        listed,
        kwargs.null_parts,
        kwargs.empty_parts,
    )?;
    // SAFETY: the chunk is the storage of a multi-geometry of `geo`'s kind and dimension.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), vec![chunk], &geo.dtype())
    })
}
