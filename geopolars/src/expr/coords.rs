//! Shared functionality for expressions is often implemented over coordinates,
//! the shared building block of geometries.

use polars::prelude::*;

use crate::geoarrow::describe;

/// For an operation that hands back the geometry it got.
pub fn kernel_output_type_wrapper_same_geometry(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(field.clone())
}
