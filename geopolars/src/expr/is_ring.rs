//! Whether a linestring is a ring (see [`geopolars_compute::native::line_strings_are_rings`]),
//! or whether the parts of a multilinestring chain into one
//! (see [`geopolars_compute::native::multi_line_strings_are_rings`]).

use geopolars_compute::native::{line_strings_are_rings, multi_line_strings_are_rings};
use geopolars_compute::with_const_flags;
use geopolars_views::{Kind, LineStringArray, MultiLineStringArray};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::describe;

fn line_kind(field: &Field) -> PolarsResult<Kind> {
    match describe(field.dtype())?.kind {
        kind @ (Kind::LineString | Kind::MultiLineString) => Ok(kind),
        kind @ (Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon) => {
            polars_bail!(
                SchemaMismatch: "is_ring expects a `{}` or `{}` column, got: `{}`",
                Kind::LineString.name(), Kind::MultiLineString.name(), kind.name()
            )
        }
    }
}

/// For [`is_ring`]: a `bool` per row.
fn kernel_output_type_is_ring(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    line_kind(field)?;
    Ok(Field::new(field.name().clone(), DataType::Boolean))
}

/// See `is_ring`.
#[polars_expr(output_type_func=kernel_output_type_is_ring)]
fn is_ring(inputs: &[Series]) -> PolarsResult<Series> {
    let kind = line_kind(&inputs[0].field())?;
    let storage = inputs[0].ext()?.storage();
    let name = inputs[0].name().clone();
    let chunks = storage.list()?.downcast_iter();
    if kind == Kind::LineString {
        let chunks = chunks.map(|rows| {
            let array = LineStringArray::new_untrimmed(rows)?;
            let z = array.coords().dim().has_z();
            PolarsResult::Ok(with_const_flags!(Z = z; line_strings_are_rings::<Z>(&array)))
        });
        return Ok(BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series());
    }
    let chunks = chunks.map(|rows| {
        let array = MultiLineStringArray::new_untrimmed(rows)?;
        let z = array.coords().dim().has_z();
        PolarsResult::Ok(with_const_flags!(Z = z; multi_line_strings_are_rings::<Z>(&array)))
    });
    Ok(BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series())
}
