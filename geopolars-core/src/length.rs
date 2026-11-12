use std::fmt::Display;

use geopolars_compute::native::{line_string_lengths, multi_line_string_lengths, LengthMetric};
use geopolars_compute::with_const_flags;
use geopolars_views::{downcast, LineStringArray, MultiLineStringArray};
use polars_arrow::array::{Array, PrimitiveArray};
use polars_core::prelude::{polars_err, PolarsResult};

use crate::LineKind;

/// Measures the length of every linestring or multilinestring in one chunk of a `kind` column's storage,
/// as `metric` measures it.
/// Implements the Python `length`.
pub fn lengths(
    chunk: &dyn Array,
    kind: LineKind,
    metric: &impl LengthMetric<Error: Display>,
    ignore_nan: bool,
) -> PolarsResult<PrimitiveArray<f64>> {
    match kind {
        LineKind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            let z = array.coords.dim().has_z();
            with_const_flags!(
                Z = z, IGNORE_NAN = ignore_nan;
                line_string_lengths::<Z, IGNORE_NAN, _>(metric, &array)
            )
        }
        LineKind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            let z = array.coords.dim().has_z();
            with_const_flags!(
                Z = z, IGNORE_NAN = ignore_nan;
                multi_line_string_lengths::<Z, IGNORE_NAN, _>(metric, &array)
            )
        }
    }
    .map_err(|e| polars_err!(ComputeError: "{e}"))
}
