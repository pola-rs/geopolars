use geopolars_compute::inner_nulls::propagate_inner_nulls_of_multi_line_strings;
use geopolars_compute::native::{
    multi_line_strings_joined, multi_line_strings_split_at_gaps, Coincidence, Gaps,
};
use geopolars_views::UncheckedStorage;
use polars_arrow::array::Array;
use polars_core::prelude::PolarsResult;

/// `gaps` of the Python `join_line_strings`.
#[derive(Debug, Clone, Copy)]
pub enum JoinGaps {
    Null,
    Bridge,
    Split,
}

/// Joins the linestrings on every row of one chunk of a multilinestring column's storage
/// (or of a list of linestrings, which is the same storage).
/// Implements the Python `join_line_strings`.
/// The result is the storage of a multilinestring with [`JoinGaps::Split`],
/// and of a linestring otherwise.
pub fn join_line_strings(
    chunk: &dyn Array,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    gaps: JoinGaps,
    drop_repeated_joins: bool,
) -> PolarsResult<Box<dyn Array>> {
    let array = propagate_inner_nulls_of_multi_line_strings(UncheckedStorage::new(chunk)?);
    let drop = drop_repeated_joins;
    let joined =
        |gaps| multi_line_strings_joined(&array, coincidence, unique_z, unique_m, gaps, drop);
    Ok(match gaps {
        JoinGaps::Null => joined(Gaps::Null).into_list_array().boxed(),
        JoinGaps::Bridge => joined(Gaps::Bridge).into_list_array().boxed(),
        JoinGaps::Split => {
            multi_line_strings_split_at_gaps(&array, coincidence, unique_z, unique_m, drop)
                .into_list_array()
                .boxed()
        }
    })
}
