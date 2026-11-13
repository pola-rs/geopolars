use geopolars_compute::inner_nulls::propagate_inner_nulls_of_multi_line_strings;
use geopolars_compute::native::{
    multi_line_strings_joined, multi_line_strings_split_at_gaps, Coincidence, Gaps,
};
use geopolars_views::{GeometryArray, UncheckedStorage};
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
/// The result is a multilinestring with [`JoinGaps::Split`],
/// and a linestring otherwise.
pub fn join_line_strings(
    chunk: &dyn Array,
    coincidence: Coincidence,
    unique_z: bool,
    unique_m: bool,
    gaps: JoinGaps,
    drop_repeated_joins: bool,
) -> PolarsResult<GeometryArray> {
    let array = propagate_inner_nulls_of_multi_line_strings(UncheckedStorage::new(chunk)?);
    let drop = drop_repeated_joins;
    let joined =
        |gaps| multi_line_strings_joined(&array, coincidence, unique_z, unique_m, gaps, drop);
    Ok(match gaps {
        JoinGaps::Null => GeometryArray::LineString(joined(Gaps::Null)),
        JoinGaps::Bridge => GeometryArray::LineString(joined(Gaps::Bridge)),
        JoinGaps::Split => GeometryArray::MultiLineString(multi_line_strings_split_at_gaps(
            &array,
            coincidence,
            unique_z,
            unique_m,
            drop,
        )),
    })
}
