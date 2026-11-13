//! Linestrings and rings that an operation left too short.

/// The `too_short` of the Python `remove_repeats_in_succession`, `filter_coordinates`,
/// and `fix_out_of_bounds`.
#[derive(Debug, Clone, Copy)]
pub enum TooShort {
    Null,
    Keep,
    Empty,
}
