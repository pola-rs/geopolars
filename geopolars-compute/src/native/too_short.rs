//! Linestrings and rings that an operation left too short.

/// See `too_short` of `remove_repeats_in_succession` in Python.
#[derive(Debug, Clone, Copy)]
pub enum TooShort {
    Null,
    Keep,
    Empty,
}
