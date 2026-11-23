/// What a kernel does with a row holding a coordinate outside the bounds of its CRS.
/// Implements `on_out_of_bounds`, which `geopolars/docs/source/reference/common_arguments.rst` documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnOutOfBounds {
    Ignore,
    Error,
    Null,
}
