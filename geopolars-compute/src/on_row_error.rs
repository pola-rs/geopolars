const_enum!(
    /// What a kernel does with a row it fails on.
    /// Implements `on_row_error`, which `docs/Common arguments.md` documents.
    /// Dropping the row is not an option,
    /// as the kernels are registered with `is_elementwise=True`.
    /// Dropping rows changes the length of the column,
    /// which breaks `with_columns`, broadcasting a single geometry against a column,
    /// and pairing with other columns of the same frame.
    pub enum OnRowError {
        Null,
        Raise,
    }
);
