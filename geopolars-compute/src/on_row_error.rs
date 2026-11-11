const_enum!(
    /// What a kernel does with a row it fails on.
    /// Implements `on_row_error`, which `docs/Common arguments.md` documents.
    pub enum OnRowError {
        Null,
        Drop,
        Raise,
    }
);
