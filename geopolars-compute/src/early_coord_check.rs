const_enum!(
    /// Which coordinates a kernel rejects in a pass before its work.
    /// A row with such a coordinate goes to `on_row_error`.
    /// Implements `early_coord_check`, which `geopolars/docs/source/reference/common_arguments.rst` documents.
    pub enum EarlyCoordCheck {
        Off,
        Finite,
        FiniteAndEmpty,
    }
);
