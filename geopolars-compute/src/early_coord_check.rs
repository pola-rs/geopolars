const_enum!(
    /// How a kernel checks coordinates in a pass before its work,
    /// handing a row with one to `on_row_error`.
    /// Implements `early_coord_check`, which `docs/Common arguments.md` documents.
    pub enum EarlyCoordCheck {
        Off,
        Finite,
        FiniteAndEmpty,
    }
);
