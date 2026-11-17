const_enum!(
    /// What a kernel does with a linestring or ring that an operation left too short.
    /// Implements `short_lines`, which `docs/Common arguments.md` documents.
    pub enum ShortLines {
        Null,
        Keep,
        Empty,
    }
);
