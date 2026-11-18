const_enum!(
    /// What a null part does to the geometry it is added to.
    /// Implements `null_parts`, which `docs/Common arguments.md` documents.
    #[derive(Default)]
    pub enum NullParts {
        Drop,
        #[default]
        Null,
    }
);
