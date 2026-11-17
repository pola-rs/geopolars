const_enum!(
    /// What a kernel does with a polygon it would give an empty exterior but a hole with vertices.
    /// Implements `empty_polygon_exterior`, which `docs/Common arguments.md` documents.
    #[derive(Default)]
    pub enum EmptyPolygonExterior {
        Keep,
        #[default]
        Null,
        Error,
        Empty,
    }
);
