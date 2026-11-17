const_enum!(
    /// Which way a kernel turns the rings it makes.
    /// Implements `new_ring_orientation`, which `docs/Common arguments.md` documents.
    #[derive(Default)]
    pub enum NewRingOrientation {
        #[default]
        Keep,
        Ccw,
        Cw,
    }
);

const_enum!(
    /// Whether the new holes run against the exterior or along it, under [NewRingOrientation].
    /// Implements `new_holes`, which `docs/Common arguments.md` documents.
    #[derive(Default)]
    pub enum NewHoles {
        #[default]
        Opposite,
        Same,
    }
);
