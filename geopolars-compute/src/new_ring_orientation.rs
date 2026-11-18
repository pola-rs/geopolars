const_enum!(
    /// Which way a kernel turns the rings it makes.
    /// Implements `new_ring_orientation`, which `docs/Common arguments.md` documents.
    pub enum NewRingOrientation {
        Keep,
        Ccw,
        Cw,
    }
);

const_enum!(
    /// Whether the holes that [enum@NewRingOrientation] orients run against the exterior or along it.
    /// Implements `new_holes`, which `docs/Common arguments.md` documents.
    pub enum NewHoles {
        Opposite,
        Same,
    }
);
