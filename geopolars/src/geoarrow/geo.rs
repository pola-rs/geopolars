//! The extension type every GeoArrow geometry is an instance of.

use std::any::Any;
use std::borrow::Cow;
use std::hash::{DefaultHasher, Hash, Hasher};

use polars_core::datatypes::extension::{
    ExtensionTypeFactory, ExtensionTypeImpl, ExtensionTypeInstance,
};
use polars_core::prelude::DataType;

use super::coord::dimension_of_storage;
use super::{Dimension, Kind};

/// A GeoArrow geometry column's type: which geometry, over which coordinates.
///
/// Both are derived from the storage once, in [`GeoFactory`];
/// everything downstream dispatches on them
/// instead of re-inspecting struct fields.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Geo {
    kind: Kind,
    dim: Dimension,

    /// The raw `ARROW:extension:metadata` string, carried through verbatim.
    ///
    /// This version does not interpret it. Keeping the bytes rather than dropping
    /// them means reading and re-writing a file that *does* carry a `crs`
    /// preserves it instead of silently losing it. When CRS lands this becomes
    /// a parsed `Metadata { crs, crs_type, edges }`.
    metadata: Option<String>,
}

impl Geo {
    /// A geometry of the given kind and dimension, carrying `metadata` verbatim.
    pub fn new(kind: Kind, dim: Dimension, metadata: Option<String>) -> Self {
        Self {
            kind,
            dim,
            metadata,
        }
    }

    /// This type, ready to label a column of the matching storage with.
    pub fn instance(self) -> ExtensionTypeInstance {
        ExtensionTypeInstance(Box::new(self))
    }

    /// The dtype a column of these geometries has.
    pub fn dtype(self) -> DataType {
        let storage = self.dim.storage(self.kind.nesting());
        DataType::Extension(self.instance(), Box::new(storage))
    }
}

impl ExtensionTypeImpl for Geo {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.kind.name())
    }

    fn serialize_metadata(&self) -> Option<Cow<'_, str>> {
        self.metadata.as_deref().map(Cow::Borrowed)
    }

    fn dyn_clone(&self) -> Box<dyn ExtensionTypeImpl> {
        Box::new(self.clone())
    }

    fn dyn_eq(&self, other: &dyn ExtensionTypeImpl) -> bool {
        (other as &dyn Any)
            .downcast_ref::<Self>()
            .is_some_and(|o| self == o)
    }

    fn dyn_hash(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }

    /// Shown as `ext[point[xyzm]]` in a DataFrame header.
    fn dyn_display(&self) -> Cow<'_, str> {
        Cow::Owned(format!("{}[{}]", self.kind.display(), self.dim.tag()))
    }

    fn dyn_debug(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "{}{}",
            self.kind.type_name(),
            self.dim.tag().to_uppercase()
        ))
    }
}

/// A geometry whose storage this version does not implement.
/// If you see this, you might have an interleaved Arrow encoding.
///
/// [`ExtensionTypeFactory::create_type_instance`] cannot return a `Result`,
/// so it has no way to reject storage it does not understand.
/// A user might want to manually read the fields or investigate,
/// so we return this rather than panic.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Unsupported {
    kind: Kind,
    storage: DataType,
    metadata: Option<String>,
}

impl ExtensionTypeImpl for Unsupported {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.kind.name())
    }

    fn serialize_metadata(&self) -> Option<Cow<'_, str>> {
        self.metadata.as_deref().map(Cow::Borrowed)
    }

    fn dyn_clone(&self) -> Box<dyn ExtensionTypeImpl> {
        Box::new(self.clone())
    }

    fn dyn_eq(&self, other: &dyn ExtensionTypeImpl) -> bool {
        (other as &dyn Any)
            .downcast_ref::<Self>()
            .is_some_and(|o| self == o)
    }

    fn dyn_hash(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }

    fn dyn_display(&self) -> Cow<'_, str> {
        Cow::Owned(format!("{}[?]", self.kind.display()))
    }

    fn dyn_debug(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "Unsupported{}({:?})",
            self.kind.type_name(),
            self.storage
        ))
    }
}

/// Builds the concrete type for one geometry, given its storage layout.
pub struct GeoFactory(pub Kind);

impl ExtensionTypeFactory for GeoFactory {
    fn create_type_instance(
        &self,
        _name: &str,
        storage: &DataType,
        metadata: Option<&str>,
    ) -> Box<dyn ExtensionTypeImpl> {
        let kind = self.0;
        let metadata = metadata.map(str::to_owned);
        // We only have to get the dimension out of the storage type once.
        // Every later downstream expression can match on the dimension.
        match dimension_of_storage(storage, kind.nesting()) {
            Some(dim) => Box::new(Geo::new(kind, dim, metadata)),
            None => Box::new(Unsupported {
                kind,
                storage: storage.clone(),
                metadata,
            }),
        }
    }
}
