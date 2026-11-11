//! The extension type every GeoArrow geometry is an instance of.
//! In this module, the polars traits [ExtensionTypeImpl] and [ExtensionTypeFactory]
//! are implemented for our extension type,
//! so that Polars can use our extension type generically.

use std::any::Any;
use std::borrow::Cow;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use geopolars_core::Kind;
use polars_core::datatypes::extension::{
    ExtensionTypeFactory, ExtensionTypeImpl, ExtensionTypeInstance,
};
use polars_core::prelude::DataType;

use super::coord::dimension_of_storage;
use super::crs::ExtensionMetadata;
use super::GeoDimension;

/// A GeoArrow geometry column's type:
/// which geometry, over which coordinates.
///
/// Both are derived from the storage once, in [`GeoFactory`];
/// everything downstream dispatches on them
/// instead of re-inspecting struct fields.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Geo {
    kind: Kind,
    dim: GeoDimension,
    /// Behind an [`Arc`], since dtypes get cloned a lot during planning.
    metadata: Arc<ExtensionMetadata>,
}

impl Geo {
    /// A geometry of the given kind and dimension, carrying `metadata`.
    pub fn new(kind: Kind, dim: GeoDimension, metadata: Arc<ExtensionMetadata>) -> Self {
        Self {
            kind,
            dim,
            metadata,
        }
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub fn dim(&self) -> GeoDimension {
        self.dim
    }

    pub fn metadata(&self) -> &Arc<ExtensionMetadata> {
        &self.metadata
    }

    /// Produces a Polars Arrow extension type from this type.
    /// Ready to label a column of the matching storage with.
    pub fn instance(self) -> ExtensionTypeInstance {
        ExtensionTypeInstance(Box::new(self))
    }

    /// Produces a Polars dtype.
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
        self.metadata.serialize().map(Cow::Owned)
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

    /// shown as `ext[point[xyzm] EPSG:4326]` table header on python side
    fn dyn_display(&self) -> Cow<'_, str> {
        let kind = self.kind.display();
        let dim = self.dim.tag();
        Cow::Owned(match self.metadata.crs_label() {
            Some(crs) => format!("{kind}[{dim}] {crs}"),
            None => format!("{kind}[{dim}]"),
        })
    }

    /// Shown as `PointXY`, or `PointXY({"crs":"EPSG:4326"})` with all json metadata.
    fn dyn_debug(&self) -> Cow<'_, str> {
        let kind = self.kind.type_name();
        let dim = self.dim.tag().to_uppercase();
        Cow::Owned(match self.metadata.serialize() {
            Some(metadata) => format!("{kind}[{dim}] (meta={metadata})"),
            None => format!("{kind}[{dim}]"),
        })
    }
}

/// A geometry whose storage this version does not implement,
/// or whose metadata is not a JSON object.
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
    /// Kept verbatim, as it might not parse.
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
        // We only have to get the dimension and metadata out once.
        // Every later downstream expression can match on the dimension.
        match (
            dimension_of_storage(storage, kind.nesting()),
            ExtensionMetadata::parse(metadata),
        ) {
            (Some(dim), Ok(parsed)) => Box::new(Geo::new(kind, dim, Arc::new(parsed))),
            _ => Box::new(Unsupported {
                kind,
                storage: storage.clone(),
                metadata: metadata.map(str::to_owned),
            }),
        }
    }
}
