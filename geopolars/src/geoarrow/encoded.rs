//! The `geoarrow.wkb` and `geoarrow.wkt` extension types.

use std::any::Any;
use std::borrow::Cow;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use polars::prelude::*;
use polars_core::datatypes::extension::{
    ExtensionTypeFactory, ExtensionTypeImpl, ExtensionTypeInstance,
};

use super::crs::ExtensionMetadata;

/// How the geometries of an [`Encoded`] column are serialised, one value per row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// [Well-Known Binary](https://libgeos.org/specifications/wkb/), over `Binary`.
    Wkb,
    /// [Well-Known Text](https://libgeos.org/specifications/wkt/), over `String`.
    Wkt,
}

impl Encoding {
    pub const ALL: [Encoding; 2] = [Encoding::Wkb, Encoding::Wkt];

    /// The `ARROW:extension:name` the encoding is registered under.
    /// The Python side must spell the same name.
    pub const fn name(self) -> &'static str {
        match self {
            Encoding::Wkb => "geoarrow.wkb",
            Encoding::Wkt => "geoarrow.wkt",
        }
    }

    pub const fn storage(self) -> DataType {
        match self {
            Encoding::Wkb => DataType::Binary,
            Encoding::Wkt => DataType::String,
        }
    }

    /// How the encoding is shown in a DataFrame header: `wkb`.
    pub const fn display(self) -> &'static str {
        match self {
            Encoding::Wkb => "wkb",
            Encoding::Wkt => "wkt",
        }
    }

    /// How the encoding is spelled in an error message: `WKB`.
    pub const fn label(self) -> &'static str {
        match self {
            Encoding::Wkb => "WKB",
            Encoding::Wkt => "WKT",
        }
    }

    /// How the storage is spelled in an error message: `binary`.
    const fn storage_name(self) -> &'static str {
        match self {
            Encoding::Wkb => "binary",
            Encoding::Wkt => "string",
        }
    }
}

/// A `geoarrow.wkb` or `geoarrow.wkt` column's type.
///
/// These are for exchange only (e.g. GeoParquet).
/// They are deliberately not a [`Kind`](geopolars_views::Kind): no expression operates on them,
/// other than converting them to and from the native geometries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Encoded {
    encoding: Encoding,
    // Behind an [`Arc`], since dtypes get cloned a lot during planning.
    metadata: Arc<ExtensionMetadata>,
}

impl Encoded {
    pub fn new(encoding: Encoding, metadata: Arc<ExtensionMetadata>) -> Self {
        Self { encoding, metadata }
    }

    pub fn instance(self) -> ExtensionTypeInstance {
        ExtensionTypeInstance(Box::new(self))
    }

    pub fn dtype(self) -> DataType {
        let storage = self.encoding.storage();
        DataType::Extension(self.instance(), Box::new(storage))
    }
}

/// Polars arrow extension type trait impl
impl ExtensionTypeImpl for Encoded {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.encoding.name())
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

    fn dyn_display(&self) -> Cow<'_, str> {
        let enc = self.encoding.display();
        if let Some(crs) = self.metadata.crs_label() {
            Cow::Owned(format!("{enc} {crs}"))
        } else {
            Cow::Borrowed(self.encoding.display())
        }
    }

    /// Shown as `WKB`, or `WKB({"crs":"EPSG:4326"})` with metadata.
    fn dyn_debug(&self) -> Cow<'_, str> {
        let name = self.encoding.label();
        match self.metadata.serialize() {
            Some(metadata) => Cow::Owned(format!("{name} (meta={metadata})")),
            None => Cow::Borrowed(name),
        }
    }
}

/// An [`Encoded`] type whose storage is not the encoding's, or whose metadata is not a JSON object.
/// The counterpart of [`Unsupported`](super::geo::Unsupported).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnsupportedEncoded {
    encoding: Encoding,
    storage: DataType,
    /// Kept verbatim, as it might not parse.
    metadata: Option<String>,
}

impl ExtensionTypeImpl for UnsupportedEncoded {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.encoding.name())
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
        Cow::Owned(format!("{}[?]", self.encoding.display()))
    }

    fn dyn_debug(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "Unsupported{}({:?})",
            self.encoding.label(),
            self.storage
        ))
    }
}

pub struct EncodedFactory(pub Encoding);

impl ExtensionTypeFactory for EncodedFactory {
    fn create_type_instance(
        &self,
        _name: &str,
        storage: &DataType,
        metadata: Option<&str>,
    ) -> Box<dyn ExtensionTypeImpl> {
        match ExtensionMetadata::parse(metadata) {
            Ok(parsed) if *storage == self.0.storage() => {
                Box::new(Encoded::new(self.0, Arc::new(parsed)))
            }
            _ => Box::new(UnsupportedEncoded {
                encoding: self.0,
                storage: storage.clone(),
                metadata: metadata.map(str::to_owned),
            }),
        }
    }
}

/// The metadata of a column holding `encoding`:
/// a `geoarrow.wkb`/`geoarrow.wkt` column's own, or none for its plain storage.
pub fn describe_encoded(
    dtype: &DataType,
    encoding: Encoding,
) -> PolarsResult<Arc<ExtensionMetadata>> {
    match dtype {
        DataType::Extension(typ, _) if typ.name() == encoding.name() => {
            match (typ.0.as_ref() as &dyn Any).downcast_ref::<Encoded>() {
                Some(encoded) => Ok(encoded.metadata.clone()),
                None => polars_bail!(
                    SchemaMismatch: "expected a `{}` column over {} storage, got: {}",
                    encoding.name(), encoding.storage_name(), dtype
                ),
            }
        }
        _ if *dtype == encoding.storage() => Ok(Default::default()),
        _ => polars_bail!(
            SchemaMismatch: "expected a `{}` or {} column, got: {}",
            encoding.name(), encoding.storage_name(), dtype
        ),
    }
}
