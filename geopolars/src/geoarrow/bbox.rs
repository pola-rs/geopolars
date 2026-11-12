//! The `geoarrow.box` extension type: one axis-aligned rectangle per row.
//! (The module is not called `box`, which is a keyword.)

use std::any::Any;
use std::borrow::Cow;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use polars_core::datatypes::extension::{
    ExtensionTypeFactory, ExtensionTypeImpl, ExtensionTypeInstance,
};
use polars_core::prelude::DataType;

use super::crs::ExtensionMetadata;
use super::GeoDimension;

/// The `ARROW:extension:name` a box is registered under.
/// The Python side must spell the same name.
pub const NAME: &str = "geoarrow.box";

/// A `geoarrow.box` column's type.
///
/// Its storage is a flat struct of bounds rather than nested coordinates,
/// so it is deliberately not a [`Kind`](geopolars_views::Kind):
/// [`describe`](super::describe) refuses it, and so does every expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GeoBox {
    dim: GeoDimension,
    // Behind an `Arc` because dtypes get cloned a lot during planning.
    metadata: Arc<ExtensionMetadata>,
}

impl GeoBox {
    pub fn new(dim: GeoDimension, metadata: Arc<ExtensionMetadata>) -> Self {
        Self { dim, metadata }
    }

    /// Produces a Polars dtype.
    pub fn dtype(self) -> DataType {
        let storage = self.dim.box_storage();
        DataType::Extension(ExtensionTypeInstance(Box::new(self)), Box::new(storage))
    }
}

impl ExtensionTypeImpl for GeoBox {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(NAME)
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
        let dim = self.dim.tag();
        Cow::Owned(match self.metadata.crs_label() {
            Some(crs) => format!("box[{dim}] {crs}"),
            None => format!("box[{dim}]"),
        })
    }

    /// Shown as `BoxXY`, or `BoxXY({"crs":"EPSG:4326"})` with metadata.
    fn dyn_debug(&self) -> Cow<'_, str> {
        let name = format!("Box{}", self.dim.tag().to_uppercase());
        Cow::Owned(match self.metadata.serialize() {
            Some(metadata) => format!("{name} (meta={metadata})"),
            None => name,
        })
    }
}

/// A box whose storage is not the spec's struct of bounds,
/// or whose metadata is not a JSON object.
/// The counterpart of [`Unsupported`](super::geo::Unsupported).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnsupportedBox {
    storage: DataType,
    /// Kept verbatim, as it might not parse.
    metadata: Option<String>,
}

impl ExtensionTypeImpl for UnsupportedBox {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(NAME)
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
        Cow::Borrowed("box[?]")
    }

    fn dyn_debug(&self) -> Cow<'_, str> {
        Cow::Owned(format!("UnsupportedBox({:?})", self.storage))
    }
}

/// Reads the dimension off a box's storage.
/// `None` if this is not a layout we recognise.
/// The same rules as [`coord::dimension_of`](super::coord::dimension_of), over the bounds' names.
pub fn dimension_of(storage: &DataType) -> Option<GeoDimension> {
    let DataType::Struct(fields) = storage else {
        return None;
    };
    if !fields
        .iter()
        .all(|f| matches!(f.dtype(), DataType::Float64))
    {
        return None;
    }
    GeoDimension::ALL.into_iter().find(|dim| {
        let names = dim.box_field_names();
        fields.len() == names.len()
            && fields
                .iter()
                .zip(names)
                .all(|(f, name)| f.name().as_str() == *name)
    })
}

pub struct BoxFactory;

impl ExtensionTypeFactory for BoxFactory {
    fn create_type_instance(
        &self,
        _name: &str,
        storage: &DataType,
        metadata: Option<&str>,
    ) -> Box<dyn ExtensionTypeImpl> {
        match (dimension_of(storage), ExtensionMetadata::parse(metadata)) {
            (Some(dim), Ok(parsed)) => Box::new(GeoBox::new(dim, Arc::new(parsed))),
            _ => Box::new(UnsupportedBox {
                storage: storage.clone(),
                metadata: metadata.map(str::to_owned),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Only what the Python suite cannot reach.

    use super::*;
    use polars_core::prelude::Field;

    fn struct_of(names: &[&str]) -> DataType {
        DataType::Struct(
            names
                .iter()
                .map(|n| Field::new((*n).into(), DataType::Float64))
                .collect(),
        )
    }

    #[test]
    fn storage_round_trips_through_dimension_of() {
        for dim in GeoDimension::ALL {
            assert_eq!(dimension_of(&dim.box_storage()), Some(dim));
        }
    }

    #[test]
    fn unrecognized_layouts_are_rejected() {
        // Wrong order: the spec puts every minimum before every maximum.
        assert_eq!(
            dimension_of(&struct_of(&["xmin", "xmax", "ymin", "ymax"])),
            None
        );
        assert_eq!(
            dimension_of(&struct_of(&[
                "xmin", "ymin", "mmin", "zmin", "xmax", "ymax", "mmax", "zmax"
            ])),
            None
        );
        // A point's coordinates are not bounds.
        assert_eq!(dimension_of(&GeoDimension::XY.coordinates()), None);
        // Right names, wrong type.
        let f32_xy = DataType::Struct(
            GeoDimension::XY
                .box_field_names()
                .iter()
                .map(|n| Field::new((*n).into(), DataType::Float32))
                .collect(),
        );
        assert_eq!(dimension_of(&f32_xy), None);
        assert_eq!(dimension_of(&DataType::Float64), None);
    }
}
