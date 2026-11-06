//! Every GeoArrow geometry is made up of n-dimensional *Coordinates*.
//! This module contains functionality for this inner type that is used
//! by all of the other geometry types.

use polars_core::prelude::DataType;

use super::GeoDimension;

/// Reads the dimension off a separated-coordinate storage type.
/// `None` if this is not a layout we recognise.
///
/// Reading a dimension back off an Arrow layout happens here and nowhere else.
/// Everything downstream matches on the [`Dimension`] instead of re-inspecting
/// struct fields.
pub fn dimension_of(coordinates: &DataType) -> Option<GeoDimension> {
    let DataType::Struct(fields) = coordinates else {
        return None;
    };
    // GeoArrow spec says that all coordinates *have to be* doubles.
    // a f32 "point" is a different type, not a point we should silently widen.
    if !fields
        .iter()
        .all(|f| matches!(f.dtype(), DataType::Float64))
    {
        return None;
    }
    GeoDimension::ALL.into_iter().find(|dim| {
        let names = dim.field_names();
        fields.len() == names.len()
            && fields
                .iter()
                .zip(names)
                .all(|(f, name)| f.name().as_str() == *name)
    })
}

/// Reads the dimension off the storage of a geometry
/// that nests its coordinates `nesting` `List` layers deep.
/// `None` if this is not a layout we recognise.
pub fn dimension_of_storage(storage: &DataType, nesting: u8) -> Option<GeoDimension> {
    let mut inner = storage;
    for _ in 0..nesting {
        let DataType::List(child) = inner else {
            return None;
        };
        inner = child;
    }
    dimension_of(inner)
}

/// The dtype below the outer list levels of `dtype`, and how many there are.
pub fn under_lists(mut dtype: &DataType) -> (&DataType, u8) {
    let mut lists = 0;
    while let DataType::List(inner) = dtype {
        dtype = inner;
        lists += 1;
    }
    (dtype, lists)
}

#[cfg(test)]
mod tests {
    //! Only what the Python suite cannot reach.

    use super::super::Kind;
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

    /// The path into [`Unsupported`](super::super::Geo):
    /// Can be reached by reading someone's GeoArrow file using rust only (no python).
    #[test]
    fn unrecognized_layouts_are_rejected() {
        // Wrong order: the spec fixes x before y, and z before m.
        assert_eq!(dimension_of(&struct_of(&["y", "x"])), None);
        assert_eq!(dimension_of(&struct_of(&["x", "y", "m", "z"])), None);
        // Wrong names.
        assert_eq!(dimension_of(&struct_of(&["lon", "lat"])), None);
        // Too many / too few coordinates.
        assert_eq!(dimension_of(&struct_of(&["x"])), None);
        assert_eq!(dimension_of(&struct_of(&["x", "y", "z", "m", "t"])), None);
        // Right names, wrong coordinate type.
        let f32_xy = DataType::Struct(vec![
            Field::new("x".into(), DataType::Float32),
            Field::new("y".into(), DataType::Float32),
        ]);
        assert_eq!(dimension_of(&f32_xy), None);
        // Not a struct at all. An interleaved encoding lands here too.
        assert_eq!(dimension_of(&DataType::Float64), None);
    }

    /// Storage too shallow for the geometry that claims it.
    /// e.g. a `Struct` where a linestring's `List<Struct>` belongs.
    #[test]
    fn nesting_has_to_match() {
        let xy = GeoDimension::XY.coordinates();
        assert_eq!(dimension_of_storage(&xy, 1), None);
        assert_eq!(dimension_of_storage(&DataType::List(Box::new(xy)), 0), None);
    }

    #[test]
    fn storage_round_trips_through_dimension_of() {
        for kind in Kind::ALL {
            for dim in GeoDimension::ALL {
                assert_eq!(
                    dimension_of_storage(&dim.storage(kind.nesting()), kind.nesting()),
                    Some(dim)
                );
            }
        }
    }
}
