//! Coordinates: the inner level of every GeoArrow geometry.
//!
//! Reading a dimension back off an Arrow layout happens here and nowhere else.
//! Everything downstream matches on the [`Dimension`] instead of re-inspecting
//! struct fields.

use polars_core::prelude::DataType;

use super::Dimension;

/// Reads the dimension off a separated-coordinate storage type.
/// `None` if this is not a layout we recognise.
pub fn dimension_of(coordinates: &DataType) -> Option<Dimension> {
    let DataType::Struct(fields) = coordinates else {
        return None;
    };
    // Every coordinate is a double; a f32 "point" is a different type, not a
    // point we should silently widen.
    if !fields
        .iter()
        .all(|f| matches!(f.dtype(), DataType::Float64))
    {
        return None;
    }
    Dimension::ALL.into_iter().find(|dim| {
        let names = dim.field_names();
        fields.len() == names.len()
            && fields
                .iter()
                .zip(names)
                .all(|(f, name)| f.name().as_str() == *name)
    })
}

/// Reads the dimension off the storage of a geometry that nests its coordinates
/// `nesting` `List` layers deep. `None` if this is not a layout we recognise.
pub fn dimension_of_storage(storage: &DataType, nesting: u8) -> Option<Dimension> {
    let mut inner = storage;
    for _ in 0..nesting {
        let DataType::List(child) = inner else {
            return None;
        };
        inner = child;
    }
    dimension_of(inner)
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

    /// The path into [`Unsupported`](super::super::Geo): storage carrying one of
    /// our extension names that we do not implement, which is what reading
    /// someone else's GeoArrow file can hand us.
    ///
    /// Python cannot reach this: `ext_from_params` rejects an unrecognised layout
    /// before a column can be built, so the factories never see one from that
    /// direction.
    #[test]
    fn unrecognised_layouts_are_rejected() {
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

    /// Storage too shallow for the geometry that claims it: a `Struct` where a
    /// linestring's `List<Struct>` belongs.
    #[test]
    fn nesting_has_to_match() {
        let xy = Dimension::XY.coordinates();
        assert_eq!(dimension_of_storage(&xy, 1), None);
        assert_eq!(
            dimension_of_storage(&DataType::List(Box::new(xy)), 0),
            None
        );
    }

    /// [`Dimension::storage`] is what an expression declares as its output type,
    /// so it has to be the layout `dimension_of_storage` accepts back -- for
    /// every geometry, at the nesting that geometry claims.
    #[test]
    fn storage_round_trips_through_dimension_of() {
        for kind in Kind::ALL {
            for dim in Dimension::ALL {
                assert_eq!(
                    dimension_of_storage(&dim.storage(kind.nesting()), kind.nesting()),
                    Some(dim)
                );
            }
        }
    }
}
