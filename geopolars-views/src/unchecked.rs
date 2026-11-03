//! A view over storage that might not hold to GeoArrow's rule on null values:
//! nulls only at the outermost level.
//! This is the one place that reads validity below the rows.

use polars_arrow::array::{Array, ListArray, StructArray};
use polars_arrow::compute::utils::combine_validities_and;
use polars_core::prelude::PolarsResult;

use crate::boxes::bound_arrays;
use crate::{downcast, AxisArrays, BoxArray, Coords, List, ValidCoords};

/// A chunk of storage as it comes from Polars, which might have null values below the rows.
/// Use it only to move the null values below the rows out to the rows
/// (see `inner_nulls` in geopolars-compute).
pub struct UncheckedStorage {
    /// The list levels above the coordinates, outermost first. Empty for points.
    pub lists: Vec<List>,
    /// Null where the coordinate struct is, or one of its axes.
    pub coords: Coords,
}

impl UncheckedStorage {
    pub fn new(chunk: &dyn Array) -> PolarsResult<Self> {
        let mut lists = Vec::new();
        let mut level = chunk;
        while let Some(list) = level.as_any().downcast_ref::<ListArray<i64>>() {
            lists.push(List::new(list));
            level = list.values().as_ref();
        }

        let coords: &StructArray = downcast(level, "a coordinate struct")?;
        let axes = AxisArrays::new(coords)?;
        // Polars also nulls every axis under a null coordinate, which changes nothing here.
        let validity = [Some(axes.x), Some(axes.y), axes.z, axes.m]
            .into_iter()
            .flatten()
            .fold(coords.validity().cloned(), |validity, axis| {
                combine_validities_and(validity.as_ref(), axis.validity())
            });
        let coords = Coords::from_valid(ValidCoords::new(coords)?, validity);
        Ok(Self { lists, coords }.trimmed())
    }

    fn trimmed(self) -> Self {
        let mut items = 0..self.lists.first().map_or(self.coords.len(), List::len);
        let lists = (self.lists.iter())
            .map(|list| {
                let (list, inner) = list.sliced(items.clone());
                items = inner;
                list
            })
            .collect();
        Self {
            lists,
            coords: self.coords.sliced(items),
        }
    }
}

/// A chunk of boxes as it comes from Polars, which might have null bounds under a valid row.
/// Use it only to move those nulls out to the rows (see `inner_nulls` in geopolars-compute).
pub struct UncheckedBoxes {
    /// Null where the box struct is, or one of its bounds.
    pub boxes: BoxArray,
}

impl UncheckedBoxes {
    pub fn new(chunk: &dyn Array) -> PolarsResult<Self> {
        let boxes: &StructArray = downcast(chunk, "a box struct")?;
        let validity = (bound_arrays(boxes)?.into_iter())
            .fold(boxes.validity().cloned(), |validity, bound| {
                combine_validities_and(validity.as_ref(), bound.validity())
            });
        Ok(Self {
            boxes: BoxArray::new(chunk)?.with_validity(validity),
        })
    }
}
