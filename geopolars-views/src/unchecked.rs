//! A view over storage that might not hold to GeoArrow's rule on null values:
//! nulls only at the outermost level.
//! This is the one place that reads validity below the rows.

use polars_arrow::array::{Array, ListArray, StructArray};
use polars_arrow::compute::utils::combine_validities_and;
use polars_core::prelude::PolarsResult;

use crate::{downcast, AxisArrays, Coords, List, ValidCoords};

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

    /// This storage, with every level sliced to what its rows use (see the README on trimming).
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
