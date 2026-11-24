//! The view over a `geoarrow.box` column: per row, a range on every axis.

use polars_arrow::array::{Array, PrimitiveArray, StructArray};
use polars_arrow::bitmap::Bitmap;
use polars_buffer::Buffer;
use polars_core::prelude::PolarsResult;

use crate::downcast;

/// A column of boxes, with null rows recorded in the struct's own validity.
/// Like [`Coords`](crate::Coords), it trusts that a bound is only null under a null row.
pub struct BoxArray {
    validity: Option<Bitmap>,
    /// Every minimum, then every maximum,
    /// in the order of [`GeoDimension::box_field_names`](crate::GeoDimension::box_field_names).
    bounds: Vec<Buffer<f64>>,
}

impl BoxArray {
    pub fn new(chunk: &dyn Array) -> PolarsResult<Self> {
        let boxes: &StructArray = downcast(chunk, "a box struct")?;
        let bounds = bound_arrays(boxes)?;
        Ok(Self::from_bounds(boxes.validity().cloned(), &bounds))
    }

    pub(crate) fn from_bounds(validity: Option<Bitmap>, bounds: &[&PrimitiveArray<f64>]) -> Self {
        Self {
            validity,
            bounds: bounds.iter().map(|bound| bound.values().clone()).collect(),
        }
    }

    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        self.bounds[0].len()
    }

    /// `None` when every box is there.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// How many axes each box has a range on.
    #[inline]
    pub fn axes(&self) -> usize {
        self.bounds.len() / 2
    }

    /// The minimum of `axis` per row, an axis being an index into the box's dimension.
    #[inline]
    pub fn min(&self, axis: usize) -> &[f64] {
        &self.bounds[axis]
    }

    /// The maximum of `axis` per row, as [`min`](Self::min).
    #[inline]
    pub fn max(&self, axis: usize) -> &[f64] {
        &self.bounds[self.axes() + axis]
    }
}

/// The bounds of `boxes`, with their own validity, in field order.
pub(crate) fn bound_arrays(boxes: &StructArray) -> PolarsResult<Vec<&PrimitiveArray<f64>>> {
    (boxes.values().iter())
        .map(|bound| downcast(bound.as_ref(), "f64 bounds"))
        .collect()
}
