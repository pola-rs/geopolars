//! The view over a `geoarrow.box` column: per row, a range on every axis.

use polars_arrow::array::{Array, PrimitiveArray, StructArray};
use polars_arrow::bitmap::Bitmap;
use polars_buffer::Buffer;
use polars_core::prelude::{polars_err, PolarsResult};

use crate::{downcast, GeoDimension};

/// A column of boxes, with null rows recorded in the struct's own validity.
/// Like [`Coords`](crate::Coords), it trusts that a bound is only null under a null row.
pub struct BoxArray {
    validity: Option<Bitmap>,
    /// Every minimum, then every maximum, in the order of [`GeoDimension::box_field_names`].
    bounds: Vec<Buffer<f64>>,
    dim: GeoDimension,
}

impl BoxArray {
    pub fn new(chunk: &dyn Array) -> PolarsResult<Self> {
        let boxes: &StructArray = downcast(chunk, "a box struct")?;
        Ok(Self {
            validity: boxes.validity().cloned(),
            bounds: (bound_arrays(boxes)?.iter())
                .map(|bound| bound.values().clone())
                .collect(),
            dim: box_dimension(boxes)?,
        })
    }

    pub(crate) fn with_validity(self, validity: Option<Bitmap>) -> Self {
        Self { validity, ..self }
    }

    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        self.bounds[0].len()
    }

    #[inline]
    pub fn dim(&self) -> GeoDimension {
        self.dim
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

fn box_dimension(boxes: &StructArray) -> PolarsResult<GeoDimension> {
    let names: Vec<&str> = boxes.fields().iter().map(|f| f.name.as_str()).collect();
    GeoDimension::ALL
        .into_iter()
        .find(|dim| dim.box_field_names() == names.as_slice())
        .ok_or_else(|| polars_err!(ComputeError: "expected the fields of a box, got: {names:?}"))
}
