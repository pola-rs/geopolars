//! The arrays kernels build their output in.
//! A row is null where the given validity says, and at the rows in `failed`.

use polars_arrow::array::{Array, BooleanArray, ListArray, PrimitiveArray, StructArray};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::compute::utils::combine_validities_and3;
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::offset::OffsetsBuffer;
use polars_buffer::Buffer;
use polars_core::prelude::CompatLevel;

use crate::GeoDimension;

/// How many coordinates and parts an output column will hold, to allocate its buffers once.
/// `parts` has one count per list level below the rows, innermost first:
/// a polygon's rings, or a multipolygon's rings and then its polygons.
/// It may hold more levels than the kind built has.
/// Then only the first ones are used.
/// The rows are not counted, the caller knows how many there are.
/// Being off costs reallocations or unused memory, never correctness.
pub struct SizeHint<const N: usize> {
    pub coords: usize,
    pub parts: [usize; N],
}

/// Every value of the x, y, z and m axes, in coordinate order.
/// z and m are `None` for a dimension without them.
pub type OwnedAxes = (Vec<f64>, Vec<f64>, Option<Vec<f64>>, Option<Vec<f64>>);

/// [`OwnedAxes`], as the buffers a [`ValidCoords`](crate::ValidCoords) holds.
pub(crate) type BufferAxes = (
    Buffer<f64>,
    Buffer<f64>,
    Option<Buffer<f64>>,
    Option<Buffer<f64>>,
);

/// `values` has one entry per row. `validity` is `None` when every row is there.
pub fn f64_array(values: Vec<f64>, validity: Option<&Bitmap>) -> PrimitiveArray<f64> {
    PrimitiveArray::from_vec(values).with_validity(validity.cloned())
}

/// [f64_array], with a row null where `a` or `b` is, and at the rows in `failed`.
pub fn f64_array_masked(
    values: Vec<f64>,
    a: Option<&Bitmap>,
    b: Option<&Bitmap>,
    failed: &[usize],
) -> PrimitiveArray<f64> {
    let measured = succeeded(values.len(), failed);
    let validity = combine_validities_and3(a, b, measured.as_ref());
    PrimitiveArray::from_vec(values).with_validity(validity)
}

/// [f64_array], for a `bool` per row.
pub fn bool_array(values: Vec<bool>, validity: Option<&Bitmap>) -> BooleanArray {
    BooleanArray::from_slice(values).with_validity(validity.cloned())
}

/// The coordinate struct of `dim`, taking over the `axes` buffers.
/// `validity` is `None` when every coordinate is there.
/// Polars expects a null coordinate to be null in every field too,
/// (not needed for geoarrow spec), so they all get it.
pub(crate) fn coords_struct(
    dim: GeoDimension,
    (x, y, z, m): BufferAxes,
    validity: Option<Bitmap>,
) -> StructArray {
    let len = x.len();
    let axes = [Some(x), Some(y), z, m]
        .into_iter()
        .flatten()
        .map(|values| PrimitiveArray::new(ArrowDataType::Float64, values, validity.clone()).boxed())
        .collect();
    StructArray::new(
        dim.coordinates().to_arrow(CompatLevel::newest()),
        len,
        axes,
        validity,
    )
}

/// The struct of a `geoarrow.box` of `dim`.
/// `bounds` holds one array per field, in the order of
/// [`box_field_names`](GeoDimension::box_field_names).
/// `validity` is `None` when every box is there.
/// Like [`coords_struct`], a null box is null in every field too.
pub fn box_struct(
    dim: GeoDimension,
    bounds: Vec<Vec<f64>>,
    validity: Option<Bitmap>,
) -> StructArray {
    let len = bounds[0].len();
    let fields = bounds
        .into_iter()
        .map(|values| {
            PrimitiveArray::from_vec(values)
                .with_validity(validity.clone())
                .boxed()
        })
        .collect();
    StructArray::new(
        dim.box_storage().to_arrow(CompatLevel::newest()),
        len,
        fields,
        validity,
    )
}

/// `values` split into lists at `offsets`.
/// `validity` is `None` when every list is there.
pub fn list_array(
    values: Box<dyn Array>,
    offsets: OffsetsBuffer<i64>,
    validity: Option<Bitmap>,
) -> ListArray<i64> {
    ListArray::<i64>::new(
        ListArray::<i64>::default_datatype(values.dtype().clone()),
        offsets,
        values,
        validity,
    )
}

/// Every one of `len` rows but those in `failed`.
/// `None` when nothing failed.
fn succeeded(len: usize, failed: &[usize]) -> Option<Bitmap> {
    (!failed.is_empty()).then(|| {
        let mut succeeded = MutableBitmap::from_len_set(len);
        for &i in failed {
            succeeded.set(i, false);
        }
        Bitmap::from(succeeded)
    })
}
