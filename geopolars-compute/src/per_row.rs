//! Running a kernel over every row of a chunk.
//!
//! # Const generic flags
//!
//! ## `ALL_VALID`
//!
//! The loop over the rows is generic over `ALL_VALID`.
//! If `ALL_VALID=true`, every row is there, so none is checked for being null.

use geopolars_views::output::{bool_array, f64_array, list_array};
use geopolars_views::{List, ValidList};
use polars_arrow::array::{BooleanArray, ListArray, PrimitiveArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::offset::Offsets;

/// `measure` of every row of a chunk, with the rows' own validity.
#[inline(always)]
pub(crate) fn per_row<E>(
    rows: &List,
    measure: impl Fn(usize) -> Result<f64, E>,
) -> Result<PrimitiveArray<f64>, E> {
    Ok(f64_array(per_row_values(rows, measure)?, rows.validity()))
}

/// [per_row], for a `bool` per row.
#[inline(always)]
pub(crate) fn try_per_row_bool<E>(
    rows: &List,
    test: impl Fn(usize) -> Result<bool, E>,
) -> Result<BooleanArray, E> {
    Ok(bool_array(per_row_values(rows, test)?, rows.validity()))
}

#[inline(always)]
pub(crate) fn per_row_bool(rows: &List, test: impl Fn(usize) -> bool) -> BooleanArray {
    let Ok(tested) = try_per_row_bool(rows, |i| Ok::<_, std::convert::Infallible>(test(i)));
    tested
}

/// [per_row], for a count per row.
#[inline(always)]
pub(crate) fn per_row_count(rows: &List, count: impl Fn(usize) -> u32) -> PrimitiveArray<u32> {
    let Ok(counts) = per_row_values(rows, |i| Ok::<_, std::convert::Infallible>(count(i)));
    PrimitiveArray::from_vec(counts).with_validity(rows.validity().cloned())
}

/// A `bool` per item, null where it is `None`.
pub(crate) fn optional_bool_array(tested: &[Option<bool>]) -> BooleanArray {
    let values = tested
        .iter()
        .map(|tested| tested.unwrap_or(false))
        .collect();
    let validity = Bitmap::from_iter(tested.iter().map(Option::is_some));
    bool_array(values, Some(&validity))
}

/// `test` of every row of a chunk, null where the row is or where `test` gives `None`.
#[inline(always)]
pub(crate) fn per_row_optional_bool<E>(
    rows: &List,
    test: impl Fn(usize) -> Result<Option<bool>, E>,
) -> Result<BooleanArray, E> {
    Ok(optional_bool_array(&per_row_values(rows, test)?))
}

/// `measure` of every row of a chunk, with a default value for a null row.
#[inline(always)]
pub(crate) fn per_row_values<T: Default, E>(
    rows: &List,
    measure: impl Fn(usize) -> Result<T, E>,
) -> Result<Vec<T>, E> {
    with_const_flags!(
        ALL_VALID = rows.is_all_valid();
        per_row_values_with::<ALL_VALID, T, E>(rows, measure)
    )
}

#[inline(always)]
fn per_row_values_with<const ALL_VALID: bool, T: Default, E>(
    rows: &List,
    measure: impl Fn(usize) -> Result<T, E>,
) -> Result<Vec<T>, E> {
    (0..rows.len())
        // A null row's offsets point anywhere, so it is not measured.
        .map(|i| {
            if ALL_VALID || rows.is_valid(i) {
                measure(i)
            } else {
                Ok(T::default())
            }
        })
        .collect()
}

/// [per_row_optional_bool], as a list per row of a chunk, one per part.
/// `parts` holds where the parts of each row sit, and `test` takes the index of a part.
pub(crate) fn per_part_optional_bool<E>(
    rows: &List,
    parts: &ValidList,
    test: impl Fn(usize) -> Result<Option<bool>, E>,
) -> Result<ListArray<i64>, E> {
    let mut tested = Vec::with_capacity(parts.len());
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    offsets.push(0);
    for i in 0..rows.len() {
        // A null row's offsets point anywhere, so it gets no parts.
        if rows.is_valid(i) {
            for part in rows.range(i) {
                tested.push(test(part)?);
            }
        }
        offsets.push(tested.len() as i64);
    }
    // SAFETY: they start at 0, and every end is the number of values so far, which only grows.
    let offsets = unsafe { Offsets::new_unchecked(offsets) }.into();
    Ok(list_array(
        optional_bool_array(&tested).boxed(),
        offsets,
        rows.validity().cloned(),
    ))
}
