//! A true (`pass`) or false check for geometries with levels.
//!
//! A check writes a mask for the level it looks at, such as the coordinates, the rings or the rows.
//! A set bit passes, as in a validity bitmap.
//! `None` means every item at that level passes.
//! [rows_passing] folds the masks up to the rows.
//! A list passes when it passes itself, and every item in it does.

#[cfg(feature = "native")]
use geopolars_views::List;
use geopolars_views::ValidList;
#[cfg(feature = "native")]
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::compute::utils::combine_validities_and;
#[cfg(feature = "native")]
use polars_arrow::datatypes::ArrowDataType;

fn unless_all_set(passing: Option<Bitmap>) -> Option<Bitmap> {
    passing.filter(|passing| passing.unset_bits() > 0)
}

/// [lists_passing] takes the sparse way when there is fewer than 1 unset item per `SPARSE_RATIO` lists.
/// The sparse way costs a short search per unset item, the dense way a count per list.
const SPARSE_RATIO: usize = 8;

/// How many of `ends` are at most `item`.
/// `ends` grow, and `item` is usually close to the start of them.
/// So this doubles a window until it passes `item`, and only searches within it.
/// A binary search over all of `ends` would miss the cache at almost every step.
#[inline(always)]
fn ends_before(ends: &[i64], item: usize) -> usize {
    let mut window = 1;
    while window < ends.len() && ends[window - 1] as usize <= item {
        window *= 2;
    }
    ends[..window.min(ends.len())].partition_point(|&end| end as usize <= item)
}

/// Which of `lists` hold only items set in `items`.
/// `lists` have to span all of `items` (be trimmed): the sparse way finds an item's list by the offsets.
fn lists_passing(lists: &ValidList, items: &Bitmap) -> Bitmap {
    debug_assert!(
        lists.spans_all_of(items.len()),
        "the lists have to be trimmed (see \"Trimming\" in docs/View Types.md)"
    );
    if items.unset_bits() * SPARSE_RATIO < lists.len() {
        lists_passing_sparse(lists, items)
    } else {
        lists_passing_dense(lists, items)
    }
}

/// [lists_passing] by counting the unset items of every list.
fn lists_passing_dense(lists: &ValidList, items: &Bitmap) -> Bitmap {
    Bitmap::from_iter(
        lists
            .ranges()
            .map(|range| items.null_count_range(range.start, range.len()) == 0),
    )
}

/// [lists_passing] by finding the list of every unset item.
fn lists_passing_sparse(lists: &ValidList, items: &Bitmap) -> Bitmap {
    let offsets = lists.offsets().buffer();
    let mut passing = MutableBitmap::from_len_set(lists.len());
    let mut list = 0;
    for item in (!items).true_idx_iter() {
        // The unset items come in order, so the search starts at the list of the last one.
        list += ends_before(&offsets[list + 1..], item);
        passing.set(list, false);
    }
    passing.into()
}

/// The rows passing every mask in `passing`.
/// `lists` are the list levels, outermost first.
/// `passing` holds a mask per list level, in the same order.
/// Its last mask is for the items of the innermost list level (usually the coordinates).
pub(crate) fn rows_passing(lists: &[&ValidList], passing: Vec<Option<Bitmap>>) -> Option<Bitmap> {
    assert_eq!(passing.len(), lists.len() + 1, "a mask per level");
    let mut levels = passing.into_iter().rev();
    let coords = unless_all_set(levels.next().flatten());
    lists
        .iter()
        .rev()
        .zip(levels)
        .fold(coords, |items, (lists, own)| {
            let parts = items.map(|items| lists_passing(lists, &items));
            unless_all_set(combine_validities_and(own.as_ref(), parts.as_ref()))
        })
}

/// The valid rows of `rows` whose items all pass `items` (see [rows_passing]).
/// `inner_lists` are the list levels below the rows, outermost first.
/// The items are those of the innermost list level (usually the coordinates).
#[cfg(feature = "native")]
pub(crate) fn valid_rows_passing(
    rows: &List,
    inner_lists: &[&ValidList],
    items: Option<Bitmap>,
) -> Option<Bitmap> {
    let lists: Vec<&ValidList> = [rows.ignoring_validity()]
        .into_iter()
        .chain(inner_lists.iter().copied())
        .collect();
    let passing = [rows.validity().cloned()]
        .into_iter()
        .chain(inner_lists.iter().map(|_| None))
        .chain([items])
        .collect();
    rows_passing(&lists, passing)
}

/// A `bool` per row, `true` where the row is set in `passing` (see [rows_passing]).
/// `validity` is the validity of the rows.
/// A row that is null in `validity` has to be unset in `passing`.
#[cfg(feature = "native")]
pub(crate) fn passing_bool_array(
    len: usize,
    validity: Option<&Bitmap>,
    passing: Option<Bitmap>,
) -> BooleanArray {
    let values = passing.unwrap_or_else(|| Bitmap::new_with_value(true, len));
    BooleanArray::new(ArrowDataType::Boolean, values, validity.cloned())
}

#[cfg(test)]
mod tests {
    //! The Python suite mostly reaches the dense way: its columns are too short for rare failures.

    use polars_arrow::offset::OffsetsBuffer;

    use super::*;

    #[test]
    fn the_sparse_way_finds_the_lists_the_dense_way_does() {
        // Lists 1 and 4 are empty.
        let offsets = vec![0i64, 3, 3, 4, 8, 8, 10];
        let lists = ValidList::from_offsets(OffsetsBuffer::try_from(offsets).unwrap());
        // Unset: the first item of list 0, the only item of list 2,
        // and the last items of lists 3 and 5.
        let unset = [0, 3, 7, 9];
        let items = Bitmap::from_iter((0..10).map(|item| !unset.contains(&item)));

        let sparse = lists_passing_sparse(&lists, &items);

        assert_eq!(sparse, lists_passing_dense(&lists, &items));
        assert_eq!(
            sparse.iter().collect::<Vec<_>>(),
            [false, true, false, false, true, false]
        );
    }
}
