//! Views over the list offsets of a geometry column.

use std::ops::Range;

use polars_arrow::array::ListArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::offset::OffsetsBuffer;

/// Lists that are all assumed to be valid:
/// the inner levels of a nested geometry (the linestrings of a multilinestring).
/// Otherwise, use [List].
///
/// See the README on views.
#[derive(Clone)]
pub struct ValidList {
    offsets: OffsetsBuffer<i64>,
}

impl ValidList {
    pub fn new(list: &ListArray<i64>) -> Self {
        Self {
            offsets: list.offsets().clone(),
        }
    }

    pub fn from_offsets(offsets: OffsetsBuffer<i64>) -> Self {
        Self { offsets }
    }

    pub(crate) fn into_offsets(self) -> OffsetsBuffer<i64> {
        self.offsets
    }

    #[inline]
    pub fn offsets(&self) -> &OffsetsBuffer<i64> {
        &self.offsets
    }

    /// How many lists there are.
    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len_proxy()
    }

    /// Where list `i`'s items sit in the child array.
    #[inline(always)]
    pub fn range(&self, i: usize) -> Range<usize> {
        let (start, end) = self.offsets.start_end(i);
        start..end
    }

    /// Here the items of lists `lists` sit in the child array.
    #[inline(always)]
    pub fn span(&self, lists: Range<usize>) -> Range<usize> {
        let offsets = self.offsets.buffer();
        offsets[lists.start] as usize..offsets[lists.end] as usize
    }

    /// The [range](Self::range) of every list, in order.
    #[inline(always)]
    pub fn ranges(&self) -> impl Iterator<Item = Range<usize>> + Clone + '_ {
        (0..self.len()).map(move |i| self.range(i))
    }

    /// Whether these lists hold all `items` of the child array, from the first one on.
    /// See the README on trimming.
    #[inline]
    pub fn spans_all_of(&self, items: usize) -> bool {
        *self.offsets.first() == 0 && *self.offsets.last() as usize == items
    }

    /// The lists in range `lists`, with their offsets rebased to start at zero.
    /// Also returns the range their items take up in the child array.
    /// This copies the offsets only when the first one is not zero.
    pub(crate) fn sliced(&self, lists: Range<usize>) -> (Self, Range<usize>) {
        let mut offsets = self.offsets.clone();
        offsets.slice(lists.start, lists.len() + 1);
        let (first, last) = (*offsets.first(), *offsets.last());
        if first != 0 {
            let rebased: Vec<i64> = offsets
                .buffer()
                .iter()
                .map(|offset| offset - first)
                .collect();
            // SAFETY: subtracting the first offset keeps them growing, and starts them at zero.
            offsets = unsafe { OffsetsBuffer::new_unchecked(rebased.into()) };
        }
        (Self { offsets }, first as usize..last as usize)
    }
}

/// Lists where individual lists can be null: the outer level of a geometry column.
/// Otherwise, use [ValidList].
///
/// See the README on views.
#[derive(Clone)]
pub struct List {
    /// Which lists are there.
    validity: Option<Bitmap>,
    lists: ValidList,
}

impl List {
    pub fn new(list: &ListArray<i64>) -> Self {
        Self {
            validity: list.validity().cloned(),
            lists: ValidList::new(list),
        }
    }

    /// `validity` is `None` when every list is there.
    pub fn from_offsets(offsets: OffsetsBuffer<i64>, validity: Option<Bitmap>) -> Self {
        Self {
            validity,
            lists: ValidList::from_offsets(offsets),
        }
    }

    pub fn with_validity(self, validity: Option<Bitmap>) -> Self {
        Self { validity, ..self }
    }

    /// These lists, with the lists at `failed` null too.
    pub fn with_null_at(self, failed: &[usize]) -> Self {
        let succeeded = crate::output::succeeded(self.len(), failed);
        let validity = combine_validities_and(self.validity(), succeeded.as_ref());
        self.with_validity(validity)
    }

    /// The lists, for once it is known which are there.
    #[inline]
    pub fn ignoring_validity(&self) -> &ValidList {
        &self.lists
    }

    pub(crate) fn into_parts(self) -> (OffsetsBuffer<i64>, Option<Bitmap>) {
        (self.lists.into_offsets(), self.validity)
    }

    /// See [`ValidList::len`].
    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        self.lists.len()
    }

    /// See [`ValidList::range`].
    #[inline(always)]
    pub fn range(&self, i: usize) -> Range<usize> {
        self.lists.range(i)
    }

    /// See [`ValidList::span`].
    #[inline(always)]
    pub fn span(&self, lists: Range<usize>) -> Range<usize> {
        self.lists.span(lists)
    }

    #[inline(always)]
    pub fn is_valid(&self, i: usize) -> bool {
        self.validity
            .as_ref()
            .is_none_or(|validity| validity.get_bit(i))
    }

    /// `None` when every list is there.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// Whether every list is there.
    #[inline]
    pub fn is_all_valid(&self) -> bool {
        self.validity
            .as_ref()
            .is_none_or(|validity| validity.unset_bits() == 0)
    }

    /// See [`ValidList::sliced`].
    pub(crate) fn sliced(&self, lists: Range<usize>) -> (Self, Range<usize>) {
        let validity =
            (self.validity.clone()).map(|validity| validity.sliced(lists.start, lists.len()));
        let (lists, items) = self.lists.sliced(lists);
        (Self { validity, lists }, items)
    }
}
