//! Linestrings and multilinestrings:
//! the list levels and coordinates that make one up, held together.

use std::ops::Range;

use polars_arrow::array::ListArray;
use polars_core::prelude::PolarsResult;

use crate::output::{coords_struct, list_array};
use crate::{downcast, List, Untrimmed, ValidCoords, ValidList};

/// One linestring: a run of coordinates.
pub struct ScalarLineString<'a> {
    pub coords: &'a ValidCoords,
    pub vertices: Range<usize>,
}

impl<'a> ScalarLineString<'a> {
    /// The position of every vertex in order: x, y, and with `Z` its z (`0.0` without).
    #[inline(always)]
    pub fn positions<const Z: bool>(&self) -> impl Iterator<Item = [f64; 3]> + 'a {
        self.coordinates::<Z, false>().map(|[x, y, z, _]| [x, y, z])
    }

    /// [positions](Self::positions), with the m of every vertex last (`0.0` without `M`).
    #[inline(always)]
    pub fn coordinates<const Z: bool, const M: bool>(&self) -> impl Iterator<Item = [f64; 4]> + 'a {
        let vertices = self.vertices.clone();
        let x = &self.coords.x()[vertices.clone()];
        let y = &self.coords.y()[vertices.clone()];
        // `Some` exactly when `Z=true` (`M=true`), which the compiler knows,
        // so no vertex checks it.
        let z = Z.then(|| &self.coords.z().expect("a z, with `Z`")[vertices.clone()]);
        let m = M.then(|| &self.coords.m().expect("an m, with `M`")[vertices]);
        (0..x.len()).map(move |i| [x[i], y[i], z.map_or(0.0, |z| z[i]), m.map_or(0.0, |m| m[i])])
    }
}

/// One multilinestring: a run of linestrings.
pub struct ScalarMultiLineString<'a> {
    pub lines: &'a ValidList,
    pub coords: &'a ValidCoords,
    pub parts: Range<usize>,
}

impl<'a> ScalarMultiLineString<'a> {
    /// Its linestrings, in order.
    pub fn line_strings(&self) -> impl Iterator<Item = ScalarLineString<'a>> + 'a {
        let (lines, coords) = (self.lines, self.coords);
        self.parts.clone().map(move |j| ScalarLineString {
            coords,
            vertices: lines.range(j),
        })
    }
}

/// A column of linestrings: rows of coordinates.
#[non_exhaustive]
pub struct LineStringArray {
    pub rows: List,
    pub coords: ValidCoords,
}

impl LineStringArray {
    /// The view of `rows`, trimmed (see the README on trimming).
    pub fn new(rows: &ListArray<i64>) -> PolarsResult<Self> {
        Ok(Self::new_untrimmed(rows)?.trimmed())
    }

    /// [new](Self::new), without trimming (see [Untrimmed]).
    pub fn new_untrimmed(rows: &ListArray<i64>) -> PolarsResult<Untrimmed<Self>> {
        Ok(Untrimmed(Self {
            rows: List::new(rows),
            coords: ValidCoords::new(rows.values().as_ref())?,
        }))
    }

    /// The view of these parts, which have to be trimmed (see the README on trimming).
    pub fn from_parts(rows: List, coords: ValidCoords) -> Self {
        let array = Self { rows, coords };
        debug_assert!(array.is_trimmed(), "the parts of a view have to be trimmed");
        array
    }

    /// Whether every list and coordinate this holds belongs to one of its rows.
    /// A chunk sliced from a larger array is not: see the README on trimming.
    pub(crate) fn is_trimmed(&self) -> bool {
        self.rows
            .ignoring_validity()
            .spans_all_of(self.coords.len())
    }

    /// The polars storage of the column, taking over its buffers.
    pub fn into_list_array(self) -> ListArray<i64> {
        let dim = self.coords.dim();
        let coords = coords_struct(dim, self.coords.into_axes(), None);
        let (rows, validity) = self.rows.into_parts();
        list_array(coords.boxed(), rows, validity)
    }

    /// Row `i`. A null row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> ScalarLineString<'_> {
        ScalarLineString {
            coords: &self.coords,
            vertices: self.rows.range(i),
        }
    }
}

/// A column of multilinestrings: rows of linestrings of coordinates.
#[non_exhaustive]
pub struct MultiLineStringArray {
    pub rows: List,
    pub lines: ValidList,
    pub coords: ValidCoords,
}

impl MultiLineStringArray {
    /// See [`LineStringArray::new`].
    pub fn new(rows: &ListArray<i64>) -> PolarsResult<Self> {
        Ok(Self::new_untrimmed(rows)?.trimmed())
    }

    /// See [`LineStringArray::new_untrimmed`].
    pub fn new_untrimmed(rows: &ListArray<i64>) -> PolarsResult<Untrimmed<Self>> {
        let lines: &ListArray<i64> = downcast(rows.values().as_ref(), "a list of linestrings")?;
        Ok(Untrimmed(Self {
            rows: List::new(rows),
            lines: ValidList::new(lines),
            coords: ValidCoords::new(lines.values().as_ref())?,
        }))
    }

    /// See [`LineStringArray::from_parts`].
    pub fn from_parts(rows: List, lines: ValidList, coords: ValidCoords) -> Self {
        let array = Self {
            rows,
            lines,
            coords,
        };
        debug_assert!(array.is_trimmed(), "the parts of a view have to be trimmed");
        array
    }

    /// See [`LineStringArray::is_trimmed`].
    pub(crate) fn is_trimmed(&self) -> bool {
        self.rows.ignoring_validity().spans_all_of(self.lines.len())
            && self.lines.spans_all_of(self.coords.len())
    }

    /// See [`LineStringArray::into_list_array`].
    pub fn into_list_array(self) -> ListArray<i64> {
        let dim = self.coords.dim();
        let coords = coords_struct(dim, self.coords.into_axes(), None);
        let lines = list_array(coords.boxed(), self.lines.into_offsets(), None);
        let (rows, validity) = self.rows.into_parts();
        list_array(lines.boxed(), rows, validity)
    }

    /// Row `i`. A null row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> ScalarMultiLineString<'_> {
        ScalarMultiLineString {
            lines: &self.lines,
            coords: &self.coords,
            parts: self.rows.range(i),
        }
    }
}
