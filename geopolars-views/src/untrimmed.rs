//! Views that may hold lists and coordinates outside their rows, and trimming them.
//! See "Trimming" in docs/View Types.md.

use crate::{
    LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PolygonArray,
    ScalarLineString, ScalarMultiLineString, ScalarMultiPoint, ScalarMultiPolygon, ScalarPolygon,
    ValidCoords, ValidList,
};

/// A view that may hold lists and coordinates outside its rows (see "Trimming" in docs/View Types.md).
/// A kernel reaches the levels below the rows only through the offsets of the rows.
/// It reads the view part by part, so it cannot pass it on as a plain one.
pub struct Untrimmed<T>(pub(crate) T);

/// A trimmed view is an untrimmed one too.
impl<T> From<T> for Untrimmed<T> {
    fn from(view: T) -> Self {
        Self(view)
    }
}

impl Untrimmed<LineStringArray> {
    #[inline(always)]
    pub fn rows(&self) -> &List {
        &self.0.rows
    }

    #[inline(always)]
    pub fn coords(&self) -> &ValidCoords {
        &self.0.coords
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> ScalarLineString<'_> {
        self.0.get(i)
    }

    /// The view, with every level sliced to what its rows use.
    /// Free when it already is.
    pub fn trimmed(self) -> LineStringArray {
        let array = self.0;
        if array.is_trimmed() {
            return array;
        }
        let (rows, coords) = array.rows.sliced(0..array.rows.len());
        LineStringArray {
            rows,
            coords: array.coords.sliced(coords),
        }
    }
}

impl Untrimmed<MultiPointArray> {
    #[inline(always)]
    pub fn rows(&self) -> &List {
        &self.0.rows
    }

    #[inline(always)]
    pub fn coords(&self) -> &ValidCoords {
        &self.0.coords
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> ScalarMultiPoint<'_> {
        self.0.get(i)
    }

    /// See [`Untrimmed::<LineStringArray>::trimmed`].
    pub fn trimmed(self) -> MultiPointArray {
        let array = self.0;
        if array.is_trimmed() {
            return array;
        }
        let (rows, coords) = array.rows.sliced(0..array.rows.len());
        MultiPointArray {
            rows,
            coords: array.coords.sliced(coords),
        }
    }
}

impl Untrimmed<PolygonArray> {
    #[inline(always)]
    pub fn rows(&self) -> &List {
        &self.0.rows
    }

    #[inline(always)]
    pub fn rings(&self) -> &ValidList {
        &self.0.rings
    }

    #[inline(always)]
    pub fn coords(&self) -> &ValidCoords {
        &self.0.coords
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> ScalarPolygon<'_> {
        self.0.get(i)
    }

    /// See [`Untrimmed::<LineStringArray>::trimmed`].
    pub fn trimmed(self) -> PolygonArray {
        let array = self.0;
        if array.is_trimmed() {
            return array;
        }
        let (rows, rings) = array.rows.sliced(0..array.rows.len());
        let (rings, coords) = array.rings.sliced(rings);
        PolygonArray {
            rows,
            rings,
            coords: array.coords.sliced(coords),
        }
    }
}

impl Untrimmed<MultiLineStringArray> {
    #[inline(always)]
    pub fn rows(&self) -> &List {
        &self.0.rows
    }

    #[inline(always)]
    pub fn lines(&self) -> &ValidList {
        &self.0.lines
    }

    #[inline(always)]
    pub fn coords(&self) -> &ValidCoords {
        &self.0.coords
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> ScalarMultiLineString<'_> {
        self.0.get(i)
    }

    /// See [`Untrimmed::<LineStringArray>::trimmed`].
    pub fn trimmed(self) -> MultiLineStringArray {
        let array = self.0;
        if array.is_trimmed() {
            return array;
        }
        let (rows, lines) = array.rows.sliced(0..array.rows.len());
        let (lines, coords) = array.lines.sliced(lines);
        MultiLineStringArray {
            rows,
            lines,
            coords: array.coords.sliced(coords),
        }
    }
}

impl Untrimmed<MultiPolygonArray> {
    #[inline(always)]
    pub fn rows(&self) -> &List {
        &self.0.rows
    }

    #[inline(always)]
    pub fn polygons(&self) -> &ValidList {
        &self.0.polygons
    }

    #[inline(always)]
    pub fn rings(&self) -> &ValidList {
        &self.0.rings
    }

    #[inline(always)]
    pub fn coords(&self) -> &ValidCoords {
        &self.0.coords
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> ScalarMultiPolygon<'_> {
        self.0.get(i)
    }

    /// See [`Untrimmed::<LineStringArray>::trimmed`].
    pub fn trimmed(self) -> MultiPolygonArray {
        let array = self.0;
        if array.is_trimmed() {
            return array;
        }
        let (rows, polygons) = array.rows.sliced(0..array.rows.len());
        let (polygons, rings) = array.polygons.sliced(polygons);
        let (rings, coords) = array.rings.sliced(rings);
        MultiPolygonArray {
            rows,
            polygons,
            rings,
            coords: array.coords.sliced(coords),
        }
    }
}
