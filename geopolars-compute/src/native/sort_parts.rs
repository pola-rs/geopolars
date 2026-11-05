//! Sorting the parts of a geometry.
//! See `sort_parts` in Python for the rules.
//!
//! Each row first gets an order map: the index of every part, in sorted order.
//! [PartSorter] builds it one row at a time,
//! so other kernels can walk the parts in sorted order without moving them.
//! The kernels here then copy the parts into that order.
//! The rows keep their offsets.
//!
//! # Const generic flags
//!
//! ## `Z`
//!
//! If `Z=true`, vertices that are equal in x and y are compared on z.
//!
//! ## `M`
//!
//! If `M=true`, vertices that are equal in every other axis are compared on m.
//!
//! ## `DESCENDING`
//!
//! If `DESCENDING=true`, the parts are sorted from greatest to smallest.
//! Equal parts keep their order either way.

use std::cmp::Ordering;

use geopolars_core::{
    List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PolygonArray,
    ScalarMultiLineString, ScalarMultiPoint, ScalarMultiPolygon, ScalarPolygon, ValidCoords,
    ValidList,
};

use super::rings::offsets_of;

/// Where to look next in parts that are equal up to here.
/// For a linestring or a point, `ring` stays `0`.
#[derive(Clone, Copy, Default)]
struct Cursor {
    ring: usize,
    vertex: usize,
}

/// What a part has at a [Cursor].
/// `End` sorts before `EndRing`, which sorts before any `Vertex`,
/// so a part that runs out first is the smaller one.
#[derive(Clone, Copy)]
enum Token {
    End,
    EndRing,
    /// The index of the vertex in the coordinates.
    Vertex(usize),
}

impl Token {
    /// Where parts that are equal up to and including this token continue,
    /// or `None` if they have ended.
    #[inline(always)]
    fn advanced(self, cursor: Cursor) -> Option<Cursor> {
        match self {
            Token::End => None,
            Token::EndRing => Some(Cursor {
                ring: cursor.ring + 1,
                vertex: 0,
            }),
            Token::Vertex(_) => Some(Cursor {
                vertex: cursor.vertex + 1,
                ..cursor
            }),
        }
    }

    #[inline(always)]
    fn rank(self) -> u8 {
        match self {
            Token::End => 0,
            Token::EndRing => 1,
            Token::Vertex(_) => 2,
        }
    }
}

/// NaN is greater than any other value, and equal to itself.
/// `-0.0` and `0.0` are equal.
#[inline(always)]
fn cmp_values(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b)
        .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
}

#[inline(always)]
fn cmp_coordinates<const Z: bool, const M: bool>(
    coords: &ValidCoords,
    i: usize,
    j: usize,
) -> Ordering {
    let cmp_axis = |axis: &[f64]| cmp_values(axis[i], axis[j]);
    cmp_axis(coords.x())
        .then_with(|| cmp_axis(coords.y()))
        .then_with(|| {
            if Z {
                cmp_axis(coords.z().expect("a z, with `Z`"))
            } else {
                Ordering::Equal
            }
        })
        .then_with(|| {
            if M {
                cmp_axis(coords.m().expect("an m, with `M`"))
            } else {
                Ordering::Equal
            }
        })
}

#[inline(always)]
fn cmp_tokens<const Z: bool, const M: bool>(coords: &ValidCoords, a: Token, b: Token) -> Ordering {
    match (a, b) {
        (Token::Vertex(i), Token::Vertex(j)) => cmp_coordinates::<Z, M>(coords, i, j),
        _ => a.rank().cmp(&b.rank()),
    }
}

#[inline(always)]
fn in_direction<const DESCENDING: bool>(ordering: Ordering) -> Ordering {
    if DESCENDING {
        ordering.reverse()
    } else {
        ordering
    }
}

#[inline(always)]
fn line_token(lines: &ValidList, line: usize, cursor: Cursor) -> Token {
    let vertices = lines.range(line);
    if cursor.vertex < vertices.len() {
        Token::Vertex(vertices.start + cursor.vertex)
    } else {
        Token::End
    }
}

/// Builds the order maps of [the module](self), one row at a time.
/// Each `push_*_order` pushes the index of every part of one geometry onto `order`, in sorted order.
/// An index is that of the part in its level of the storage
/// (the coordinates of a multipoint, the rings of a polygon, ...).
///
/// Reuse one sorter for every row: it keeps its scratch space between them.
#[derive(Default)]
pub struct PartSorter {
    /// Runs of parts that are equal up to the cursor, as a range of the parts being sorted.
    open: Vec<(std::ops::Range<usize>, Cursor)>,
    /// The token at the cursor of every part of the run being sorted, with that part.
    keyed: Vec<(Token, usize)>,
}

impl PartSorter {
    /// Sorts `parts` by the tokens `token_at` gives them, one cursor at a time.
    /// Only the parts that are still equal up to a cursor are looked at past it.
    fn sort<const Z: bool, const M: bool, const DESCENDING: bool>(
        &mut self,
        parts: &mut [usize],
        coords: &ValidCoords,
        token_at: impl Fn(usize, Cursor) -> Token,
    ) {
        let Self { open, keyed } = self;
        open.clear();
        if parts.len() > 1 {
            open.push((0..parts.len(), Cursor::default()));
        }
        while let Some((run, cursor)) = open.pop() {
            keyed.clear();
            keyed.extend(
                parts[run.clone()]
                    .iter()
                    .map(|&part| (token_at(part, cursor), part)),
            );
            // Breaking ties on the index keeps equal parts in their order,
            // without the scratch space of a stable sort.
            keyed.sort_unstable_by(|&(a, i), &(b, j)| {
                in_direction::<DESCENDING>(cmp_tokens::<Z, M>(coords, a, b)).then(i.cmp(&j))
            });
            let mut tie_start = 0;
            for k in 1..=keyed.len() {
                parts[run.start + k - 1] = keyed[k - 1].1;
                let tie_ends = k == keyed.len()
                    || cmp_tokens::<Z, M>(coords, keyed[tie_start].0, keyed[k].0).is_ne();
                if tie_ends {
                    let token = keyed[tie_start].0;
                    if let Some(next) = token.advanced(cursor).filter(|_| k - tie_start > 1) {
                        open.push((run.start + tie_start..run.start + k, next));
                    }
                    tie_start = k;
                }
            }
        }
    }

    pub fn push_point_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        &mut self,
        multi: ScalarMultiPoint,
        order: &mut Vec<usize>,
    ) {
        let first = order.len();
        order.extend(multi.vertices);
        // A point is a single vertex, so it needs no runs to be compared one vertex at a time.
        order[first..].sort_unstable_by(|&i, &j| {
            in_direction::<DESCENDING>(cmp_coordinates::<Z, M>(multi.coords, i, j)).then(i.cmp(&j))
        });
    }

    pub fn push_line_string_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        &mut self,
        multi: ScalarMultiLineString,
        order: &mut Vec<usize>,
    ) {
        let first = order.len();
        order.extend(multi.parts);
        self.sort::<Z, M, DESCENDING>(&mut order[first..], multi.coords, |line, cursor| {
            line_token(multi.lines, line, cursor)
        });
    }

    /// The exterior stays first.
    pub fn push_hole_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        &mut self,
        polygon: ScalarPolygon,
        order: &mut Vec<usize>,
    ) {
        let holes_start = order.len() + 1;
        order.extend(polygon.parts);
        if let Some(holes) = order.get_mut(holes_start..) {
            self.sort::<Z, M, DESCENDING>(holes, polygon.coords, |ring, cursor| {
                line_token(polygon.rings, ring, cursor)
            });
        }
    }

    /// Polygons are compared ring by ring, with their holes in the order of `ring_order`.
    /// `ring_order` holds the rings of every polygon of `multi`, from its first ring on,
    /// as [push_hole_order](Self::push_hole_order) pushes them.
    pub fn push_polygon_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        &mut self,
        multi: ScalarMultiPolygon,
        ring_order: &[usize],
        order: &mut Vec<usize>,
    ) {
        let (polygons, rings) = (multi.polygons, multi.rings);
        let first_ring = polygons.span(multi.parts.clone()).start;
        let first = order.len();
        order.extend(multi.parts);
        self.sort::<Z, M, DESCENDING>(&mut order[first..], multi.coords, |polygon, cursor| {
            let polygon_rings = polygons.range(polygon);
            if cursor.ring >= polygon_rings.len() {
                return Token::End;
            }
            let ring = ring_order[polygon_rings.start + cursor.ring - first_ring];
            let vertices = rings.range(ring);
            if cursor.vertex < vertices.len() {
                Token::Vertex(vertices.start + cursor.vertex)
            } else {
                Token::EndRing
            }
        });
    }
}

/// The order map of every row, with `push_row` pushing that of a row that is there.
/// A null row's parts keep their place.
fn order_per_row(
    rows: &List,
    parts: usize,
    mut push_row: impl FnMut(usize, &mut Vec<usize>),
) -> Vec<usize> {
    // PERF: this could be a pre-sized array, with every row sorting its own slice of it in place.
    let mut order = Vec::with_capacity(parts);
    for i in 0..rows.len() {
        if rows.is_valid(i) {
            push_row(i, &mut order);
        } else {
            order.extend(rows.range(i));
        }
    }
    order
}

fn is_identity(order: &[usize]) -> bool {
    order.iter().enumerate().all(|(k, &j)| k == j)
}

/// `lists` with its lists in the order `order`, and the coordinates they hold moved with them.
fn lists_in_order(
    lists: &ValidList,
    coords: &ValidCoords,
    order: impl Iterator<Item = usize> + Clone,
) -> (ValidList, ValidCoords) {
    (
        ValidList::from_offsets(offsets_of(order.clone().map(|j| lists.range(j).len()))),
        coords.copy_ranges(order.map(|j| lists.range(j))),
    )
}

fn multi_points_sorted_with<const Z: bool, const M: bool, const DESCENDING: bool>(
    array: &MultiPointArray,
) -> MultiPointArray {
    let mut sorter = PartSorter::default();
    let order = order_per_row(&array.rows, array.coords.len(), |i, order| {
        sorter.push_point_order::<Z, M, DESCENDING>(array.get(i), order)
    });
    let coords = if is_identity(&order) {
        array.coords.clone()
    } else {
        array.coords.gathered(&order)
    };
    MultiPointArray::from_parts(array.rows.clone(), coords)
}

fn multi_line_strings_sorted_with<const Z: bool, const M: bool, const DESCENDING: bool>(
    array: &MultiLineStringArray,
) -> MultiLineStringArray {
    let mut sorter = PartSorter::default();
    let order = order_per_row(&array.rows, array.lines.len(), |i, order| {
        sorter.push_line_string_order::<Z, M, DESCENDING>(array.get(i), order)
    });
    let (lines, coords) = if is_identity(&order) {
        (array.lines.clone(), array.coords.clone())
    } else {
        lists_in_order(&array.lines, &array.coords, order.iter().copied())
    };
    MultiLineStringArray::from_parts(array.rows.clone(), lines, coords)
}

fn polygons_with_holes_sorted_with<const Z: bool, const M: bool, const DESCENDING: bool>(
    array: &PolygonArray,
) -> PolygonArray {
    let mut sorter = PartSorter::default();
    let order = order_per_row(&array.rows, array.rings.len(), |i, order| {
        sorter.push_hole_order::<Z, M, DESCENDING>(array.get(i), order)
    });
    let (rings, coords) = if is_identity(&order) {
        (array.rings.clone(), array.coords.clone())
    } else {
        lists_in_order(&array.rings, &array.coords, order.iter().copied())
    };
    PolygonArray::from_parts(array.rows.clone(), rings, coords)
}

fn multi_polygons_sorted_with<const Z: bool, const M: bool, const DESCENDING: bool>(
    array: &MultiPolygonArray,
) -> MultiPolygonArray {
    let (polygons, rings, coords) = (&array.polygons, &array.rings, &array.coords);
    let mut sorter = PartSorter::default();
    // The polygons of a null row are sorted too, which does no harm, as they are masked.
    let mut ring_order = Vec::with_capacity(rings.len());
    for polygon in polygons.ranges() {
        let polygon = ScalarPolygon {
            rings,
            coords,
            parts: polygon,
        };
        sorter.push_hole_order::<Z, M, DESCENDING>(polygon, &mut ring_order);
    }
    let polygon_order = order_per_row(&array.rows, polygons.len(), |i, order| {
        let multi = array.get(i);
        let first_ring = polygons.span(multi.parts.clone()).start;
        sorter.push_polygon_order::<Z, M, DESCENDING>(multi, &ring_order[first_ring..], order)
    });
    if is_identity(&polygon_order) && is_identity(&ring_order) {
        return MultiPolygonArray::from_parts(
            array.rows.clone(),
            polygons.clone(),
            rings.clone(),
            coords.clone(),
        );
    }
    let sorted_rings = polygon_order
        .iter()
        .flat_map(|&j| ring_order[polygons.range(j)].iter().copied());
    let (rings, coords) = lists_in_order(rings, coords, sorted_rings);
    MultiPolygonArray::from_parts(
        array.rows.clone(),
        ValidList::from_offsets(offsets_of(
            polygon_order.iter().map(|&j| polygons.range(j).len()),
        )),
        rings,
        coords,
    )
}

pub fn multi_points_sorted(array: &MultiPointArray, descending: bool) -> MultiPointArray {
    let (z, m) = (array.coords.z().is_some(), array.coords.m().is_some());
    with_const_flags!(
        Z = z, M = m, DESCENDING = descending;
        multi_points_sorted_with::<Z, M, DESCENDING>(array)
    )
}

pub fn multi_line_strings_sorted(
    array: &MultiLineStringArray,
    descending: bool,
) -> MultiLineStringArray {
    let (z, m) = (array.coords.z().is_some(), array.coords.m().is_some());
    with_const_flags!(
        Z = z, M = m, DESCENDING = descending;
        multi_line_strings_sorted_with::<Z, M, DESCENDING>(array)
    )
}

/// The exterior of every polygon stays first.
pub fn polygons_with_holes_sorted(array: &PolygonArray, descending: bool) -> PolygonArray {
    let (z, m) = (array.coords.z().is_some(), array.coords.m().is_some());
    with_const_flags!(
        Z = z, M = m, DESCENDING = descending;
        polygons_with_holes_sorted_with::<Z, M, DESCENDING>(array)
    )
}

/// The holes of every polygon are sorted first, as in [polygons_with_holes_sorted].
/// The polygons are then compared with their holes in that order.
pub fn multi_polygons_sorted(array: &MultiPolygonArray, descending: bool) -> MultiPolygonArray {
    let (z, m) = (array.coords.z().is_some(), array.coords.m().is_some());
    with_const_flags!(
        Z = z, M = m, DESCENDING = descending;
        multi_polygons_sorted_with::<Z, M, DESCENDING>(array)
    )
}
