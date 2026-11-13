//! Sorting the parts of a geometry.
//! Implements the Python `sort_parts`, which documents the order.
//!
//! Each row first gets an order map: the index of every part, in sorted order.
//! [PartSorter] builds it one row at a time,
//! so other kernels can walk the parts in sorted order without moving them.
//! The kernels here then copy the parts into that order.
//! The rows keep their offsets.
//!
//! Rings can also be rotated to start at their smallest rotation, before the parts are compared.
//! An order map then holds a [RingStart] per ring: the ring, and the vertex it starts at.
//! Rotated rings have to be closed or empty (see [SortOptions::rotate_rings]).
//! They are checked with [is_closed_or_empty] rather than [is_ring](geopolars_views::is_ring),
//! as rotating needs no minimum number of vertices.
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
//!
//! ## `ROTATE_RINGS`
//!
//! The multilinestring, polygon, and multipolygon kernels are generic over `ROTATE_RINGS`.
//! Linestrings are only ever rotated, and multipoints never.
//! If `ROTATE_RINGS=true`, every ring is rotated first, and the parts are compared as they are then.

use std::cmp::Ordering;
use std::ops::Range;

use super::is_valid::is_closed_or_empty;
use super::rings::offsets_of;
use crate::{GeoComputeError, GeoComputeResult};
use geopolars_views::{
    LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PolygonArray,
    ScalarLineString, ScalarMultiLineString, ScalarMultiPoint, ScalarMultiPolygon, ScalarPolygon,
    ValidCoords, ValidList,
};

/// Where to look next in parts that are equal up to here.
/// For a part that is a single ring (a linestring or a hole), `ring` stays `0`.
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

/// Where vertex `v` of the ring at `vertices` sits in the coordinates, once it begins at `start`.
/// A ring with `start = 0` is left as it is, so it keeps its own closing vertex.
/// A rotated ring is closed again by its new first vertex.
#[inline(always)]
fn rotated_vertex(vertices: &Range<usize>, start: usize, v: usize) -> usize {
    if start == 0 {
        return vertices.start + v;
    }
    let distinct = vertices.len() - 1;
    let k = if v == distinct {
        start
    } else if start + v >= distinct {
        start + v - distinct
    } else {
        start + v
    };
    vertices.start + k
}

/// The token of the ring at `vertices` at `cursor`, once it begins at `start`.
/// Past its last vertex, the ring has `past_end`.
#[inline(always)]
fn ring_token(vertices: Range<usize>, start: usize, cursor: Cursor, past_end: Token) -> Token {
    if cursor.vertex < vertices.len() {
        Token::Vertex(rotated_vertex(&vertices, start, cursor.vertex))
    } else {
        past_end
    }
}

/// A ring of an order map, rotated to begin at its vertex `start`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingStart {
    /// The index of the ring in its level of the storage.
    pub ring: usize,
    /// The vertex the ring begins at, counted from its first vertex (not an index into the coordinates).
    /// A ring that is not rotated has `start = 0`, as does an empty one.
    pub start: usize,
}

/// What an order map holds per ring.
/// A plain index is a ring that is never rotated.
trait RingEntry: Copy {
    fn ring(self) -> usize;
    fn start(self) -> usize;
    fn unrotated(ring: usize) -> Self;

    /// The ranges of the coordinates of this ring of `lists` to copy, in order.
    fn pieces(self, lists: &ValidList) -> impl Iterator<Item = Range<usize>> + Clone;

    fn push_line_string_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        sorter: &mut PartSorter,
        multi: ScalarMultiLineString,
        order: &mut Vec<Self>,
    );

    fn push_hole_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        sorter: &mut PartSorter,
        polygon: ScalarPolygon,
        order: &mut Vec<Self>,
    );

    fn push_polygon_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        sorter: &mut PartSorter,
        multi: ScalarMultiPolygon,
        ring_order: &[Self],
        order: &mut Vec<usize>,
    );
}

impl RingEntry for usize {
    #[inline(always)]
    fn ring(self) -> usize {
        self
    }

    #[inline(always)]
    fn start(self) -> usize {
        0
    }

    #[inline(always)]
    fn unrotated(ring: usize) -> Self {
        ring
    }

    #[inline(always)]
    fn pieces(self, lists: &ValidList) -> impl Iterator<Item = Range<usize>> + Clone {
        std::iter::once(lists.range(self))
    }

    fn push_line_string_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        sorter: &mut PartSorter,
        multi: ScalarMultiLineString,
        order: &mut Vec<Self>,
    ) {
        sorter.push_line_string_order::<Z, M, DESCENDING>(multi, order);
    }

    fn push_hole_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        sorter: &mut PartSorter,
        polygon: ScalarPolygon,
        order: &mut Vec<Self>,
    ) {
        sorter.push_hole_order::<Z, M, DESCENDING>(polygon, order);
    }

    fn push_polygon_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        sorter: &mut PartSorter,
        multi: ScalarMultiPolygon,
        ring_order: &[Self],
        order: &mut Vec<usize>,
    ) {
        sorter.push_polygon_order::<Z, M, DESCENDING>(multi, ring_order, order);
    }
}

impl RingEntry for RingStart {
    #[inline(always)]
    fn ring(self) -> usize {
        self.ring
    }

    #[inline(always)]
    fn start(self) -> usize {
        self.start
    }

    #[inline(always)]
    fn unrotated(ring: usize) -> Self {
        RingStart { ring, start: 0 }
    }

    /// From the start to the old closing vertex (left out), then from the first vertex to the start,
    /// then the start again to close it.
    #[inline(always)]
    fn pieces(self, lists: &ValidList) -> impl Iterator<Item = Range<usize>> + Clone {
        let vertices = lists.range(self.ring);
        let first = vertices.start + self.start;
        let pieces = if self.start == 0 {
            [vertices, 0..0, 0..0]
        } else {
            [
                first..vertices.end - 1,
                vertices.start..first,
                first..first + 1,
            ]
        };
        pieces.into_iter().filter(|piece| !piece.is_empty())
    }

    fn push_line_string_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        sorter: &mut PartSorter,
        multi: ScalarMultiLineString,
        order: &mut Vec<Self>,
    ) {
        sorter.push_rotated_line_string_order::<Z, M, DESCENDING>(multi, order);
    }

    fn push_hole_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        sorter: &mut PartSorter,
        polygon: ScalarPolygon,
        order: &mut Vec<Self>,
    ) {
        sorter.push_rotated_hole_order::<Z, M, DESCENDING>(polygon, order);
    }

    fn push_polygon_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        sorter: &mut PartSorter,
        multi: ScalarMultiPolygon,
        ring_order: &[Self],
        order: &mut Vec<usize>,
    ) {
        sorter.push_rotated_polygon_order::<Z, M, DESCENDING>(multi, ring_order, order);
    }
}

/// Picks the [RingEntry] of `ROTATE_RINGS`.
/// This is a [RingStart] if `ROTATE_RINGS=true`, otherwise a plain index.
struct Rings<const ROTATE_RINGS: bool>;

trait RingEntryOf {
    type Entry: RingEntry;
}

impl RingEntryOf for Rings<false> {
    type Entry = usize;
}

impl RingEntryOf for Rings<true> {
    type Entry = RingStart;
}

type Entry<const ROTATE_RINGS: bool> = <Rings<ROTATE_RINGS> as RingEntryOf>::Entry;

/// Scratch space for [Runs::sort], kept between rows.
#[derive(Default)]
struct Runs {
    /// Runs of parts that are equal up to the cursor, as a range of the parts being sorted.
    open: Vec<(Range<usize>, Cursor)>,
    /// The token at the cursor of every part of the run being sorted, with that part.
    keyed: Vec<(Token, usize)>,
}

impl Runs {
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
}

/// Builds order maps, one row at a time.
/// Each `push_*_order` pushes the index of every part of one geometry onto `order`, in sorted order.
/// An index is that of the part in its level of the storage
/// (the coordinates of a multipoint, the rings of a polygon, ...).
///
/// Reuse one sorter for every row. It keeps its scratch space between them.
#[derive(Default)]
pub struct PartSorter {
    runs: Runs,
    /// The candidates of [ring_start](Self::ring_start) still tied for the smallest rotation.
    candidates: Vec<usize>,
    /// The start of every ring of the geometry being sorted, indexed from its first ring.
    starts: Vec<usize>,
    /// The rings of the geometry being sorted, for a `push_rotated_*_order`.
    rings: Vec<usize>,
}

impl PartSorter {
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
        let lines = multi.lines;
        self.runs
            .sort::<Z, M, DESCENDING>(&mut order[first..], multi.coords, |line, cursor| {
                ring_token(lines.range(line), 0, cursor, Token::End)
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
        let rings = polygon.rings;
        if let Some(holes) = order.get_mut(holes_start..) {
            self.runs
                .sort::<Z, M, DESCENDING>(holes, polygon.coords, |ring, cursor| {
                    ring_token(rings.range(ring), 0, cursor, Token::End)
                });
        }
    }

    /// Polygons are compared ring by ring, with their holes in the order of `ring_order`.
    /// `ring_order` holds the rings of every polygon of `multi`, indexed from its first ring,
    /// as [push_hole_order](Self::push_hole_order) pushes them.
    pub fn push_polygon_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        &mut self,
        multi: ScalarMultiPolygon,
        ring_order: &[usize],
        order: &mut Vec<usize>,
    ) {
        self.push_polygon_order_of::<Z, M, DESCENDING, usize>(multi, ring_order, order);
    }

    /// The `start` of a [RingStart] for `ring`, once rotated to its smallest rotation.
    /// Rotations are always compared ascending, and without the closing vertex.
    /// Of rotations that are equal, the earliest is taken.
    /// `ring` has to be closed or empty (see [SortOptions::rotate_rings]).
    pub fn ring_start<const Z: bool, const M: bool>(&mut self, ring: ScalarLineString) -> usize {
        let (vertices, coords) = (ring.vertices, ring.coords);
        // The closing vertex repeats the first, so it is not a place to start.
        let distinct = vertices.len().saturating_sub(1);
        if distinct < 2 {
            return 0;
        }
        let at = |k: usize, depth: usize| vertices.start + (k + depth) % distinct;
        let candidates = &mut self.candidates;
        candidates.clear();
        candidates.push(0);
        // Nothing wraps round yet, so this needs no `at`.
        for k in 1..distinct {
            let smallest = vertices.start + candidates[0];
            match cmp_coordinates::<Z, M>(coords, vertices.start + k, smallest) {
                Ordering::Less => {
                    candidates.clear();
                    candidates.push(k);
                }
                Ordering::Equal => candidates.push(k),
                Ordering::Greater => {}
            }
        }
        // PERF: a ring that repeats a pattern keeps its candidates tied up to its full length,
        // which is quadratic in that length.
        // Booth's or Zhou's minimal rotation is linear, but reads every vertex.
        let mut depth = 1;
        while candidates.len() > 1 && depth < distinct {
            let mut smallest = candidates[0];
            let mut kept = 1;
            for c in 1..candidates.len() {
                let k = candidates[c];
                match cmp_coordinates::<Z, M>(coords, at(k, depth), at(smallest, depth)) {
                    Ordering::Less => {
                        smallest = k;
                        candidates[0] = k;
                        kept = 1;
                    }
                    Ordering::Equal => {
                        candidates[kept] = k;
                        kept += 1;
                    }
                    Ordering::Greater => {}
                }
            }
            candidates.truncate(kept);
            depth += 1;
        }
        candidates[0]
    }

    /// [push_line_string_order](Self::push_line_string_order), with every linestring rotated.
    /// Each linestring of `multi` has to be closed or empty (see [SortOptions::rotate_rings]).
    pub fn push_rotated_line_string_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        &mut self,
        multi: ScalarMultiLineString,
        order: &mut Vec<RingStart>,
    ) {
        let first = multi.parts.start;
        self.starts.clear();
        for line in multi.line_strings() {
            let start = self.ring_start::<Z, M>(line);
            self.starts.push(start);
        }
        self.rings.clear();
        self.rings.extend(multi.parts.clone());
        let (lines, starts) = (multi.lines, &self.starts);
        self.runs
            .sort::<Z, M, DESCENDING>(&mut self.rings, multi.coords, |line, cursor| {
                ring_token(lines.range(line), starts[line - first], cursor, Token::End)
            });
        order.extend(self.rings.iter().map(|&ring| RingStart {
            ring,
            start: self.starts[ring - first],
        }));
    }

    /// [push_hole_order](Self::push_hole_order), with every ring rotated, the exterior too.
    /// Each ring of `polygon` has to be closed or empty (see [SortOptions::rotate_rings]).
    pub fn push_rotated_hole_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        &mut self,
        polygon: ScalarPolygon,
        order: &mut Vec<RingStart>,
    ) {
        let first = polygon.parts.start;
        self.starts.clear();
        for ring in polygon.rings() {
            let start = self.ring_start::<Z, M>(ring);
            self.starts.push(start);
        }
        self.rings.clear();
        self.rings.extend(polygon.parts.clone());
        let (rings, starts) = (polygon.rings, &self.starts);
        if let Some(holes) = self.rings.get_mut(1..) {
            self.runs
                .sort::<Z, M, DESCENDING>(holes, polygon.coords, |ring, cursor| {
                    ring_token(rings.range(ring), starts[ring - first], cursor, Token::End)
                });
        }
        order.extend(self.rings.iter().map(|&ring| RingStart {
            ring,
            start: self.starts[ring - first],
        }));
    }

    /// [push_polygon_order](Self::push_polygon_order), with the rings rotated as `ring_order`.
    /// `ring_order` is pushed by [push_rotated_hole_order](Self::push_rotated_hole_order).
    pub fn push_rotated_polygon_order<const Z: bool, const M: bool, const DESCENDING: bool>(
        &mut self,
        multi: ScalarMultiPolygon,
        ring_order: &[RingStart],
        order: &mut Vec<usize>,
    ) {
        self.push_polygon_order_of::<Z, M, DESCENDING, RingStart>(multi, ring_order, order);
    }

    fn push_polygon_order_of<const Z: bool, const M: bool, const DESCENDING: bool, E: RingEntry>(
        &mut self,
        multi: ScalarMultiPolygon,
        ring_order: &[E],
        order: &mut Vec<usize>,
    ) {
        let (polygons, rings) = (multi.polygons, multi.rings);
        let first_ring = polygons.span(multi.parts.clone()).start;
        let first = order.len();
        order.extend(multi.parts);
        self.runs
            .sort::<Z, M, DESCENDING>(&mut order[first..], multi.coords, |polygon, cursor| {
                let polygon_rings = polygons.range(polygon);
                if cursor.ring >= polygon_rings.len() {
                    return Token::End;
                }
                let entry = ring_order[polygon_rings.start + cursor.ring - first_ring];
                ring_token(
                    rings.range(entry.ring()),
                    entry.start(),
                    cursor,
                    Token::EndRing,
                )
            });
    }
}

/// The options of the Python `sort_parts`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SortOptions {
    pub descending: bool,
    /// `rings="rotate"` in Python.
    pub rotate_rings: bool,
    pub ignore_per_row_errors: bool,
}

/// The order map of every row, with `push_row` pushing that of a row that is not null.
/// A null row's parts keep their place.
fn order_per_row<E: RingEntry>(
    rows: &List,
    parts: usize,
    mut push_row: impl FnMut(usize, &mut Vec<E>),
) -> Vec<E> {
    // PERF: this could be a pre-sized array, with every row sorting its own slice of it in place.
    let mut order = Vec::with_capacity(parts);
    for i in 0..rows.len() {
        if rows.is_valid(i) {
            push_row(i, &mut order);
        } else {
            order.extend(rows.range(i).map(E::unrotated));
        }
    }
    order
}

fn is_unchanged<E: RingEntry>(order: &[E]) -> bool {
    order
        .iter()
        .enumerate()
        .all(|(k, entry)| entry.ring() == k && entry.start() == 0)
}

/// `lists` with its lists in the order `order`, and the coordinates they hold moved with them.
fn lists_in_order<E: RingEntry>(
    lists: &ValidList,
    coords: &ValidCoords,
    order: impl Iterator<Item = E> + Clone,
) -> (ValidList, ValidCoords) {
    (
        ValidList::from_offsets(offsets_of(
            order.clone().map(|entry| lists.range(entry.ring()).len()),
        )),
        coords.copy_ranges(order.flat_map(|entry| entry.pieces(lists))),
    )
}

/// `rows`, with a row null where one of its rings (from `rings_of`) fails [is_closed_or_empty].
/// Unless `ignore_per_row_errors=true`, the first such ring is returned as an error instead.
fn rows_null_where_not_closed<const Z: bool, L: Iterator<Item = Range<usize>>>(
    rows: &List,
    coords: &ValidCoords,
    rings_of: impl Fn(usize) -> L,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<List> {
    let mut failed = Vec::new();
    for i in (0..rows.len()).filter(|&i| rows.is_valid(i)) {
        let not_closed = rings_of(i).find(|ring| !is_closed_or_empty::<Z>(coords, ring.clone()));
        if let Some(ring) = not_closed {
            if !ignore_per_row_errors {
                return Err(GeoComputeError::NotClosed {
                    vertices: ring.len(),
                });
            }
            failed.push(i);
        }
    }
    Ok(rows.clone().with_null_at(&failed))
}

fn multi_points_sorted_with<const Z: bool, const M: bool, const DESCENDING: bool>(
    array: &MultiPointArray,
) -> MultiPointArray {
    let mut sorter = PartSorter::default();
    let order = order_per_row(&array.rows, array.coords.len(), |i, order| {
        sorter.push_point_order::<Z, M, DESCENDING>(array.get(i), order)
    });
    let coords = if is_unchanged(&order) {
        array.coords.clone()
    } else {
        array.coords.gathered(&order)
    };
    MultiPointArray::from_parts(array.rows.clone(), coords)
}

fn line_strings_with_rings_rotated_with<const Z: bool, const M: bool>(
    array: &LineStringArray,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<LineStringArray> {
    let rings_of = |i| std::iter::once(array.rows.range(i));
    let rows = rows_null_where_not_closed::<Z, _>(
        &array.rows,
        &array.coords,
        rings_of,
        ignore_per_row_errors,
    )?;
    let mut sorter = PartSorter::default();
    let order: Vec<RingStart> = (0..rows.len())
        .map(|i| RingStart {
            ring: i,
            start: if rows.is_valid(i) {
                sorter.ring_start::<Z, M>(array.get(i))
            } else {
                0
            },
        })
        .collect();
    let coords = if is_unchanged(&order) {
        array.coords.clone()
    } else {
        let lines = rows.ignoring_validity();
        array
            .coords
            .copy_ranges(order.iter().flat_map(|entry| entry.pieces(lines)))
    };
    Ok(LineStringArray::from_parts(rows, coords))
}

fn multi_line_strings_sorted_with<
    const Z: bool,
    const M: bool,
    const DESCENDING: bool,
    const ROTATE_RINGS: bool,
>(
    array: &MultiLineStringArray,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<MultiLineStringArray>
where
    Rings<ROTATE_RINGS>: RingEntryOf,
{
    let lines = &array.lines;
    let rows = if ROTATE_RINGS {
        let rings_of = |i| array.rows.range(i).map(|j| lines.range(j));
        rows_null_where_not_closed::<Z, _>(
            &array.rows,
            &array.coords,
            rings_of,
            ignore_per_row_errors,
        )?
    } else {
        array.rows.clone()
    };
    let mut sorter = PartSorter::default();
    let order = order_per_row(&rows, lines.len(), |i, order| {
        <Entry<ROTATE_RINGS>>::push_line_string_order::<Z, M, DESCENDING>(
            &mut sorter,
            array.get(i),
            order,
        )
    });
    let (lines, coords) = if is_unchanged(&order) {
        (lines.clone(), array.coords.clone())
    } else {
        lists_in_order(lines, &array.coords, order.iter().copied())
    };
    Ok(MultiLineStringArray::from_parts(rows, lines, coords))
}

fn polygons_with_holes_sorted_with<
    const Z: bool,
    const M: bool,
    const DESCENDING: bool,
    const ROTATE_RINGS: bool,
>(
    array: &PolygonArray,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<PolygonArray>
where
    Rings<ROTATE_RINGS>: RingEntryOf,
{
    let rows = if ROTATE_RINGS {
        let rings_of = |i| array.rows.range(i).map(|j| array.rings.range(j));
        rows_null_where_not_closed::<Z, _>(
            &array.rows,
            &array.coords,
            rings_of,
            ignore_per_row_errors,
        )?
    } else {
        array.rows.clone()
    };
    let mut sorter = PartSorter::default();
    let order = order_per_row(&rows, array.rings.len(), |i, order| {
        <Entry<ROTATE_RINGS>>::push_hole_order::<Z, M, DESCENDING>(&mut sorter, array.get(i), order)
    });
    let (rings, coords) = if is_unchanged(&order) {
        (array.rings.clone(), array.coords.clone())
    } else {
        lists_in_order(&array.rings, &array.coords, order.iter().copied())
    };
    Ok(PolygonArray::from_parts(rows, rings, coords))
}

fn multi_polygons_sorted_with<
    const Z: bool,
    const M: bool,
    const DESCENDING: bool,
    const ROTATE_RINGS: bool,
>(
    array: &MultiPolygonArray,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<MultiPolygonArray>
where
    Rings<ROTATE_RINGS>: RingEntryOf,
{
    let (polygons, rings, coords) = (&array.polygons, &array.rings, &array.coords);
    let rows = if ROTATE_RINGS {
        let rings_of = |i| polygons.span(array.rows.range(i)).map(|j| rings.range(j));
        rows_null_where_not_closed::<Z, _>(&array.rows, coords, rings_of, ignore_per_row_errors)?
    } else {
        array.rows.clone()
    };
    let mut sorter = PartSorter::default();
    // `ring_order` is indexed by ring, so it needs the rings of every polygon, those of null rows too.
    // Those are sorted (and rotated) without being checked.
    // That does no harm, as the row is null.
    let mut ring_order: Vec<Entry<ROTATE_RINGS>> = Vec::with_capacity(rings.len());
    for polygon in polygons.ranges() {
        let polygon = ScalarPolygon {
            rings,
            coords,
            parts: polygon,
        };
        <Entry<ROTATE_RINGS>>::push_hole_order::<Z, M, DESCENDING>(
            &mut sorter,
            polygon,
            &mut ring_order,
        );
    }
    let polygon_order = order_per_row(&rows, polygons.len(), |i, order| {
        let multi = array.get(i);
        let first_ring = polygons.span(multi.parts.clone()).start;
        <Entry<ROTATE_RINGS>>::push_polygon_order::<Z, M, DESCENDING>(
            &mut sorter,
            multi,
            &ring_order[first_ring..],
            order,
        )
    });
    if is_unchanged(&polygon_order) && is_unchanged(&ring_order) {
        return Ok(MultiPolygonArray::from_parts(
            rows,
            polygons.clone(),
            rings.clone(),
            coords.clone(),
        ));
    }
    let sorted_rings = polygon_order
        .iter()
        .flat_map(|&j| ring_order[polygons.range(j)].iter().copied());
    let (rings, coords) = lists_in_order(rings, coords, sorted_rings);
    Ok(MultiPolygonArray::from_parts(
        rows,
        ValidList::from_offsets(offsets_of(
            polygon_order.iter().map(|&j| polygons.range(j).len()),
        )),
        rings,
        coords,
    ))
}

pub fn multi_points_sorted(array: &MultiPointArray, descending: bool) -> MultiPointArray {
    let (z, m) = (array.coords.z().is_some(), array.coords.m().is_some());
    with_const_flags!(
        Z = z, M = m, DESCENDING = descending;
        multi_points_sorted_with::<Z, M, DESCENDING>(array)
    )
}

/// Every linestring rotated as a ring, as [SortOptions::rotate_rings] does for the parts of [multi_line_strings_sorted].
/// Every linestring has to be closed or empty, as for [SortOptions::rotate_rings].
pub fn line_strings_with_rings_rotated(
    array: &LineStringArray,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<LineStringArray> {
    let (z, m) = (array.coords.z().is_some(), array.coords.m().is_some());
    with_const_flags!(
        Z = z, M = m;
        line_strings_with_rings_rotated_with::<Z, M>(array, ignore_per_row_errors)
    )
}

pub fn multi_line_strings_sorted(
    array: &MultiLineStringArray,
    options: SortOptions,
) -> GeoComputeResult<MultiLineStringArray> {
    let (z, m) = (array.coords.z().is_some(), array.coords.m().is_some());
    with_const_flags!(
        Z = z, M = m, DESCENDING = options.descending, ROTATE_RINGS = options.rotate_rings;
        multi_line_strings_sorted_with::<Z, M, DESCENDING, ROTATE_RINGS>(
            array,
            options.ignore_per_row_errors,
        )
    )
}

/// The exterior of every polygon stays first.
pub fn polygons_with_holes_sorted(
    array: &PolygonArray,
    options: SortOptions,
) -> GeoComputeResult<PolygonArray> {
    let (z, m) = (array.coords.z().is_some(), array.coords.m().is_some());
    with_const_flags!(
        Z = z, M = m, DESCENDING = options.descending, ROTATE_RINGS = options.rotate_rings;
        polygons_with_holes_sorted_with::<Z, M, DESCENDING, ROTATE_RINGS>(
            array,
            options.ignore_per_row_errors,
        )
    )
}

/// The holes of every polygon are sorted first, as in [polygons_with_holes_sorted].
/// The polygons are then compared with their holes in that order.
pub fn multi_polygons_sorted(
    array: &MultiPolygonArray,
    options: SortOptions,
) -> GeoComputeResult<MultiPolygonArray> {
    let (z, m) = (array.coords.z().is_some(), array.coords.m().is_some());
    with_const_flags!(
        Z = z, M = m, DESCENDING = options.descending, ROTATE_RINGS = options.rotate_rings;
        multi_polygons_sorted_with::<Z, M, DESCENDING, ROTATE_RINGS>(
            array,
            options.ignore_per_row_errors,
        )
    )
}
