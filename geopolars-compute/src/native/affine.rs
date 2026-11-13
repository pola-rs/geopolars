//! Geometric transformations that preserve lines and parallelism,
//! but not necessarily Euclidean distances and angles.
//!
//! Translating, rotating, scaling and skewing are all one of these,
//! so they share one kernel, and differ only in the matrix they hand it.
//! Implements the Python `affine_transform`, which documents how a geometry is transformed.
//!
//! # Const generic flags
//!
//! ## `NULLS`
//!
//! [transform_points] is generic over `NULLS`.
//! If `NULLS=false`, every point is assumed to be there.

use std::ops::Range;

use geopolars_views::{
    Coords, LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray, ValidCoords,
};
use polars_arrow::compute::utils::combine_validities_and;

/// An affine transformation of the coordinates that are positions (`x`, `y` and `z`),
/// as the top three rows of its augmented matrix:
///
/// ```text
/// | x' |   | a  b  c  xoff |   | x |
/// | y' | = | d  e  f  yoff | · | y |
/// | z' |   | g  h  i  zoff |   | z |
///                              | 1 |
/// ```
///
/// Stored row by row in one flat array, `[a, b, c, xoff, d, e, f, yoff, g, h, i, zoff]`.
///
/// A geometry without a `z` reads it as 0 and has no `z'` to write,
/// so `c`, `f` and the whole bottom row do nothing to it.
pub struct AffineMatrix {
    // Only the coordinates the row actually uses are read:
    // a kernel this simple spends its time moving data, not multiplying it,
    // and most rows (of a translation, a scaling, a rotation in the plane) use only one or two.
    // A coordinate a row multiplies by 0 therefore has no say in it, not even when it is NaN.
    pub coefficients: [f64; 12],
}

impl AffineMatrix {
    /// The row that makes output coordinate `axis` (0 is `x`, 1 is `y`, 2 is `z`),
    /// with its offset last.
    fn row(&self, axis: usize) -> [f64; 4] {
        let (rows, _) = self.coefficients.as_chunks::<4>();
        rows[axis]
    }

    /// Whether the row that makes `axis` takes that coordinate as it is,
    /// before its offset is added.
    fn keeps(&self, axis: usize) -> bool {
        let mut unchanged = [0.0; 3];
        unchanged[axis] = 1.0;
        self.row(axis)[..3] == unchanged
    }
}

/// What each row adds to the matrix
pub enum RowShift<'a> {
    None,
    /// `M.(p − o) + o`: the matrix applied about each row's own origin `o`.
    ///
    /// That is `L.p + offset + (o − L.o)`, with `L` the matrix without its offsets:
    /// the plain matrix, with a shift that is the same for every position of a row.
    About(&'a PointArray),
    /// `M.p + v`: the matrix, then a move by each row's own vector `v`.
    /// A vector without a `z` leaves it where the matrix puts it.
    Then(&'a PointArray),
}

impl<'a> RowShift<'a> {
    /// The points the rows are shifted by.
    /// one per row.
    fn points(&self, rows: usize) -> Option<&'a PointArray> {
        let points = match *self {
            RowShift::None => return None,
            RowShift::About(points) | RowShift::Then(points) => points,
        };
        assert_eq!(points.coords.len(), rows, "one point per row");
        Some(points)
    }

    /// How output axis `axis` is shifted.
    /// by each row's point along `axis` - c*p
    /// `None` if it is not shifted.
    fn coefficients(&self, matrix: &AffineMatrix, axis: usize) -> Option<[f64; 3]> {
        match *self {
            RowShift::None => None,
            RowShift::Then(vectors) => {
                (axis < 2 || vectors.coords.dim().has_z()).then_some([0.0; 3])
            }
            // A row that keeps its coordinate moves it by its offset alone, about any origin.
            RowShift::About(_) => (!matrix.keeps(axis)).then(|| {
                let [a, b, c, _] = matrix.row(axis);
                [a, b, c]
            }),
        }
    }
}

pub fn affine_points(matrix: &AffineMatrix, array: &PointArray, shift: RowShift) -> PointArray {
    let coords = &array.coords;
    let rows = coords.len();
    let points = shift.points(rows);
    let positions = new_positions(matrix, coords.dim().has_z(), &shift, |axis, row| {
        let Some((coefficients, points)) = shift.coefficients(matrix, axis).zip(points) else {
            return axis_of_points(row, coords, rows, None::<fn(usize) -> f64>);
        };
        let values = points.coords.ignoring_validity();
        let (x, y) = (values.x(), values.y());
        match values.z() {
            None => axis_of_points(
                row,
                coords,
                rows,
                Some(point_shift(coefficients, axis, [x, y], rows)),
            ),
            Some(z) => axis_of_points(
                row,
                coords,
                rows,
                Some(point_shift(coefficients, axis, [x, y, z], rows)),
            ),
        }
    });
    let origin_validity = points.and_then(|points| points.coords.validity());
    let validity = combine_validities_and(coords.validity(), origin_validity);
    let values = coords.ignoring_validity().with_positions(positions);
    PointArray::new(Coords::from_valid(values, validity))
}

pub fn affine_line_strings(
    matrix: &AffineMatrix,
    array: &LineStringArray,
    shift: RowShift,
) -> LineStringArray {
    let (rows, coords) = affine_nested(matrix, &array.rows, &array.coords, &shift, |i| {
        array.rows.range(i)
    });
    LineStringArray::from_parts(rows, coords)
}

pub fn affine_multi_points(
    matrix: &AffineMatrix,
    array: &MultiPointArray,
    shift: RowShift,
) -> MultiPointArray {
    let (rows, coords) = affine_nested(matrix, &array.rows, &array.coords, &shift, |i| {
        array.rows.range(i)
    });
    MultiPointArray::from_parts(rows, coords)
}

pub fn affine_polygons(
    matrix: &AffineMatrix,
    array: &PolygonArray,
    shift: RowShift,
) -> PolygonArray {
    let (rows, coords) = affine_nested(matrix, &array.rows, &array.coords, &shift, |i| {
        array.rings.span(array.rows.range(i))
    });
    PolygonArray::from_parts(rows, array.rings.clone(), coords)
}

pub fn affine_multi_line_strings(
    matrix: &AffineMatrix,
    array: &MultiLineStringArray,
    shift: RowShift,
) -> MultiLineStringArray {
    let (rows, coords) = affine_nested(matrix, &array.rows, &array.coords, &shift, |i| {
        array.lines.span(array.rows.range(i))
    });
    MultiLineStringArray::from_parts(rows, array.lines.clone(), coords)
}

pub fn affine_multi_polygons(
    matrix: &AffineMatrix,
    array: &MultiPolygonArray,
    shift: RowShift,
) -> MultiPolygonArray {
    let (rows, coords) = affine_nested(matrix, &array.rows, &array.coords, &shift, |i| {
        array.rings.span(array.polygons.span(array.rows.range(i)))
    });
    MultiPolygonArray::from_parts(rows, array.polygons.clone(), array.rings.clone(), coords)
}

/// The new `rows` and `coords` of a nested kind, whose row `i` is coordinates `row_coords(i)`.
fn affine_nested(
    matrix: &AffineMatrix,
    rows: &List,
    coords: &ValidCoords,
    shift: &RowShift,
    row_coords: impl Fn(usize) -> Range<usize>,
) -> (List, ValidCoords) {
    let points = shift.points(rows.len());
    let positions = new_positions(matrix, coords.dim().has_z(), shift, |axis, row| {
        let rows = rows.len();
        let Some((coefficients, points)) = shift.coefficients(matrix, axis).zip(points) else {
            return axis_of_coords(row, coords, &row_coords, rows, None::<fn(usize) -> f64>);
        };
        let values = points.coords.ignoring_validity();
        let (x, y) = (values.x(), values.y());
        match values.z() {
            None => axis_of_coords(
                row,
                coords,
                &row_coords,
                rows,
                Some(point_shift(coefficients, axis, [x, y], rows)),
            ),
            Some(z) => axis_of_coords(
                row,
                coords,
                &row_coords,
                rows,
                Some(point_shift(coefficients, axis, [x, y, z], rows)),
            ),
        }
    });
    let origin_validity = points.and_then(|points| points.coords.validity());
    let validity = combine_validities_and(rows.validity(), origin_validity);
    (
        rows.clone().with_validity(validity),
        coords.with_positions(positions),
    )
}

/// The new `x`, `y` and `z`, each made by `transform(axis, row)`,
/// or `None` where that would leave the coordinate as it is.
/// The `z` is always `None` if `has_z=false`.
fn new_positions(
    matrix: &AffineMatrix,
    has_z: bool,
    shift: &RowShift,
    transform: impl Fn(usize, [f64; 4]) -> Vec<f64>,
) -> [Option<Vec<f64>>; 3] {
    std::array::from_fn(|axis| {
        let row = matrix.row(axis);
        let shifted = shift.coefficients(matrix, axis).is_some();
        let unchanged = matrix.keeps(axis) && row[3] == 0.0 && !shifted;
        (!unchanged && (axis < 2 || has_z)).then(|| transform(axis, row))
    })
}

/// The shift of output axis `axis` for each of `rows` rows (see [`RowShift`]).
/// The point is read from its `columns`, the x, y, and z (if `D=3`)
/// A point without a `z` is at `z` = 0, like one given as two numbers.
fn point_shift<'a, const D: usize>(
    coefficients: [f64; 3],
    axis: usize,
    columns: [&'a [f64]; D],
    rows: usize,
) -> impl Fn(usize) -> f64 + 'a {
    let coefficients: [f64; D] = std::array::from_fn(|k| coefficients[k]);
    let columns = columns.map(|column| &column[..rows]);
    move |row| {
        let point = columns.map(|column| column[row]);
        // `o − L.o`, rounded the same as the offset of an origin given as numbers.
        point.get(axis).copied().unwrap_or(0.0) - transform(coefficients, point)
    }
}

/// [`transform_points`] for one output `row` of the matrix,
/// with the position of `coords` as `[x, y]` or `[x, y, z]`.
fn axis_of_points(
    [a, b, c, offset]: [f64; 4],
    coords: &Coords,
    rows: usize,
    shift: Option<impl Fn(usize) -> f64>,
) -> Vec<f64> {
    let values = coords.ignoring_validity();
    let (x, y) = (values.x(), values.y());
    let nulls = !coords.is_all_valid();
    match values.z() {
        None => with_const_flags!(
            NULLS = nulls;
            transform_points::<2, NULLS>([a, b], offset, [x, y], coords, rows, shift)
        ),
        Some(z) => with_const_flags!(
            NULLS = nulls;
            transform_points::<3, NULLS>([a, b, c], offset, [x, y, z], coords, rows, shift)
        ),
    }
}

/// [`transform_coords`] dispatcher for z
fn axis_of_coords(
    [a, b, c, offset]: [f64; 4],
    coords: &ValidCoords,
    row_coords: impl Fn(usize) -> Range<usize>,
    rows: usize,
    shift: Option<impl Fn(usize) -> f64>,
) -> Vec<f64> {
    let (x, y) = (coords.x(), coords.y());
    match coords.z() {
        None => transform_coords([a, b], offset, [x, y], row_coords, rows, shift),
        Some(z) => transform_coords([a, b, c], offset, [x, y, z], row_coords, rows, shift),
    }
}

/// `coefficients` times `position`, summed in order.
/// A coefficient of 0 is left out, so the coordinate it multiplies has no say,
/// not even when it is NaN.
#[inline(always)]
fn transform<const D: usize>(coefficients: [f64; D], position: [f64; D]) -> f64 {
    // `-0.0 + v` is exactly `v`, and adding `-0.0` is exactly adding nothing.
    let mut sum = -0.0;
    for (coefficient, value) in coefficients.into_iter().zip(position) {
        sum += if coefficient != 0.0 {
            coefficient * value
        } else {
            -0.0
        };
    }
    sum
}

/// One output coordinate of each of the `rows` points: `coefficients` times its position
/// (from `columns`), plus its `shift(i)` if there is one, then `offset`.
/// A null point gets `0.0`, and is never read.
fn transform_points<const D: usize, const NULLS: bool>(
    coefficients: [f64; D],
    offset: f64,
    columns: [&[f64]; D],
    coords: &Coords,
    rows: usize,
    shift: Option<impl Fn(usize) -> f64>,
) -> Vec<f64> {
    // The same `rows` the shift has its columns cut to, so neither is checked per point.
    let len = rows;
    let columns = columns.map(|column| &column[..len]);
    let there = |i: usize| !NULLS || coords.is_valid(i);
    let position = |i: usize| columns.map(|column| column[i]);
    match shift {
        None => (0..len)
            .map(|i| {
                if there(i) {
                    transform(coefficients, position(i)) + offset
                } else {
                    0.0
                }
            })
            .collect(),
        // Shifted last, so a shift equal to a constant offset adds up exactly the same.
        Some(shift) => (0..len)
            .map(|i| {
                if there(i) {
                    transform(coefficients, position(i)) + shift(i) + offset
                } else {
                    0.0
                }
            })
            .collect(),
    }
}

/// One output coordinate of every coordinate: `coefficients` times its position (from `columns`),
/// plus its row's `shift(row)` if there is one, then `offset`.
/// Row `i` of `rows` is coordinates `row_coords(i)`.
/// Coordinates outside of every row are not shifted,
/// so there is always one output for every input.
fn transform_coords<const D: usize>(
    coefficients: [f64; D],
    offset: f64,
    columns: [&[f64]; D],
    row_coords: impl Fn(usize) -> Range<usize>,
    rows: usize,
    shift: Option<impl Fn(usize) -> f64>,
) -> Vec<f64> {
    let len = columns[0].len();
    let unshifted = |range: Range<usize>| {
        let columns = columns.map(|column| &column[range.clone()]);
        (0..range.len())
            .map(move |i| transform(coefficients, columns.map(|column| column[i])) + offset)
    };
    let Some(shift) = shift else {
        return unshifted(0..len).collect();
    };
    let mut out = Vec::with_capacity(len);
    for row in 0..rows {
        let shift = shift(row);
        let range = row_coords(row);
        // The rows follow one another, so only the coordinates before the first one are skipped.
        out.extend(unshifted(out.len()..range.start));
        let columns = columns.map(|column| &column[range.clone()]);
        // Shifted last, so a shift equal to a constant offset adds up exactly the same.
        out.extend(
            (0..range.len())
                .map(|i| transform(coefficients, columns.map(|column| column[i])) + shift + offset),
        );
    }
    out.extend(unshifted(out.len()..len));
    out
}

#[cfg(test)]
mod tests {
    //! What the Python suite cannot reach yet:
    //! a matrix that mixes coordinates, rather than only offsetting them.

    use polars_arrow::offset::OffsetsBuffer;

    use super::*;

    fn points(x: &[f64], y: &[f64], z: Option<&[f64]>) -> PointArray {
        let axes = (x.to_vec(), y.to_vec(), z.map(<[f64]>::to_vec), None);
        PointArray::new(Coords::from_valid(ValidCoords::from_axes(axes), None))
    }

    /// One linestring per item of `lengths`, over consecutive coordinates.
    fn line_strings(x: &[f64], y: &[f64], z: &[f64], lengths: &[i64]) -> LineStringArray {
        let offsets: Vec<i64> = std::iter::once(0)
            .chain(lengths.iter().scan(0, |end, len| {
                *end += len;
                Some(*end)
            }))
            .collect();
        let axes = (x.to_vec(), y.to_vec(), Some(z.to_vec()), None);
        LineStringArray::from_parts(
            List::from_offsets(OffsetsBuffer::try_from(offsets).unwrap(), None),
            ValidCoords::from_axes(axes),
        )
    }

    #[test]
    fn each_row_mixes_the_coordinates_in_order() {
        let matrix = AffineMatrix {
            coefficients: [
                1.0, 2.0, 3.0, 4.0, // x' = x + 2y + 3z + 4
                5.0, 6.0, 7.0, 8.0, // y' = 5x + 6y + 7z + 8
                9.0, 10.0, 11.0, 12.0, // z' = 9x + 10y + 11z + 12
            ],
        };
        let input = points(&[1.0, 0.0], &[0.0, 1.0], Some(&[2.0, 3.0]));

        let out = affine_points(&matrix, &input, RowShift::None);
        let out = out.coords.ignoring_validity();
        assert_eq!(out.x(), [11.0, 15.0]);
        assert_eq!(out.y(), [27.0, 35.0]);
        assert_eq!(out.z().unwrap(), [43.0, 55.0]);
    }

    #[test]
    fn without_z_the_z_column_does_nothing() {
        let matrix = AffineMatrix {
            coefficients: [
                0.0, -1.0, 100.0, 1.0, // x' = -y + 1, plus 100z if there were a z
                1.0, 0.0, 100.0, 2.0, // y' = x + 2, plus 100z if there were a z
                0.0, 0.0, 1.0, 0.0,
            ],
        };
        let input = points(&[1.0, 3.0], &[2.0, 4.0], None);

        let out = affine_points(&matrix, &input, RowShift::None);
        let out = out.coords.ignoring_validity();
        assert_eq!(out.x(), [-1.0, -3.0]);
        assert_eq!(out.y(), [3.0, 5.0]);
    }

    #[test]
    fn a_shift_moves_each_row_by_its_own_amount() {
        let matrix = AffineMatrix {
            coefficients: [
                0.0, -1.0, 0.0, 0.0, // x' = -y + shift
                1.0, 0.0, 0.0, 0.0, // y' = x
                0.0, 0.0, 1.0, 0.0, // z' = z + shift
            ],
        };
        // Two linestrings, of two and one coordinates.
        let input = line_strings(
            &[1.0, 3.0, 5.0],
            &[2.0, 4.0, 6.0],
            &[5.0, 6.0, 7.0],
            &[2, 1],
        );
        let by = points(&[10.0, 20.0], &[0.0, 0.0], Some(&[10.0, 20.0]));

        let out = affine_line_strings(&matrix, &input, RowShift::Then(&by));
        assert_eq!(out.coords.x(), [8.0, 6.0, 14.0]);
        assert_eq!(out.coords.y(), [1.0, 3.0, 5.0]);
        assert_eq!(out.coords.z().unwrap(), [15.0, 16.0, 27.0]);
    }

    #[test]
    fn a_coordinate_the_matrix_leaves_alone_is_not_rebuilt() {
        let matrix = AffineMatrix {
            coefficients: [
                1.0, 0.0, 0.0, 5.0, // x' = x + 5
                0.0, 1.0, 0.0, 0.0, // y' = y
                0.0, 0.0, 0.0, 7.0, // z' = 7
            ],
        };
        let input = points(&[1.0, 2.0], &[3.0, 4.0], Some(&[5.0, 6.0]));

        let out = affine_points(&matrix, &input, RowShift::None);
        let (out, input) = (
            out.coords.ignoring_validity(),
            input.coords.ignoring_validity(),
        );
        assert_eq!(out.x(), [6.0, 7.0]);
        assert_eq!(out.y().as_ptr(), input.y().as_ptr());
        assert_eq!(out.z().unwrap(), [7.0, 7.0]);
    }
}
