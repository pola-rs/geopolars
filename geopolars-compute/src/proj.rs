//! Interfacing with PROJ, to move coordinates from one CRS to another.
//!
//! PROJ takes a coordinate as four floats: `x`, `y`, `z` and the time `t`.
//! It does so even for a 2D transformation.
//! `x` and `y` always go through it.
//! `z` only goes through it if `z=true` (see below).
//! `m` never does.
//! What does not go through PROJ keeps its buffer.
//! `t` is the coordinate epoch, as a decimal year.
//! It is always `f64::INFINITY`, which PROJ reads as "no time given".
//!
//! # `z`
//!
//! If `z=true`, `z` goes through PROJ too, which needs coordinates with a `z`.
//! If `z=false`, PROJ gets `0.0` for `z`, and there is no new `z`.

// TODO: Add support for CRSes that have a time/epoch

use std::ffi::{CStr, CString};
use std::ops::Range;
use std::os::raw::c_int;
use std::ptr;

use geopolars_views::{Coords, ValidCoords};
use polars_arrow::bitmap::utils::SlicesIterator;
use polars_arrow::bitmap::Bitmap;
use proj::ProjError;
use proj_sys::{
    proj_clone, proj_context_create, proj_context_destroy, proj_context_errno,
    proj_context_errno_string, proj_create_crs_to_crs, proj_destroy, proj_errno, proj_errno_reset,
    proj_log_level, proj_normalize_for_visualization, proj_trans, proj_trans_generic, PJ,
    PJ_CONTEXT, PJ_COORD, PJ_DIRECTION_PJ_FWD, PJ_LOG_LEVEL_PJ_LOG_NONE, PJ_XYZT,
};

use crate::{GeoComputeError, GeoComputeResult};

/// A PROJ transformation from one CRS to another.
/// `x` is the easting or longitude, and `y` the northing or latitude, whatever the CRSs' axis order.
/// Can be moved to another thread, but not shared between threads.
pub struct Transformation {
    ctx: *mut PJ_CONTEXT,
    pj: *mut PJ,
    from: String,
    to: String,
}

// SAFETY: a `Transformation` owns its context and every PROJ object in it.
// PROJ lets a context move between threads, as long as one thread at a time uses it,
// which holds as `Transformation` is not `Sync`.
unsafe impl Send for Transformation {}

impl Transformation {
    /// `from` and `to` are anything PROJ accepts.
    pub fn new(from: &str, to: &str) -> GeoComputeResult<Self> {
        let nul = |_| GeoComputeError::Transformation {
            from: from.to_owned(),
            to: to.to_owned(),
            message: "a CRS cannot contain a NUL byte".to_owned(),
        };
        let (c_from, c_to) = (
            CString::new(from).map_err(nul)?,
            CString::new(to).map_err(nul)?,
        );
        // SAFETY: `ctx` is alive, and a null PJ is not used.
        Self::in_new_context(from, to, |ctx| unsafe {
            let pj = proj_create_crs_to_crs(ctx, c_from.as_ptr(), c_to.as_ptr(), ptr::null_mut());
            if pj.is_null() {
                return pj;
            }
            let normalised = proj_normalize_for_visualization(ctx, pj);
            proj_destroy(pj);
            normalised
        })
    }

    /// The same transformation, in a PROJ context of its own.
    /// This costs a few percent of [new](Self::new), which has to look the transformation up in PROJ's database.
    pub fn try_clone(&self) -> GeoComputeResult<Self> {
        // SAFETY: `self.pj` is alive while `self` is, and PROJ only reads it.
        Self::in_new_context(&self.from, &self.to, |ctx| unsafe {
            proj_clone(ctx, self.pj)
        })
    }

    /// A new context, with the PJ that `create_pj` makes in it, where null means PROJ failed.
    fn in_new_context(
        from: &str,
        to: &str,
        create_pj: impl FnOnce(*mut PJ_CONTEXT) -> *mut PJ,
    ) -> GeoComputeResult<Self> {
        let error = |message: String| GeoComputeError::Transformation {
            from: from.to_owned(),
            to: to.to_owned(),
            message,
        };
        // SAFETY: no preconditions; null is checked.
        let ctx = unsafe { proj_context_create() };
        if ctx.is_null() {
            return Err(error("failed to create a PROJ context".to_owned()));
        }
        // Owns `ctx` from here on, so it is freed on every return below.
        let mut transformation = Self {
            ctx,
            pj: ptr::null_mut(),
            from: from.to_owned(),
            to: to.to_owned(),
        };
        let pj = create_pj(ctx);
        if pj.is_null() {
            // SAFETY: `ctx` is alive.
            return Err(error(unsafe { context_error(ctx) }));
        }
        transformation.pj = pj;
        Ok(transformation)
    }
}

impl Drop for Transformation {
    fn drop(&mut self) {
        // SAFETY: both are owned, and the PJ is freed before the context it lives in.
        unsafe {
            if !self.pj.is_null() {
                proj_destroy(self.pj);
            }
            proj_context_destroy(self.ctx);
        }
    }
}

/// For a column of points.
/// A null point keeps its coordinates, and never fails.
pub fn reproject_coords(
    transformation: &Transformation,
    coords: &Coords,
    z: bool,
) -> GeoComputeResult<Coords> {
    assert!(
        !z || coords.dim().has_z(),
        "a z for coordinates without one"
    );
    let positions = reprojected(
        transformation,
        coords.ignoring_validity(),
        coords.validity(),
        z,
    )?;
    let axes = coords.ignoring_validity().with_positions(positions);
    Ok(Coords::from_valid(axes, coords.validity().cloned()))
}

/// [`reproject_coords`], for the coordinates of a nested kind, which are never null.
pub fn reproject_valid_coords(
    transformation: &Transformation,
    coords: &ValidCoords,
    z: bool,
) -> GeoComputeResult<ValidCoords> {
    assert!(
        !z || coords.dim().has_z(),
        "a z for coordinates without one"
    );
    let positions = reprojected(transformation, coords, None, z)?;
    Ok(coords.with_positions(positions))
}

/// The new positions of [`reproject_coords`] and [`reproject_valid_coords`].
/// Only the coordinates that `validity` has are reprojected.
fn reprojected(
    transformation: &Transformation,
    coords: &ValidCoords,
    validity: Option<&Bitmap>,
    z: bool,
) -> GeoComputeResult<[Option<Vec<f64>>; 3]> {
    let mut positions = copied_positions(coords, z);
    match validity {
        // Only the runs of points that are there go to PROJ,
        // so a null point keeps the coordinates it was copied with.
        Some(validity) => {
            for (start, len) in SlicesIterator::new(validity) {
                reproject_positions_in_place(transformation, &mut positions, start..start + len, z);
            }
        }
        None => reproject_positions_in_place(transformation, &mut positions, 0..coords.len(), z),
    }
    retry_failed_positions(transformation, coords, &mut positions, z, |i| {
        validity.is_some_and(|validity| !validity.get_bit(i))
    })?;
    let [xs, ys, zs] = positions;
    Ok([Some(xs), Some(ys), z.then_some(zs)])
}

fn copied_positions(coords: &ValidCoords, z: bool) -> [Vec<f64>; 3] {
    let zs = if z {
        coords.z().unwrap().to_vec()
    } else {
        Vec::new()
    };
    [coords.x().to_vec(), coords.y().to_vec(), zs]
}

fn set_position([xs, ys, zs]: &mut [Vec<f64>; 3], i: usize, [x, y, new_z]: [f64; 3], z: bool) {
    xs[i] = x;
    ys[i] = y;
    if z {
        zs[i] = new_z;
    }
}

/// Moves the positions in `range` through PROJ in a single call.
/// A coordinate PROJ fails on is marked as [failed_in_batch] rather than giving an error,
/// so [retry_failed_positions] or [ReprojectedXy::check] has to follow.
fn reproject_positions_in_place(
    transformation: &Transformation,
    [xs, ys, zs]: &mut [Vec<f64>; 3],
    range: Range<usize>,
    z: bool,
) {
    let len = range.len();
    let stride = size_of::<f64>();
    let (zs, z_len) = if z {
        (zs[range.clone()].as_mut_ptr(), len)
    } else {
        (ptr::null_mut(), 0)
    };
    let (xs, ys) = (&mut xs[range.clone()], &mut ys[range]);
    // SAFETY: `pj` and its context are alive while `transformation` is.
    // PROJ reads and writes `len` values of each buffer it gets, which all hold that many.
    // A null buffer it reads as `0.0` for `z` and as "no time given" for `t`.
    unsafe {
        // PROJ logs every coordinate it fails on to stderr.
        // Only one retried on its own should be, as before batching.
        let log_level = proj_log_level(transformation.ctx, PJ_LOG_LEVEL_PJ_LOG_NONE);
        proj_trans_generic(
            transformation.pj,
            PJ_DIRECTION_PJ_FWD,
            xs.as_mut_ptr(),
            stride,
            len,
            ys.as_mut_ptr(),
            stride,
            len,
            zs,
            stride,
            z_len,
            ptr::null_mut(),
            0,
            0,
        );
        proj_log_level(transformation.ctx, log_level);
    }
}

/// Whether [reproject_positions_in_place] failed on the position whose new `x` this is.
/// A coordinate can also come out infinite without failing, e.g. one that went in infinite.
/// On its own, it then comes out the same again.
#[inline(always)]
fn failed_in_batch(x: f64) -> bool {
    x == f64::INFINITY
}

/// The `x` and `y` of a run of vertices of some coordinates, moved through PROJ in a single call.
pub struct ReprojectedXy {
    x: Vec<f64>,
    y: Vec<f64>,
    /// The index in the coordinates of the first vertex.
    first: usize,
}

impl ReprojectedXy {
    pub(crate) fn new(
        transformation: &Transformation,
        coords: &ValidCoords,
        vertices: Range<usize>,
    ) -> Self {
        let mut positions = [
            coords.x()[vertices.clone()].to_vec(),
            coords.y()[vertices.clone()].to_vec(),
            Vec::new(),
        ];
        reproject_positions_in_place(transformation, &mut positions, 0..vertices.len(), false);
        let [x, y, _] = positions;
        Self {
            x,
            y,
            first: vertices.start,
        }
    }

    /// The error of the first of `vertices` PROJ failed on, if any,
    /// by reprojecting each it failed on once more, on its own.
    /// `transformation` and `coords` must be the ones it was made with.
    #[inline(always)]
    pub(crate) fn check(
        &self,
        transformation: &Transformation,
        coords: &ValidCoords,
        vertices: Range<usize>,
    ) -> GeoComputeResult<()> {
        let x = &self.x[vertices.start - self.first..vertices.end - self.first];
        // Does not stop at the first that failed, so that it vectorises.
        if x.iter().fold(true, |ok, &x| ok & !failed_in_batch(x)) {
            return Ok(());
        }
        for (i, &x) in vertices.zip(x) {
            if failed_in_batch(x) {
                // Comes back fine for one that went in infinite, which keeps its value from the batch.
                reproject_position(transformation, coords.position::<false>(i))?;
            }
        }
        Ok(())
    }

    /// The `x` and `y` of vertex `i` of the coordinates.
    /// `i` must lie in its vertices, in a run that passed [check](Self::check).
    #[inline(always)]
    pub(crate) fn xy(&self, i: usize) -> (f64, f64) {
        let j = i - self.first;
        (self.x[j], self.y[j])
    }
}

/// Reprojects every coordinate that [reproject_positions_in_place] failed on once more, on its own,
/// to get the error back. A coordinate for which `is_skipped` holds is left as it is.
fn retry_failed_positions(
    transformation: &Transformation,
    original: &ValidCoords,
    positions: &mut [Vec<f64>; 3],
    z: bool,
    is_skipped: impl Fn(usize) -> bool,
) -> GeoComputeResult<()> {
    for i in 0..original.len() {
        if failed_in_batch(positions[0][i]) && !is_skipped(i) {
            let position = if z {
                original.position::<true>(i)
            } else {
                original.position::<false>(i)
            };
            let position = reproject_position(transformation, position)?;
            set_position(positions, i, position, z);
        }
    }
    Ok(())
}

/// `[x, y, z]` through PROJ (see the module docs).
pub(crate) fn reproject_position(
    transformation: &Transformation,
    [x, y, z]: [f64; 3],
) -> GeoComputeResult<[f64; 3]> {
    let pj = transformation.pj;
    // SAFETY: `pj` is alive while `transformation` is. Every field of the union is 4 f64s.
    let (out, errno) = unsafe {
        proj_errno_reset(pj);
        let xyzt = PJ_XYZT {
            x,
            y,
            z,
            t: f64::INFINITY,
        };
        let out = proj_trans(pj, PJ_DIRECTION_PJ_FWD, PJ_COORD { xyzt });
        (out.xyzt, proj_errno(pj))
    };
    if errno != 0 {
        return Err(GeoComputeError::Reproject {
            position: (x, y),
            // SAFETY: the context is alive while `transformation` is.
            source: ProjError::Conversion(unsafe { errno_string(transformation.ctx, errno) }),
        });
    }
    Ok([out.x, out.y, out.z])
}

/// What PROJ last reported going wrong in `ctx`.
///
/// # Safety
/// `ctx` must be alive.
unsafe fn context_error(ctx: *mut PJ_CONTEXT) -> String {
    errno_string(ctx, proj_context_errno(ctx))
}

/// PROJ's message for `errno`, copied out.
///
/// # Safety
/// `ctx` must be alive.
unsafe fn errno_string(ctx: *mut PJ_CONTEXT, errno: c_int) -> String {
    // PROJ owns the message, which is copied out before the next call.
    let message = proj_context_errno_string(ctx, errno);
    if message.is_null() {
        "unknown error".to_owned()
    } else {
        CStr::from_ptr(message).to_string_lossy().into_owned()
    }
}
