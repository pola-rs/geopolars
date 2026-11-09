//! Interfacing with PROJ, to move coordinates from one CRS to another.
//!
//! PROJ takes a coordinate as four floats: `x`, `y`, `z` and the time `t`.
//! It does so even for a 2D transformation.
//! `x` and `y` always go through it.
//! `z` only goes through it if `Z=true` (see below).
//! `m` never does.
//! What does not go through PROJ keeps its buffer.
//! `t` is the coordinate epoch, as a decimal year.
//! It is always `f64::INFINITY`, which PROJ reads as "no time given".
//!
//! # Const generic flags
//!
//! ## `Z`
//!
//! Every function here that reprojects many coordinates is generic over `Z`, or takes it as `z`.
//! If `Z=true`, `z` goes through PROJ too, which needs coordinates with a `z`.
//! If `Z=false`, PROJ gets `0.0` for `z`, and there is no new `z`.
//!
//! ## `NULLS`
//!
//! The reprojection of a column of points is generic over `NULLS`.
//! If `NULLS=false`, every point is assumed to be there.

// TODO: Add support for CRSes that have a time/epoch

use std::ffi::{CStr, CString};
use std::ops::Range;
use std::os::raw::c_int;
use std::ptr;

use geopolars_core::{Coords, ValidCoords};
use polars_arrow::bitmap::utils::SlicesIterator;
use proj::ProjError;
use proj_sys::{
    proj_context_create, proj_context_destroy, proj_context_errno, proj_context_errno_string,
    proj_create_crs_to_crs, proj_destroy, proj_errno, proj_errno_reset, proj_log_level,
    proj_normalize_for_visualization, proj_trans, proj_trans_generic, PJ, PJ_CONTEXT, PJ_COORD,
    PJ_DIRECTION_PJ_FWD, PJ_LOG_LEVEL_PJ_LOG_NONE, PJ_XYZT,
};

use crate::{GeoComputeError, GeoComputeResult};

/// A PROJ transformation from one CRS to another.
/// `x` is the easting or longitude, and `y` the northing or latitude, whatever the CRSs' axis order.
/// Cannot be shared between threads.
pub struct Transformation {
    ctx: *mut PJ_CONTEXT,
    pj: *mut PJ,
}

impl Transformation {
    /// `from` and `to` are anything PROJ accepts.
    pub fn new(from: &str, to: &str) -> GeoComputeResult<Self> {
        let error = |message: String| GeoComputeError::Transformation {
            from: from.to_owned(),
            to: to.to_owned(),
            message,
        };
        let nul = |_| error("a CRS cannot contain a NUL byte".to_owned());
        let (c_from, c_to) = (
            CString::new(from).map_err(nul)?,
            CString::new(to).map_err(nul)?,
        );

        // SAFETY: no preconditions; null is checked.
        let ctx = unsafe { proj_context_create() };
        if ctx.is_null() {
            return Err(error("failed to create a PROJ context".to_owned()));
        }
        // Owns `ctx` from here on, so it is freed on every return below.
        let mut transformation = Self {
            ctx,
            pj: ptr::null_mut(),
        };
        // SAFETY: `ctx` is alive, and every PJ created in it is checked for null.
        unsafe {
            let pj = proj_create_crs_to_crs(ctx, c_from.as_ptr(), c_to.as_ptr(), ptr::null_mut());
            if pj.is_null() {
                return Err(error(context_error(ctx)));
            }
            let normalised = proj_normalize_for_visualization(ctx, pj);
            proj_destroy(pj);
            if normalised.is_null() {
                return Err(error(context_error(ctx)));
            }
            transformation.pj = normalised;
        }
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
    let positions = with_const_flags!(
        NULLS = !coords.is_all_valid(), Z = z;
        reprojected_points::<NULLS, Z>(transformation, coords)
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
    let positions = with_const_flags!(Z = z; reprojected_valid::<Z>(transformation, coords))?;
    Ok(coords.with_positions(positions))
}

/// The new positions of [`reproject_coords`].
fn reprojected_points<const NULLS: bool, const Z: bool>(
    transformation: &Transformation,
    coords: &Coords,
) -> GeoComputeResult<[Option<Vec<f64>>; 3]> {
    let values = coords.ignoring_validity();
    let mut positions = copied_positions::<Z>(values);
    match coords.validity().filter(|_| NULLS) {
        // Only the runs of points that are there go to PROJ,
        // so a null point keeps the coordinates it was copied with.
        Some(validity) => {
            for (start, len) in SlicesIterator::new(validity) {
                reproject_positions_in_place::<Z>(
                    transformation,
                    &mut positions,
                    start..start + len,
                );
            }
        }
        None => reproject_positions_in_place::<Z>(transformation, &mut positions, 0..coords.len()),
    }
    retry_failed_positions::<Z>(transformation, values, &mut positions, |i| {
        NULLS && !coords.is_valid(i)
    })?;
    let [xs, ys, zs] = positions;
    Ok([Some(xs), Some(ys), Z.then_some(zs)])
}

/// The new positions of [`reproject_valid_coords`].
fn reprojected_valid<const Z: bool>(
    transformation: &Transformation,
    coords: &ValidCoords,
) -> GeoComputeResult<[Option<Vec<f64>>; 3]> {
    let mut positions = copied_positions::<Z>(coords);
    reproject_positions_in_place::<Z>(transformation, &mut positions, 0..coords.len());
    retry_failed_positions::<Z>(transformation, coords, &mut positions, |_| false)?;
    let [xs, ys, zs] = positions;
    Ok([Some(xs), Some(ys), Z.then_some(zs)])
}

fn copied_positions<const Z: bool>(coords: &ValidCoords) -> [Vec<f64>; 3] {
    let zs = if Z {
        coords.z().unwrap().to_vec()
    } else {
        Vec::new()
    };
    [coords.x().to_vec(), coords.y().to_vec(), zs]
}

fn set_position<const Z: bool>([xs, ys, zs]: &mut [Vec<f64>; 3], i: usize, [x, y, z]: [f64; 3]) {
    xs[i] = x;
    ys[i] = y;
    if Z {
        zs[i] = z;
    }
}

/// Moves the positions in `range` through PROJ in a single call.
/// A coordinate PROJ fails on gets an `x` of `f64::INFINITY` rather than an error,
/// so [retry_failed_positions] has to follow.
fn reproject_positions_in_place<const Z: bool>(
    transformation: &Transformation,
    [xs, ys, zs]: &mut [Vec<f64>; 3],
    range: Range<usize>,
) {
    let len = range.len();
    let stride = size_of::<f64>();
    let (z, z_len) = if Z {
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
        // Only the one `retry_failed_positions` reports should be, as before batching.
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
            z,
            stride,
            z_len,
            ptr::null_mut(),
            0,
            0,
        );
        proj_log_level(transformation.ctx, log_level);
    }
}

/// Reprojects every coordinate that [reproject_positions_in_place] failed on once more, on its own,
/// to get the error back. A coordinate for which `is_skipped` holds is left as it is.
fn retry_failed_positions<const Z: bool>(
    transformation: &Transformation,
    original: &ValidCoords,
    positions: &mut [Vec<f64>; 3],
    is_skipped: impl Fn(usize) -> bool,
) -> GeoComputeResult<()> {
    for i in 0..original.len() {
        // A coordinate can also come out infinite without failing, e.g. one that went in infinite.
        // On its own, it then comes out the same again.
        if positions[0][i] == f64::INFINITY && !is_skipped(i) {
            let position = reproject_position(transformation, original.position::<Z>(i))?;
            set_position::<Z>(positions, i, position);
        }
    }
    Ok(())
}

/// `[x, y, z]` through PROJ (see the module docs).
fn reproject_position(
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
