//! What PROJ knows about a CRS:
//! the longitude/latitude CRS behind it and its ellipsoid, its bounds, and whether it is equal-area.
//!
//! The `proj` crate only exposes transformations.
//! So we need to interface directly through `proj-sys`.

use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::os::raw::c_int;
use std::ptr;

use polars::prelude::*;
use proj_sys::{
    proj_as_projjson, proj_context_create, proj_context_destroy, proj_context_errno,
    proj_context_errno_string, proj_coordoperation_get_method_info, proj_create,
    proj_create_crs_to_crs_from_pj, proj_crs_get_coordinate_system, proj_crs_get_coordoperation,
    proj_crs_get_geodetic_crs, proj_crs_get_sub_crs, proj_cs_get_axis_count, proj_cs_get_axis_info,
    proj_cs_get_type, proj_destroy, proj_ellipsoid_get_parameters, proj_get_area_of_use,
    proj_get_ellipsoid, proj_get_type, proj_normalize_for_visualization, proj_trans_bounds, PJ,
    PJ_CONTEXT, PJ_COORDINATE_SYSTEM_TYPE_PJ_CS_TYPE_ELLIPSOIDAL, PJ_DIRECTION_PJ_FWD,
    PJ_TYPE_PJ_TYPE_COMPOUND_CRS, PJ_TYPE_PJ_TYPE_GEOGRAPHIC_2D_CRS,
    PJ_TYPE_PJ_TYPE_GEOGRAPHIC_3D_CRS, PJ_TYPE_PJ_TYPE_PROJECTED_CRS,
};

/// The geographic CRS a CRS is defined on: the CRS itself if it is one already,
/// the CRS it projects from if it is projected.
///
/// Going from a CRS to its own geographic CRS never shifts datum:
/// it only undoes the projection, so no accuracy is lost on the way.
#[derive(Clone, Debug, PartialEq)]
pub struct GeodeticCrs {
    /// The geographic CRS, as PROJJSON.
    pub definition: String,
    /// The ellipsoid's equatorial radius, in the unit of the CRS's own `x` and `y`,
    /// or in metres when those are degrees.
    /// Everything measured on this ellipsoid comes out in that unit.
    pub semi_major: f64,
    /// The ellipsoid's flattening; `0.0` for a sphere.
    pub flattening: f64,
}

impl GeodeticCrs {
    /// `crs` is anything PROJ accepts.
    pub fn of(crs: &str) -> PolarsResult<Self> {
        let ctx = Context::new()?;
        let c_crs = CString::new(crs)
            .map_err(|_| polars_err!(ComputeError: "a CRS cannot contain a NUL byte: {crs:?}"))?;

        // SAFETY: every pointer handed to PROJ comes from `ctx` and is checked
        // for null by `Object::new`, which also frees it again.
        unsafe {
            let parsed = ctx.object(proj_create(ctx.0, c_crs.as_ptr()), crs)?;
            let geodetic = ctx.object(proj_crs_get_geodetic_crs(ctx.0, parsed.0), crs)?;
            let meters_per_unit = ctx.meters_per_horizontal_unit(&parsed, crs)?;

            let kind = proj_get_type(geodetic.0);
            polars_ensure!(
                kind == PJ_TYPE_PJ_TYPE_GEOGRAPHIC_2D_CRS || kind == PJ_TYPE_PJ_TYPE_GEOGRAPHIC_3D_CRS,
                ComputeError: "{crs} is not defined on longitude/latitude, so there is no ellipsoid to measure on"
            );

            let ellipsoid = ctx.object(proj_get_ellipsoid(ctx.0, geodetic.0), crs)?;
            let (mut semi_major, mut inverse_flattening) = (0.0, 0.0);
            let ok = proj_ellipsoid_get_parameters(
                ctx.0,
                ellipsoid.0,
                &mut semi_major,
                ptr::null_mut(),
                ptr::null_mut(),
                &mut inverse_flattening,
            );
            if ok == 0 {
                return Err(ctx.error(crs));
            }

            let json = proj_as_projjson(ctx.0, geodetic.0, ptr::null());
            if json.is_null() {
                return Err(ctx.error(crs));
            }
            // Owned by `geodetic`: copy it out before that is freed.
            let definition = CStr::from_ptr(json).to_string_lossy().into_owned();

            Ok(Self {
                definition,
                // PROJ gives it in metres.
                semi_major: semi_major / meters_per_unit,
                // PROJ gives an inverse flattening of 0 for a sphere.
                flattening: if inverse_flattening == 0.0 {
                    0.0
                } else {
                    1.0 / inverse_flattening
                },
            })
        }
    }
}

/// How far `x` goes in one full turn around the globe, in the CRS's own unit:
/// `360.0` for degrees, `400.0` for grads.
/// `None` if `x` is not a longitude, as in a projected CRS.
///
/// `crs` is anything PROJ accepts.
pub fn longitude_turn(crs: &str) -> PolarsResult<Option<f64>> {
    Ok(geographic_turns(crs)?.map(|[x, _]| x))
}

/// How far `x` and `y` go in one full turn around the globe, each in the unit of its own axis.
/// `None` if they are not a longitude and latitude, as in a projected CRS.
///
/// GeoArrow always stores longitude as `x` and latitude as `y`,
/// whatever axis order the CRS declares (EPSG:4326 declares latitude first).
/// So the axes are read from the CRS normalised to that order.
///
/// `crs` is anything PROJ accepts.
fn geographic_turns(crs: &str) -> PolarsResult<Option<[f64; 2]>> {
    let ctx = Context::new()?;
    let c_crs = CString::new(crs)
        .map_err(|_| polars_err!(ComputeError: "a CRS cannot contain a NUL byte: {crs:?}"))?;

    // SAFETY: as in `GeodeticCrs::of`.
    unsafe {
        let parsed = ctx.object(proj_create(ctx.0, c_crs.as_ptr()), crs)?;
        let horizontal = ctx.horizontal(&parsed, crs)?;
        let normalized = ctx.object(
            proj_normalize_for_visualization(ctx.0, horizontal.crs().0),
            crs,
        )?;
        let cs = ctx.object(proj_crs_get_coordinate_system(ctx.0, normalized.0), crs)?;
        if proj_cs_get_type(ctx.0, cs.0) != PJ_COORDINATE_SYSTEM_TYPE_PJ_CS_TYPE_ELLIPSOIDAL {
            return Ok(None);
        }
        let turn_of = |axis| -> PolarsResult<f64> {
            let turn = std::f64::consts::TAU / ctx.unit_factor(&cs, axis, crs)?;
            // EPSG rounds a unit's size to 15 digits, which would make a degree's turn
            // 359.99999999999994. Units with a whole turn get it back exactly.
            let whole = turn.round();
            Ok(if (turn - whole).abs() <= 1e-12 * whole {
                whole
            } else {
                turn
            })
        };
        Ok(Some([turn_of(0)?, turn_of(1)?]))
    }
}

/// Gets a bounding box of the CRS provided, as `[xmin, ymin, xmax, ymax]`.
/// The flags pick one of the `crs_bounds` of the Python `fix_out_of_bounds`, which documents them.
/// A geographic CRS is bounded in the unit of its own axes (see [geographic_turns]).
///
/// `crs` is anything PROJ accepts.
pub fn crs_bounds(
    crs: &str,
    allow_wrapped_longitude: bool,
    within_area_of_use: bool,
) -> PolarsResult<Option<[f64; 4]>> {
    let globe = geographic_turns(crs)?.map(|[x_turn, y_turn]| {
        let east = if allow_wrapped_longitude {
            x_turn
        } else {
            x_turn / 2.0
        };
        [-x_turn / 2.0, -y_turn / 4.0, east, y_turn / 4.0]
    });
    if !within_area_of_use {
        return Ok(globe);
    }
    let [xmin, ymin, xmax, ymax] = area_of_use(crs)?;
    let area = if xmin > xmax {
        // Crossing the antimeridian: only a geographic CRS gives that back.
        [f64::NEG_INFINITY, ymin, f64::INFINITY, ymax]
    } else {
        [xmin, ymin, xmax, ymax]
    };
    Ok(Some(match globe {
        None => area,
        Some(globe) => [
            globe[0].max(area[0]),
            globe[1].max(area[1]),
            globe[2].min(area[2]),
            globe[3].min(area[3]),
        ],
    }))
}

/// The area PROJ says `crs` is meant for
/// interpret as a bounding box
fn area_of_use(crs: &str) -> PolarsResult<[f64; 4]> {
    let ctx = Context::new()?;
    let c_crs = CString::new(crs)
        .map_err(|_| polars_err!(ComputeError: "a CRS cannot contain a NUL byte: {crs:?}"))?;
    let lonlat = CString::new("EPSG:4326").unwrap();

    // SAFETY: as in `GeodeticCrs::of`.
    unsafe {
        let parsed = ctx.object(proj_create(ctx.0, c_crs.as_ptr()), crs)?;
        let (mut west, mut south, mut east, mut north) = (0.0, 0.0, 0.0, 0.0);
        let known = proj_get_area_of_use(
            ctx.0,
            parsed.0,
            &mut west,
            &mut south,
            &mut east,
            &mut north,
            ptr::null_mut(),
        );
        // PROJ fills in -1000 for what it does not know.
        polars_ensure!(
            known != 0 && west != -1000.0,
            ComputeError: "PROJ knows no area of use for the CRS {crs}"
        );

        // Both normalised, so `x` is the longitude or easting whatever the axis order.
        let horizontal = ctx.horizontal(&parsed, crs)?;
        let to = ctx.object(
            proj_normalize_for_visualization(ctx.0, horizontal.crs().0),
            crs,
        )?;
        let from = ctx.object(proj_create(ctx.0, lonlat.as_ptr()), crs)?;
        let from = ctx.object(proj_normalize_for_visualization(ctx.0, from.0), crs)?;
        let transform = ctx.object(
            proj_create_crs_to_crs_from_pj(ctx.0, from.0, to.0, ptr::null_mut(), ptr::null()),
            crs,
        )?;
        let mut out = [0.0; 4];
        let [xmin, ymin, xmax, ymax] = &mut out;
        let ok = proj_trans_bounds(
            ctx.0,
            transform.0,
            PJ_DIRECTION_PJ_FWD,
            west,
            south,
            east,
            north,
            xmin,
            ymin,
            xmax,
            ymax,
            // The edges are sampled at this many points, as PROJ recommends.
            21,
        );
        if ok == 0 {
            return Err(ctx.error(crs));
        }
        Ok(out)
    }
}

/// Whether `crs` says what `z` means.
/// That is the case when it is compound (horizontal + vertical),
/// or when its coordinate system has three axes.
pub fn specifies_z(crs: &str) -> PolarsResult<bool> {
    let ctx = Context::new()?;
    let c_crs = CString::new(crs)
        .map_err(|_| polars_err!(ComputeError: "a CRS cannot contain a NUL byte: {crs:?}"))?;

    // SAFETY: as in `GeodeticCrs::of`.
    unsafe {
        let parsed = ctx.object(proj_create(ctx.0, c_crs.as_ptr()), crs)?;
        if proj_get_type(parsed.0) == PJ_TYPE_PJ_TYPE_COMPOUND_CRS {
            return Ok(true);
        }
        let cs = ctx.object(proj_crs_get_coordinate_system(ctx.0, parsed.0), crs)?;
        Ok(proj_cs_get_axis_count(ctx.0, cs.0) == 3)
    }
}

/// Equal-area projection methods whose name does not say "Equal Area".
const EQUAL_AREA_METHODS: &[&str] = &[
    "Equal Earth",
    "Mollweide",
    "Sinusoidal",
    "Eckert IV",
    "Eckert VI",
    "Goode Homolosine",
    "Interrupted Goode Homolosine",
    "Bonne",
];

/// Whether `crs` is projected with an equal-area projection,
/// so that areas on its plane are areas on the ellipsoid.
/// `false` for a CRS that is not projected, such as one in longitude/latitude.
///
/// `crs` is anything PROJ accepts.
pub fn is_equal_area(crs: &str) -> PolarsResult<bool> {
    let ctx = Context::new()?;
    let c_crs = CString::new(crs)
        .map_err(|_| polars_err!(ComputeError: "a CRS cannot contain a NUL byte: {crs:?}"))?;

    // SAFETY: as in `GeodeticCrs::of`.
    unsafe {
        let parsed = ctx.object(proj_create(ctx.0, c_crs.as_ptr()), crs)?;
        let horizontal = ctx.horizontal(&parsed, crs)?;
        if proj_get_type(horizontal.crs().0) != PJ_TYPE_PJ_TYPE_PROJECTED_CRS {
            return Ok(false);
        }
        let conversion = ctx.object(proj_crs_get_coordoperation(ctx.0, horizontal.crs().0), crs)?;
        let mut name = ptr::null();
        let ok = proj_coordoperation_get_method_info(
            ctx.0,
            conversion.0,
            &mut name,
            ptr::null_mut(),
            ptr::null_mut(),
        );
        if ok == 0 || name.is_null() {
            return Err(ctx.error(crs));
        }
        // Owned by `conversion`, which is still alive.
        let name = CStr::from_ptr(name).to_string_lossy();
        Ok(name.contains("Equal Area") || EQUAL_AREA_METHODS.contains(&name.as_ref()))
    }
}

/// A PROJ context of our own.
/// Cannot be shared between threads.
struct Context(*mut PJ_CONTEXT);

impl Context {
    fn new() -> PolarsResult<Self> {
        // SAFETY: no preconditions; null is checked.
        let ctx = unsafe { proj_context_create() };
        polars_ensure!(!ctx.is_null(), ComputeError: "failed to create a PROJ context");
        Ok(Self(ctx))
    }

    /// Takes ownership of `pj`, which PROJ created in this context.
    fn object(&self, pj: *mut PJ, crs: &str) -> PolarsResult<Object<'_>> {
        if pj.is_null() {
            return Err(self.error(crs));
        }
        Ok(Object(pj, PhantomData))
    }

    /// How many metres one unit of `crs`'s `x` and `y` is, `1.0` if they are angles.
    ///
    /// # Safety
    /// `crs` must be a CRS created in this context.
    unsafe fn meters_per_horizontal_unit(&self, crs: &Object<'_>, name: &str) -> PolarsResult<f64> {
        let horizontal = self.horizontal(crs, name)?;
        let cs = self.object(
            proj_crs_get_coordinate_system(self.0, horizontal.crs().0),
            name,
        )?;
        if proj_cs_get_type(self.0, cs.0) == PJ_COORDINATE_SYSTEM_TYPE_PJ_CS_TYPE_ELLIPSOIDAL {
            return Ok(1.0);
        }
        self.unit_factor(&cs, 0, name)
    }

    /// The part of `crs` that holds `x` and `y`:
    /// a compound CRS (horizontal + vertical) puts them in its first part.
    ///
    /// # Safety
    /// `crs` must be a CRS created in this context.
    unsafe fn horizontal<'a>(
        &'a self,
        crs: &'a Object<'a>,
        name: &str,
    ) -> PolarsResult<Horizontal<'a>> {
        Ok(if proj_get_type(crs.0) == PJ_TYPE_PJ_TYPE_COMPOUND_CRS {
            Horizontal::Part(self.object(proj_crs_get_sub_crs(self.0, crs.0, 0), name)?)
        } else {
            Horizontal::Whole(crs)
        })
    }

    /// The size of one unit of axis `axis` of the coordinate system `cs`:
    /// in metres for a length, in radians for an angle.
    ///
    /// # Safety
    /// `cs` must be a coordinate system created in this context.
    unsafe fn unit_factor(&self, cs: &Object<'_>, axis: c_int, name: &str) -> PolarsResult<f64> {
        let mut factor = 0.0;
        let ok = proj_cs_get_axis_info(
            self.0,
            cs.0,
            axis,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut factor,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        );
        if ok == 0 {
            return Err(self.error(name));
        }
        Ok(factor)
    }

    /// What PROJ last reported going wrong in this context.
    fn error(&self, crs: &str) -> PolarsError {
        // SAFETY: the context is alive; PROJ owns the message, which is copied out.
        let message = unsafe {
            let errno: c_int = proj_context_errno(self.0);
            let message = proj_context_errno_string(self.0, errno);
            if message.is_null() {
                "unknown error".to_owned()
            } else {
                CStr::from_ptr(message).to_string_lossy().into_owned()
            }
        };
        polars_err!(ComputeError: "PROJ cannot use the CRS {crs}: {message}")
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: every `Object` borrows the context, so they are all gone by now.
        unsafe { proj_context_destroy(self.0) };
    }
}

/// See [`Context::horizontal`].
enum Horizontal<'a> {
    Whole(&'a Object<'a>),
    Part(Object<'a>),
}

impl Horizontal<'_> {
    fn crs(&self) -> &Object<'_> {
        match self {
            Horizontal::Whole(crs) => crs,
            Horizontal::Part(crs) => crs,
        }
    }
}

/// A PROJ object, freed when dropped. It cannot outlive its context.
struct Object<'ctx>(*mut PJ, PhantomData<&'ctx Context>);

impl Drop for Object<'_> {
    fn drop(&mut self) {
        // SAFETY: owned, non-null, and its context is still alive.
        unsafe { proj_destroy(self.0) };
    }
}

#[cfg(test)]
mod tests {
    //! What the Python suite cannot see: which ellipsoid PROJ handed back.

    use super::*;

    #[test]
    fn a_geographic_crs_is_its_own_geodetic_crs() {
        let wgs84 = GeodeticCrs::of("EPSG:4326").unwrap();

        assert_eq!(wgs84.semi_major, 6_378_137.0);
        assert!((wgs84.flattening - 1.0 / 298.257_223_563).abs() < 1e-15);
    }

    #[test]
    fn a_projected_crs_is_measured_on_the_ellipsoid_it_projects_from() {
        // The Dutch national grid lies on Bessel 1841, not WGS 84.
        let rd = GeodeticCrs::of("EPSG:28992").unwrap();

        assert_eq!(rd.semi_major, 6_377_397.155);
        assert!((rd.flattening - 1.0 / 299.152_812_8).abs() < 1e-12);
        assert!(rd.definition.contains("Amersfoort"), "{}", rd.definition);
    }

    #[test]
    fn the_ellipsoid_is_in_the_unit_of_x_and_y() {
        // GRS 1980, which NAD83 lies on, in US survey feet.
        let grs80_in_feet = 6_378_137.0 * 3937.0 / 1200.0;
        // New York Long Island, in US survey feet.
        let ny = GeodeticCrs::of("EPSG:2263").unwrap();
        assert!(
            (ny.semi_major - grs80_in_feet).abs() < 1e-6,
            "{}",
            ny.semi_major
        );
        // The same, with NAVD88 heights in US survey feet on top.
        let compound = GeodeticCrs::of("EPSG:2263+6360").unwrap();
        assert!(
            (compound.semi_major - grs80_in_feet).abs() < 1e-6,
            "{}",
            compound.semi_major
        );
        // Its own longitude/latitude: degrees, so metres.
        assert_eq!(
            GeodeticCrs::of("EPSG:4269").unwrap().semi_major,
            6_378_137.0
        );
    }

    #[test]
    fn a_sphere_has_no_flattening() {
        let sphere = GeodeticCrs::of("+proj=longlat +R=1000 +type=crs").unwrap();

        assert_eq!((sphere.semi_major, sphere.flattening), (1000.0, 0.0));
    }

    #[test]
    fn a_turn_is_counted_in_the_unit_of_the_longitude() {
        assert_eq!(longitude_turn("EPSG:4326").unwrap(), Some(360.0));
        // NTF (Paris), in grads.
        assert_eq!(longitude_turn("EPSG:4807").unwrap(), Some(400.0));
        // WGS 84 with EGM96 heights on top: still longitude/latitude.
        assert_eq!(longitude_turn("EPSG:4326+5773").unwrap(), Some(360.0));
    }

    #[test]
    fn only_a_longitude_has_a_turn() {
        assert_eq!(longitude_turn("EPSG:28992").unwrap(), None);
        // Earth-centred XYZ.
        assert_eq!(longitude_turn("EPSG:4978").unwrap(), None);
        assert!(longitude_turn("EPSG:not-a-code").is_err());
    }

    #[test]
    fn refuses_what_is_not_on_longitude_latitude() {
        // Earth-centred XYZ: geodetic, but not geographic.
        assert!(GeodeticCrs::of("EPSG:4978").is_err());
        assert!(GeodeticCrs::of("EPSG:not-a-code").is_err());
    }
}
