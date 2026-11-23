//! PDE support — trigonometric transforms (DCT / DST) used in
//! spectral PDE solvers and signal processing.
//!
//! oneMKL exposes six trig-transform variants (sine, cosine, plus
//! four staggered combinations). The flow is:
//!
//! 1. [`TrigTransform::new`] — initialize and commit a plan for a
//!    given `n`-point transform.
//! 2. [`TrigTransform::forward`] / [`backward`](TrigTransform::backward)
//!    — run the transform on an in-place buffer.
//! 3. `Drop` releases the descriptor.
//!
//! Both single (`f32`) and double (`f64`) precision are supported via
//! the [`TrigTransformScalar`] trait. The transform is applied
//! in-place to the user-supplied buffer.
//!
//! ```ignore
//! use onemkl::pde::{TrigTransform, TrigTransformType};
//!
//! // 8-point DCT.
//! let mut f = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0];
//! //              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//! //              n+1 = 9 elements (sine / cosine transforms work on
//! //              n+1-point buffers — the boundary samples).
//! let mut plan = TrigTransform::<f64>::new(TrigTransformType::Cosine, 8).unwrap();
//! plan.forward(&mut f).unwrap();
//! ```

use core::ffi::c_int;
use core::marker::PhantomData;
use core::ptr;

use onemkl_sys::{self as sys, DFTI_DESCRIPTOR_HANDLE};

use crate::error::{Error, Result};

/// Variant of the trigonometric transform. Each maps to one of
/// MKL's `MKL_*_TRANSFORM` constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrigTransformType {
    /// Sine transform (DST-I).
    Sine,
    /// Cosine transform (DCT-I).
    Cosine,
    /// Staggered cosine transform (DCT-II / DCT-III pair).
    StaggeredCosine,
    /// Staggered sine transform (DST-II / DST-III pair).
    StaggeredSine,
    /// Twice-staggered cosine transform.
    Staggered2Cosine,
    /// Twice-staggered sine transform.
    Staggered2Sine,
}

impl TrigTransformType {
    #[inline]
    fn as_int(self) -> c_int {
        let v = match self {
            Self::Sine => sys::MKL_SINE_TRANSFORM,
            Self::Cosine => sys::MKL_COSINE_TRANSFORM,
            Self::StaggeredCosine => sys::MKL_STAGGERED_COSINE_TRANSFORM,
            Self::StaggeredSine => sys::MKL_STAGGERED_SINE_TRANSFORM,
            Self::Staggered2Cosine => sys::MKL_STAGGERED2_COSINE_TRANSFORM,
            Self::Staggered2Sine => sys::MKL_STAGGERED2_SINE_TRANSFORM,
        };
        v as c_int
    }

    /// Number of input samples expected for an `n`-point transform of
    /// this type. Sine and cosine work on `n + 1` boundary samples;
    /// the staggered variants work on `n` interior samples.
    #[inline]
    #[must_use]
    pub fn buffer_len(self, n: usize) -> usize {
        match self {
            Self::Sine | Self::Cosine => n + 1,
            _ => n,
        }
    }
}

/// Scalar types supported by the trig transform routines.
#[allow(missing_docs)]
pub trait TrigTransformScalar: Copy + Default + 'static {
    unsafe fn init_trig_transform(
        n: *mut c_int,
        tt_type: *mut c_int,
        ipar: *mut c_int,
        dpar: *mut Self,
        stat: *mut c_int,
    );
    unsafe fn commit_trig_transform(
        f: *mut Self,
        handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int,
        dpar: *mut Self,
        stat: *mut c_int,
    );
    unsafe fn forward_trig_transform(
        f: *mut Self,
        handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int,
        dpar: *mut Self,
        stat: *mut c_int,
    );
    unsafe fn backward_trig_transform(
        f: *mut Self,
        handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int,
        dpar: *mut Self,
        stat: *mut c_int,
    );
}

impl TrigTransformScalar for f32 {
    unsafe fn init_trig_transform(
        n: *mut c_int, tt_type: *mut c_int,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    ) {
        unsafe { sys::s_init_trig_transform(n, tt_type, ipar, dpar, stat) }
    }
    unsafe fn commit_trig_transform(
        f: *mut Self, handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    ) {
        unsafe { sys::s_commit_trig_transform(f, handle, ipar, dpar, stat) }
    }
    unsafe fn forward_trig_transform(
        f: *mut Self, handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    ) {
        unsafe { sys::s_forward_trig_transform(f, handle, ipar, dpar, stat) }
    }
    unsafe fn backward_trig_transform(
        f: *mut Self, handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    ) {
        unsafe { sys::s_backward_trig_transform(f, handle, ipar, dpar, stat) }
    }
}

impl TrigTransformScalar for f64 {
    unsafe fn init_trig_transform(
        n: *mut c_int, tt_type: *mut c_int,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    ) {
        unsafe { sys::d_init_trig_transform(n, tt_type, ipar, dpar, stat) }
    }
    unsafe fn commit_trig_transform(
        f: *mut Self, handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    ) {
        unsafe { sys::d_commit_trig_transform(f, handle, ipar, dpar, stat) }
    }
    unsafe fn forward_trig_transform(
        f: *mut Self, handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    ) {
        unsafe { sys::d_forward_trig_transform(f, handle, ipar, dpar, stat) }
    }
    unsafe fn backward_trig_transform(
        f: *mut Self, handle: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    ) {
        unsafe { sys::d_backward_trig_transform(f, handle, ipar, dpar, stat) }
    }
}

#[inline]
fn check_tr(stat: c_int) -> Result<()> {
    // The trig transform routines return 0 on success (standard
    // convention), not TR_SUCCESS = 1501 (which is the optimization
    // routines' convention).
    if stat == 0 {
        Ok(())
    } else {
        Err(Error::LapackComputationFailure { info: stat })
    }
}

/// An owned trigonometric transform plan.
pub struct TrigTransform<T: TrigTransformScalar> {
    handle: DFTI_DESCRIPTOR_HANDLE,
    ipar: Vec<c_int>,
    dpar: Vec<T>,
    n: c_int,
    transform_type: TrigTransformType,
    _marker: PhantomData<T>,
}

unsafe impl<T: TrigTransformScalar + Send> Send for TrigTransform<T> {}

impl<T: TrigTransformScalar> TrigTransform<T> {
    /// Build a plan for an `n`-point transform of the given type.
    /// Sine and cosine transforms operate on `n + 1`-element buffers
    /// (boundary samples included); the staggered variants operate
    /// on `n`-element buffers (interior samples). See
    /// [`TrigTransformType::buffer_len`].
    pub fn new(transform_type: TrigTransformType, n: usize) -> Result<Self> {
        if n == 0 {
            return Err(Error::InvalidArgument("n must be ≥ 1"));
        }
        let mut n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
        let mut tt_int: c_int = transform_type.as_int();
        // Allocate ipar of size 128 and dpar generously sized to fit
        // the largest variant's workspace.
        let mut ipar: Vec<c_int> = vec![0; 128];
        let mut dpar: Vec<T> = vec![T::default(); 5 * n + 2];
        let mut stat: c_int = 0;

        unsafe {
            T::init_trig_transform(
                &mut n_i,
                &mut tt_int,
                ipar.as_mut_ptr(),
                dpar.as_mut_ptr(),
                &mut stat,
            );
        }
        check_tr(stat)?;

        let mut handle: DFTI_DESCRIPTOR_HANDLE = ptr::null_mut();
        // commit_trig_transform takes f as input; it inspects the
        // first element to determine alignment / size. We can pass a
        // throwaway buffer here.
        let mut probe: Vec<T> = vec![T::default(); transform_type.buffer_len(n)];
        unsafe {
            T::commit_trig_transform(
                probe.as_mut_ptr(),
                &mut handle,
                ipar.as_mut_ptr(),
                dpar.as_mut_ptr(),
                &mut stat,
            );
        }
        check_tr(stat)?;

        Ok(Self {
            handle,
            ipar,
            dpar,
            n: n_i,
            transform_type,
            _marker: PhantomData,
        })
    }

    /// Length of the buffer this plan expects on each transform call.
    #[inline]
    #[must_use]
    pub fn buffer_len(&self) -> usize {
        self.transform_type.buffer_len(self.n as usize)
    }

    /// Apply the forward transform in place. `f` must have at least
    /// [`buffer_len`](Self::buffer_len) elements.
    pub fn forward(&mut self, f: &mut [T]) -> Result<()> {
        let need = self.buffer_len();
        if f.len() < need {
            return Err(Error::InvalidArgument(
                "buffer too small for the trig transform",
            ));
        }
        let mut stat: c_int = 0;
        unsafe {
            T::forward_trig_transform(
                f.as_mut_ptr(),
                &mut self.handle,
                self.ipar.as_mut_ptr(),
                self.dpar.as_mut_ptr(),
                &mut stat,
            );
        }
        check_tr(stat)
    }

    /// Apply the backward / inverse transform in place.
    pub fn backward(&mut self, f: &mut [T]) -> Result<()> {
        let need = self.buffer_len();
        if f.len() < need {
            return Err(Error::InvalidArgument(
                "buffer too small for the trig transform",
            ));
        }
        let mut stat: c_int = 0;
        unsafe {
            T::backward_trig_transform(
                f.as_mut_ptr(),
                &mut self.handle,
                self.ipar.as_mut_ptr(),
                self.dpar.as_mut_ptr(),
                &mut stat,
            );
        }
        check_tr(stat)
    }
}

impl<T: TrigTransformScalar> Drop for TrigTransform<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let mut stat: c_int = 0;
            unsafe {
                sys::free_trig_transform(
                    &mut self.handle,
                    self.ipar.as_mut_ptr(),
                    &mut stat,
                );
            }
        }
    }
}

// ---------------------------------------------------------------
// Cartesian Helmholtz / Poisson solver (2-D)
// ---------------------------------------------------------------

/// Boundary condition on one side of the rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Boundary {
    /// Dirichlet: `u` is prescribed (the boundary values are supplied
    /// to [`Helmholtz2D::solve`]).
    Dirichlet,
    /// Neumann: the outward-pointing-axis derivative `du/dn` is
    /// prescribed (values supplied to [`Helmholtz2D::solve`]).
    Neumann,
    /// Periodic: the side is identified with the opposite one; both
    /// sides of the pair must be `Periodic`.
    Periodic,
}

impl Boundary {
    fn as_char(self) -> u8 {
        match self {
            Self::Dirichlet => b'D',
            Self::Neumann => b'N',
            Self::Periodic => b'P',
        }
    }
}

/// Scalar types supported by the Helmholtz solver.
#[allow(missing_docs, clippy::missing_safety_doc, clippy::too_many_arguments)]
pub trait HelmholtzScalar: Copy + Default + 'static {
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
    unsafe fn init_2d(
        ax: *const Self, bx: *const Self, ay: *const Self, by: *const Self,
        nx: *const c_int, ny: *const c_int, bc: *const core::ffi::c_char,
        q: *const Self, ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    );
    unsafe fn commit_2d(
        f: *mut Self, bd_ax: *const Self, bd_bx: *const Self,
        bd_ay: *const Self, bd_by: *const Self,
        handle: *mut DFTI_DESCRIPTOR_HANDLE, ipar: *mut c_int,
        dpar: *mut Self, stat: *mut c_int,
    );
    unsafe fn solve_2d(
        f: *mut Self, bd_ax: *const Self, bd_bx: *const Self,
        bd_ay: *const Self, bd_by: *const Self,
        handle: *mut DFTI_DESCRIPTOR_HANDLE, ipar: *mut c_int,
        dpar: *const Self, stat: *mut c_int,
    );
    unsafe fn init_3d(
        lim: [*const Self; 6], n: [*const c_int; 3], bc: *const core::ffi::c_char,
        q: *const Self, ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    );
    unsafe fn commit_3d(
        f: *mut Self, bd: [*const Self; 6],
        xh: *mut DFTI_DESCRIPTOR_HANDLE, yh: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    );
    unsafe fn solve_3d(
        f: *mut Self, bd: [*const Self; 6],
        xh: *mut DFTI_DESCRIPTOR_HANDLE, yh: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *const Self, stat: *mut c_int,
    );
    unsafe fn init_sph(
        lim: [*const Self; 4], np: *const c_int, nt: *const c_int, q: *const Self,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    );
    unsafe fn commit_sph(
        f: *mut Self, h1: *mut DFTI_DESCRIPTOR_HANDLE, h2: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    );
    unsafe fn solve_sph(
        f: *mut Self, h1: *mut DFTI_DESCRIPTOR_HANDLE, h2: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    );
    unsafe fn init_sph_np(
        lim: [*const Self; 4], np: *const c_int, nt: *const c_int, q: *const Self,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    );
    unsafe fn commit_sph_np(
        f: *mut Self, h: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    );
    unsafe fn solve_sph_np(
        f: *mut Self, h: *mut DFTI_DESCRIPTOR_HANDLE,
        ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
    );
    unsafe fn free_sph_np(h: *mut DFTI_DESCRIPTOR_HANDLE, ipar: *mut c_int, stat: *mut c_int);
}

macro_rules! impl_helmholtz_scalar {
    ($t:ty, $init:ident, $commit:ident, $solve:ident, $init3:ident, $commit3:ident, $solve3:ident, $inits:ident, $commits:ident, $solves:ident, $initn:ident, $commitn:ident, $solven:ident) => {
        impl HelmholtzScalar for $t {
            fn to_f64(self) -> f64 { self as f64 }
            fn from_f64(v: f64) -> Self { v as $t }
            unsafe fn init_2d(
                ax: *const Self, bx: *const Self, ay: *const Self, by: *const Self,
                nx: *const c_int, ny: *const c_int, bc: *const core::ffi::c_char,
                q: *const Self, ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
            ) {
                unsafe { sys::$init(ax, bx, ay, by, nx, ny, bc, q, ipar, dpar, stat) }
            }
            unsafe fn commit_2d(
                f: *mut Self, bd_ax: *const Self, bd_bx: *const Self,
                bd_ay: *const Self, bd_by: *const Self,
                handle: *mut DFTI_DESCRIPTOR_HANDLE, ipar: *mut c_int,
                dpar: *mut Self, stat: *mut c_int,
            ) {
                unsafe { sys::$commit(f, bd_ax, bd_bx, bd_ay, bd_by, handle, ipar, dpar, stat) }
            }
            unsafe fn solve_2d(
                f: *mut Self, bd_ax: *const Self, bd_bx: *const Self,
                bd_ay: *const Self, bd_by: *const Self,
                handle: *mut DFTI_DESCRIPTOR_HANDLE, ipar: *mut c_int,
                dpar: *const Self, stat: *mut c_int,
            ) {
                unsafe { sys::$solve(f, bd_ax, bd_bx, bd_ay, bd_by, handle, ipar, dpar, stat) }
            }
            unsafe fn init_3d(
                l: [*const Self; 6], n: [*const c_int; 3], bc: *const core::ffi::c_char,
                q: *const Self, ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
            ) {
                unsafe {
                    sys::$init3(l[0], l[1], l[2], l[3], l[4], l[5], n[0], n[1], n[2], bc, q, ipar, dpar, stat)
                }
            }
            unsafe fn commit_3d(
                f: *mut Self, b: [*const Self; 6],
                xh: *mut DFTI_DESCRIPTOR_HANDLE, yh: *mut DFTI_DESCRIPTOR_HANDLE,
                ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
            ) {
                unsafe {
                    sys::$commit3(f, b[0], b[1], b[2], b[3], b[4], b[5], xh, yh, ipar, dpar, stat)
                }
            }
            unsafe fn solve_3d(
                f: *mut Self, b: [*const Self; 6],
                xh: *mut DFTI_DESCRIPTOR_HANDLE, yh: *mut DFTI_DESCRIPTOR_HANDLE,
                ipar: *mut c_int, dpar: *const Self, stat: *mut c_int,
            ) {
                unsafe {
                    sys::$solve3(f, b[0], b[1], b[2], b[3], b[4], b[5], xh, yh, ipar, dpar, stat)
                }
            }
            unsafe fn init_sph(
                l: [*const Self; 4], np: *const c_int, nt: *const c_int, q: *const Self,
                ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
            ) {
                unsafe { sys::$inits(l[0], l[1], l[2], l[3], np, nt, q, ipar, dpar, stat) }
            }
            unsafe fn commit_sph(
                f: *mut Self, h1: *mut DFTI_DESCRIPTOR_HANDLE, h2: *mut DFTI_DESCRIPTOR_HANDLE,
                ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
            ) {
                unsafe { sys::$commits(f, h1, h2, ipar, dpar, stat) }
            }
            unsafe fn solve_sph(
                f: *mut Self, h1: *mut DFTI_DESCRIPTOR_HANDLE, h2: *mut DFTI_DESCRIPTOR_HANDLE,
                ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
            ) {
                unsafe { sys::$solves(f, h1, h2, ipar, dpar, stat) }
            }
            unsafe fn init_sph_np(
                l: [*const Self; 4], np: *const c_int, nt: *const c_int, q: *const Self,
                ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
            ) {
                unsafe { sys::$initn(l[0], l[1], l[2], l[3], np, nt, q, ipar, dpar, stat) }
            }
            unsafe fn commit_sph_np(
                f: *mut Self, h: *mut DFTI_DESCRIPTOR_HANDLE,
                ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
            ) {
                unsafe { sys::$commitn(f, h, ipar, dpar, stat) }
            }
            unsafe fn solve_sph_np(
                f: *mut Self, h: *mut DFTI_DESCRIPTOR_HANDLE,
                ipar: *mut c_int, dpar: *mut Self, stat: *mut c_int,
            ) {
                unsafe { sys::$solven(f, h, ipar, dpar, stat) }
            }
            unsafe fn free_sph_np(h: *mut DFTI_DESCRIPTOR_HANDLE, ipar: *mut c_int, stat: *mut c_int) {
                unsafe { sys::free_sph_np(h, ipar, stat) }
            }
        }
    };
}
impl_helmholtz_scalar!(f32, s_init_helmholtz_2d, s_commit_helmholtz_2d, s_helmholtz_2d, s_init_helmholtz_3d, s_commit_helmholtz_3d, s_helmholtz_3d, s_init_sph_p, s_commit_sph_p, s_sph_p, s_init_sph_np, s_commit_sph_np, s_sph_np);
impl_helmholtz_scalar!(f64, d_init_helmholtz_2d, d_commit_helmholtz_2d, d_helmholtz_2d, d_init_helmholtz_3d, d_commit_helmholtz_3d, d_helmholtz_3d, d_init_sph_p, d_commit_sph_p, d_sph_p, d_init_sph_np, d_commit_sph_np, d_sph_np);

/// Index range of the unknown nodes along one axis of `n` intervals
/// for the boundary pair `(lo, hi)`: Dirichlet nodes are fixed,
/// Neumann boundary nodes are unknowns, a periodic axis has `n` unknowns
/// (node `n` repeats node 0).
fn unknown_range(lo: Boundary, hi: Boundary, n: usize) -> (usize, usize) {
    if lo == Boundary::Periodic {
        return (0, n - 1);
    }
    (
        if lo == Boundary::Dirichlet { 1 } else { 0 },
        if hi == Boundary::Dirichlet { n - 1 } else { n },
    )
}

/// Right-hand-side adjustment for one axis at node index `i`:
/// returns `(subtract, halvings)` where `subtract` is the amount to
/// subtract from the rhs (Dirichlet lift `g/h²`, Neumann `2g/h`) and
/// `halvings` is 1 on a Neumann boundary node (MKL weights those rows
/// by 2, measured), else 0.
fn axis_lift<T: HelmholtzScalar>(
    bc: [Boundary; 2],
    n: usize,
    i: usize,
    h: f64,
    g_lo: impl Fn() -> T,
    g_hi: impl Fn() -> T,
) -> (f64, u32) {
    let mut sub = 0.0;
    let mut halv = 0;
    if bc[0] == Boundary::Dirichlet && i == 1 {
        sub += g_lo().to_f64() / (h * h);
    }
    if bc[1] == Boundary::Dirichlet && i == n - 1 {
        sub += g_hi().to_f64() / (h * h);
    }
    if bc[0] == Boundary::Neumann && i == 0 {
        sub += 2.0 * g_lo().to_f64() / h;
        halv += 1;
    }
    if bc[1] == Boundary::Neumann && i == n {
        sub += 2.0 * g_hi().to_f64() / h;
        halv += 1;
    }
    (sub, halv)
}

fn check_pairs(bc: &[Boundary]) -> Result<()> {
    for pair in bc.chunks(2) {
        if (pair[0] == Boundary::Periodic) != (pair[1] == Boundary::Periodic) {
            return Err(Error::InvalidArgument(
                "periodic boundary must be set on both sides of the pair",
            ));
        }
    }
    Ok(())
}

/// Fast solver for the 2-D Helmholtz equation `Δu − q·u = f` on the
/// rectangle `[ax, bx] × [ay, by]` (`q = 0` gives Poisson), on a
/// uniform `nx × ny` grid of intervals (second-order finite
/// differences, FFT-based).
///
/// Grid values are stored column-by-x: node `(i, j)` lives at
/// `f[i + j * (nx + 1)]`, `0 ≤ i ≤ nx`, `0 ≤ j ≤ ny`.
/// [`solve`](Self::solve) takes the right-hand side in `f` and
/// overwrites it with the solution (Dirichlet boundary nodes are set
/// to the supplied boundary data).
///
/// Boundary data: Dirichlet sides take the value of `u`, Neumann sides
/// the outward normal derivative `∂u/∂n`, periodic sides nothing.
/// Neumann boundary nodes are unknowns, discretised with a reflected
/// ghost node. All-Neumann/periodic with `q = 0` is singular and MKL
/// reports it as an error.
///
/// **Measured MKL behaviour** (oneMKL 2026.2.1, Linux x86_64, LP64; fitted
/// to 10 digits against a dense reference): MKL returns
/// `(−Δ + q)⁻¹ f / hy²` instead of the documented `(Δ − q)⁻¹ f`, weights
/// Neumann boundary rows by 2, and ignores the boundary-data arrays, so
/// this wrapper moves the boundary data to the right-hand side itself and
/// rescales the result.
pub struct Helmholtz2D<T: HelmholtzScalar> {
    handle: DFTI_DESCRIPTOR_HANDLE,
    ipar: Vec<c_int>,
    dpar: Vec<T>,
    nx: usize,
    ny: usize,
    bc: [Boundary; 4],
    hx: f64,
    hy: f64,
    committed: bool,
}

unsafe impl<T: HelmholtzScalar + Send> Send for Helmholtz2D<T> {}

impl<T: HelmholtzScalar> Helmholtz2D<T> {
    /// Build a solver. `bc` is ordered `[x = ax, x = bx, y = ay, y = by]`;
    /// a periodic pair must be periodic on both sides. `nx` and `ny`
    /// must be even and at least 4 (MKL requirement). `q ≥ 0` is the
    /// Helmholtz coefficient.
    pub fn new(
        (ax, bx): (T, T),
        (ay, by): (T, T),
        nx: usize,
        ny: usize,
        bc: [Boundary; 4],
        q: T,
    ) -> Result<Self> {
        if nx < 4 || ny < 4 || !nx.is_multiple_of(2) || !ny.is_multiple_of(2) {
            return Err(Error::InvalidArgument(
                "nx and ny must be even and at least 4",
            ));
        }
        check_pairs(&bc)?;
        let nx_i: c_int = nx.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ny_i: c_int = ny.try_into().map_err(|_| Error::DimensionOverflow)?;
        let bc_c: [core::ffi::c_char; 5] = [
            bc[0].as_char() as _,
            bc[1].as_char() as _,
            bc[2].as_char() as _,
            bc[3].as_char() as _,
            0,
        ];
        let mut ipar: Vec<c_int> = vec![0; 128];
        // MKL documents 5*nx/2 + 7*ny/2 + 9 for dpar; pad generously.
        let mut dpar: Vec<T> = vec![T::default(); 5 * nx / 2 + 7 * ny / 2 + 64];
        let mut stat: c_int = 0;
        unsafe {
            T::init_2d(
                &ax, &bx, &ay, &by, &nx_i, &ny_i, bc_c.as_ptr(), &q,
                ipar.as_mut_ptr(), dpar.as_mut_ptr(), &mut stat,
            );
        }
        check_tr(stat)?;
        Ok(Self {
            handle: ptr::null_mut(),
            ipar,
            dpar,
            nx,
            ny,
            bc,
            hx: (bx.to_f64() - ax.to_f64()) / nx as f64,
            hy: (by.to_f64() - ay.to_f64()) / ny as f64,
            committed: false,
        })
    }

    /// Required length of the grid buffer, `(nx + 1) * (ny + 1)`.
    #[must_use]
    pub fn grid_len(&self) -> usize {
        (self.nx + 1) * (self.ny + 1)
    }

    /// Solve in place. `f` holds the right-hand side on entry and the
    /// solution on exit. `bd_ax`/`bd_bx` (length `ny + 1`) and
    /// `bd_ay`/`bd_by` (length `nx + 1`) carry the Dirichlet values or
    /// outward Neumann derivatives of the corresponding side; for a
    /// periodic side pass an empty slice (ignored).
    pub fn solve(
        &mut self,
        f: &mut [T],
        bd_ax: &[T],
        bd_bx: &[T],
        bd_ay: &[T],
        bd_by: &[T],
    ) -> Result<()> {
        if f.len() < self.grid_len() {
            return Err(Error::InvalidArgument("grid buffer too small"));
        }
        let (nx, ny) = (self.nx, self.ny);
        let w = nx + 1;
        let bd = [bd_ax, bd_bx, bd_ay, bd_by];
        let need = [ny + 1, ny + 1, nx + 1, nx + 1];
        for k in 0..4 {
            if self.bc[k] != Boundary::Periodic && bd[k].len() < need[k] {
                return Err(Error::InvalidArgument("boundary data slice too short"));
            }
        }
        let g = |k: usize, idx: usize| move || if bd[k].is_empty() { T::default() } else { bd[k][idx] };
        let (ilo, ihi) = unknown_range(self.bc[0], self.bc[1], nx);
        let (jlo, jhi) = unknown_range(self.bc[2], self.bc[3], ny);
        // MKL applies no boundary lift: do it here (see the type docs).
        for j in jlo..=jhi {
            for i in ilo..=ihi {
                let (sx, hx_) = axis_lift([self.bc[0], self.bc[1]], nx, i, self.hx, g(0, j), g(1, j));
                let (sy, hy_) = axis_lift([self.bc[2], self.bc[3]], ny, j, self.hy, g(2, i), g(3, i));
                let v = &mut f[i + j * w];
                let r = v.to_f64() - sx - sy;
                *v = T::from_f64(r / f64::from(1u32 << (hx_ + hy_)));
            }
        }
        // A periodic axis repeats its first node in the last one.
        if self.bc[0] == Boundary::Periodic {
            for j in 0..=ny { f[nx + j * w] = f[j * w]; }
        }
        if self.bc[2] == Boundary::Periodic {
            for i in 0..=nx { f[i + ny * w] = f[i]; }
        }
        let zeros: Vec<T> = vec![T::default(); nx.max(ny) + 1];
        let zp = zeros.as_ptr();
        let mut stat: c_int = 0;
        if !self.committed {
            let mut probe: Vec<T> = f[..self.grid_len()].to_vec();
            unsafe {
                T::commit_2d(
                    probe.as_mut_ptr(), zp, zp, zp, zp,
                    &mut self.handle, self.ipar.as_mut_ptr(),
                    self.dpar.as_mut_ptr(), &mut stat,
                );
            }
            check_tr(stat)?;
            self.committed = true;
        }
        unsafe {
            T::solve_2d(
                f.as_mut_ptr(), zp, zp, zp, zp,
                &mut self.handle, self.ipar.as_mut_ptr(),
                self.dpar.as_ptr(), &mut stat,
            );
        }
        check_tr(stat)?;
        // Undo MKL's sign and 1/hy² scaling; Dirichlet nodes get the data.
        let hy2 = self.hy * self.hy;
        for j in 0..=ny {
            for i in 0..=nx {
                let d = if i == 0 && self.bc[0] == Boundary::Dirichlet { Some(bd[0][j]) }
                    else if i == nx && self.bc[1] == Boundary::Dirichlet { Some(bd[1][j]) }
                    else if j == 0 && self.bc[2] == Boundary::Dirichlet { Some(bd[2][i]) }
                    else if j == ny && self.bc[3] == Boundary::Dirichlet { Some(bd[3][i]) }
                    else { None };
                let v = &mut f[i + j * w];
                *v = match d {
                    Some(d) => d,
                    None => T::from_f64(-hy2 * v.to_f64()),
                };
            }
        }
        Ok(())
    }
}

impl<T: HelmholtzScalar> Drop for Helmholtz2D<T> {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        let mut stat: c_int = 0;
        unsafe {
            sys::free_helmholtz_2d(&mut self.handle, self.ipar.as_mut_ptr(), &mut stat);
        }
    }
}

/// Fast solver for the 3-D Helmholtz equation `Δu − q·u = f` on a box
/// with a uniform `nx × ny × nz` grid. Boundary types, data meaning and
/// the measured MKL behaviour the wrapper corrects are those of
/// [`Helmholtz2D`] (the 3-D result is `(−Δ + q)⁻¹ f / hz²`).
///
/// Node `(i, j, k)` is `f[i + j*(nx+1) + k*(nx+1)*(ny+1)]`. Boundary
/// data per face, in the order `[ax, bx, ay, by, az, bz]`: x faces are
/// `(ny+1)*(nz+1)` long (`j + k*(ny+1)`), y faces `(nx+1)*(nz+1)`
/// (`i + k*(nx+1)`), z faces `(nx+1)*(ny+1)` (`i + j*(nx+1)`); periodic
/// faces take empty slices.
pub struct Helmholtz3D<T: HelmholtzScalar> {
    xh: DFTI_DESCRIPTOR_HANDLE,
    yh: DFTI_DESCRIPTOR_HANDLE,
    ipar: Vec<c_int>,
    dpar: Vec<T>,
    n: [usize; 3],
    h: [f64; 3],
    bc: [Boundary; 6],
    committed: bool,
}

unsafe impl<T: HelmholtzScalar + Send> Send for Helmholtz3D<T> {}

impl<T: HelmholtzScalar> Helmholtz3D<T> {
    /// Build a solver; `nx`, `ny`, `nz` must be even and at least 4.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        (ax, bx): (T, T),
        (ay, by): (T, T),
        (az, bz): (T, T),
        nx: usize,
        ny: usize,
        nz: usize,
        bc: [Boundary; 6],
        q: T,
    ) -> Result<Self> {
        if [nx, ny, nz].iter().any(|&n| n < 4 || !n.is_multiple_of(2)) {
            return Err(Error::InvalidArgument("nx, ny, nz must be even and at least 4"));
        }
        check_pairs(&bc)?;
        let nx_i: c_int = nx.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ny_i: c_int = ny.try_into().map_err(|_| Error::DimensionOverflow)?;
        let nz_i: c_int = nz.try_into().map_err(|_| Error::DimensionOverflow)?;
        let bc_c: [core::ffi::c_char; 7] = [
            bc[0].as_char() as _, bc[1].as_char() as _, bc[2].as_char() as _,
            bc[3].as_char() as _, bc[4].as_char() as _, bc[5].as_char() as _, 0,
        ];
        let mut ipar: Vec<c_int> = vec![0; 128];
        // MKL's 3-D dpar size is quadratic in the grid; cover the largest
        // face product generously (not MKL's documented formula).
        let face = ((nx + 1) * (ny + 1)).max((ny + 1) * (nz + 1)).max((nx + 1) * (nz + 1));
        let mut dpar: Vec<T> = vec![T::default(); 7 * face + 40 * (nx + ny + nz) + 200];
        let mut stat: c_int = 0;
        unsafe {
            T::init_3d(
                [&ax, &bx, &ay, &by, &az, &bz],
                [&nx_i, &ny_i, &nz_i],
                bc_c.as_ptr(), &q, ipar.as_mut_ptr(), dpar.as_mut_ptr(), &mut stat,
            );
        }
        check_tr(stat)?;
        let step = |a: T, b: T, n: usize| (b.to_f64() - a.to_f64()) / n as f64;
        Ok(Self {
            xh: ptr::null_mut(),
            yh: ptr::null_mut(),
            ipar,
            dpar,
            n: [nx, ny, nz],
            h: [step(ax, bx, nx), step(ay, by, ny), step(az, bz, nz)],
            bc,
            committed: false,
        })
    }

    /// Required grid length, `(nx + 1) * (ny + 1) * (nz + 1)`.
    #[must_use]
    pub fn grid_len(&self) -> usize {
        (self.n[0] + 1) * (self.n[1] + 1) * (self.n[2] + 1)
    }

    /// Solve in place; `bd` holds the boundary data of the faces
    /// `[ax, bx, ay, by, az, bz]` (layouts in the type docs).
    pub fn solve(&mut self, f: &mut [T], bd: [&[T]; 6]) -> Result<()> {
        let [nx, ny, nz] = self.n;
        let (wx, wy) = (nx + 1, ny + 1);
        if f.len() < self.grid_len() {
            return Err(Error::InvalidArgument("grid buffer too small"));
        }
        let need = [wy * (nz + 1), wy * (nz + 1), wx * (nz + 1), wx * (nz + 1), wx * wy, wx * wy];
        for k in 0..6 {
            if self.bc[k] != Boundary::Periodic && bd[k].len() < need[k] {
                return Err(Error::InvalidArgument("boundary data slice too short"));
            }
        }
        let at = |i: usize, j: usize, k: usize| i + j * wx + k * wx * wy;
        let g = |f_: usize, idx: usize| move || if bd[f_].is_empty() { T::default() } else { bd[f_][idx] };
        let (ilo, ihi) = unknown_range(self.bc[0], self.bc[1], nx);
        let (jlo, jhi) = unknown_range(self.bc[2], self.bc[3], ny);
        let (klo, khi) = unknown_range(self.bc[4], self.bc[5], nz);
        for k in klo..=khi {
            for j in jlo..=jhi {
                for i in ilo..=ihi {
                    let (sx, hx_) = axis_lift([self.bc[0], self.bc[1]], nx, i, self.h[0], g(0, j + k * wy), g(1, j + k * wy));
                    let (sy, hy_) = axis_lift([self.bc[2], self.bc[3]], ny, j, self.h[1], g(2, i + k * wx), g(3, i + k * wx));
                    let (sz, hz_) = axis_lift([self.bc[4], self.bc[5]], nz, k, self.h[2], g(4, i + j * wx), g(5, i + j * wx));
                    let v = &mut f[at(i, j, k)];
                    let r = v.to_f64() - sx - sy - sz;
                    *v = T::from_f64(r / f64::from(1u32 << (hx_ + hy_ + hz_)));
                }
            }
        }
        // A periodic axis repeats its first node in the last one.
        if self.bc[0] == Boundary::Periodic {
            for k in 0..=nz { for j in 0..=ny { f[at(nx, j, k)] = f[at(0, j, k)]; } }
        }
        if self.bc[2] == Boundary::Periodic {
            for k in 0..=nz { for i in 0..=nx { f[at(i, ny, k)] = f[at(i, 0, k)]; } }
        }
        if self.bc[4] == Boundary::Periodic {
            for j in 0..=ny { for i in 0..=nx { f[at(i, j, nz)] = f[at(i, j, 0)]; } }
        }
        let zeros: Vec<T> = vec![T::default(); wx.max(wy) * (nz + 1).max(wx.max(wy))];
        let zp = [zeros.as_ptr(); 6];
        let mut stat: c_int = 0;
        if !self.committed {
            let mut probe: Vec<T> = f[..self.grid_len()].to_vec();
            unsafe {
                T::commit_3d(
                    probe.as_mut_ptr(), zp, &mut self.xh, &mut self.yh,
                    self.ipar.as_mut_ptr(), self.dpar.as_mut_ptr(), &mut stat,
                );
            }
            check_tr(stat)?;
            self.committed = true;
        }
        unsafe {
            T::solve_3d(
                f.as_mut_ptr(), zp, &mut self.xh, &mut self.yh,
                self.ipar.as_mut_ptr(), self.dpar.as_ptr(), &mut stat,
            );
        }
        check_tr(stat)?;
        // Undo MKL's sign and 1/hz² scaling; Dirichlet nodes get the data.
        let hz2 = self.h[2] * self.h[2];
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    let d = if i == 0 && self.bc[0] == Boundary::Dirichlet { Some(bd[0][j + k * wy]) }
                        else if i == nx && self.bc[1] == Boundary::Dirichlet { Some(bd[1][j + k * wy]) }
                        else if j == 0 && self.bc[2] == Boundary::Dirichlet { Some(bd[2][i + k * wx]) }
                        else if j == ny && self.bc[3] == Boundary::Dirichlet { Some(bd[3][i + k * wx]) }
                        else if k == 0 && self.bc[4] == Boundary::Dirichlet { Some(bd[4][i + j * wx]) }
                        else if k == nz && self.bc[5] == Boundary::Dirichlet { Some(bd[5][i + j * wx]) }
                        else { None };
                    let v = &mut f[at(i, j, k)];
                    *v = match d {
                        Some(d) => d,
                        None => T::from_f64(-hz2 * v.to_f64()),
                    };
                }
            }
        }
        Ok(())
    }
}

impl<T: HelmholtzScalar> Drop for Helmholtz3D<T> {
    fn drop(&mut self) {
        if self.xh.is_null() && self.yh.is_null() {
            return;
        }
        let mut stat: c_int = 0;
        unsafe {
            sys::free_helmholtz_3d(&mut self.xh, &mut self.yh, self.ipar.as_mut_ptr(), &mut stat);
        }
    }
}

/// Fast Helmholtz solver `Δ_S u − q·u = F` on the unit sphere surface
/// (`φ ∈ [0, 2π]` periodic, `θ ∈ [0, π]` pole to pole), wrapping MKL's
/// periodic spherical solver (`?_sph_p`).
///
/// Node `(i, j)` (`φ = 2π·i/np`, `θ = π·j/nt`) is `f[i + j*(np+1)]`;
/// the nodes `i = np` repeat `i = 0`, and the pole rows `j = 0`, `j = nt`
/// should hold a single value per ring (the ring average is what the
/// solver uses). `q > 0` makes the problem well posed; with `q = 0` MKL
/// reports the constant null space as a warning, surfaced here as an error.
///
/// **Measured MKL behaviour** (oneMKL 2026.2.1, Linux x86_64, LP64): the
/// argument order is `(ap, bp, at, bt)`, and for each `φ`-mode MKL solves
/// `hθ²·sinθⱼ·(−Δ_h + q)u = f` with the conservative θ stencil built from
/// averaged node sines, and weight 1 (not `sinθ`) on the pole rows; the
/// wrapper maps `F` to `f = −hθ²·sinθⱼ·F` (weight 1 at the poles).
pub struct SphericalHelmholtz<T: HelmholtzScalar> {
    h1: DFTI_DESCRIPTOR_HANDLE,
    h2: DFTI_DESCRIPTOR_HANDLE,
    ipar: Vec<c_int>,
    dpar: Vec<T>,
    np: usize,
    nt: usize,
    committed: bool,
}

unsafe impl<T: HelmholtzScalar + Send> Send for SphericalHelmholtz<T> {}

impl<T: HelmholtzScalar> SphericalHelmholtz<T> {
    /// Build a solver for `np` intervals in `φ` (even, ≥ 4) and `nt`
    /// intervals in `θ` (≥ 2).
    pub fn new(np: usize, nt: usize, q: T) -> Result<Self> {
        if np < 4 || !np.is_multiple_of(2) || nt < 2 {
            return Err(Error::InvalidArgument("np must be even and ≥ 4, nt ≥ 2"));
        }
        let np_i: c_int = np.try_into().map_err(|_| Error::DimensionOverflow)?;
        let nt_i: c_int = nt.try_into().map_err(|_| Error::DimensionOverflow)?;
        let lim = [
            T::from_f64(0.0),
            T::from_f64(2.0 * core::f64::consts::PI),
            T::from_f64(0.0),
            T::from_f64(core::f64::consts::PI),
        ];
        let mut ipar: Vec<c_int> = vec![0; 128];
        // Documented size is 5*np/2 + nt + 10; pad generously.
        let mut dpar: Vec<T> = vec![T::default(); 4 * (5 * np / 2 + nt + 10) + 200];
        let mut stat: c_int = 0;
        unsafe {
            T::init_sph(
                [&lim[0], &lim[1], &lim[2], &lim[3]], &np_i, &nt_i, &q,
                ipar.as_mut_ptr(), dpar.as_mut_ptr(), &mut stat,
            );
        }
        check_tr(stat)?;
        Ok(Self {
            h1: ptr::null_mut(),
            h2: ptr::null_mut(),
            ipar,
            dpar,
            np,
            nt,
            committed: false,
        })
    }

    /// Required grid length, `(np + 1) * (nt + 1)`.
    #[must_use]
    pub fn grid_len(&self) -> usize {
        (self.np + 1) * (self.nt + 1)
    }

    /// Solve in place: `f` holds `F` on entry and `u` on exit.
    pub fn solve(&mut self, f: &mut [T]) -> Result<()> {
        if f.len() < self.grid_len() {
            return Err(Error::InvalidArgument("grid buffer too small"));
        }
        let (np, nt) = (self.np, self.nt);
        let w = np + 1;
        let ht = core::f64::consts::PI / nt as f64;
        for j in 0..=nt {
            let weight = if j == 0 || j == nt { 1.0 } else { (j as f64 * ht).sin() };
            for i in 0..=np {
                let v = &mut f[i + j * w];
                *v = T::from_f64(-ht * ht * weight * v.to_f64());
            }
        }
        let mut stat: c_int = 0;
        if !self.committed {
            let mut probe: Vec<T> = f[..self.grid_len()].to_vec();
            unsafe {
                T::commit_sph(
                    probe.as_mut_ptr(), &mut self.h1, &mut self.h2,
                    self.ipar.as_mut_ptr(), self.dpar.as_mut_ptr(), &mut stat,
                );
            }
            check_tr(stat)?;
            self.committed = true;
        }
        unsafe {
            T::solve_sph(
                f.as_mut_ptr(), &mut self.h1, &mut self.h2,
                self.ipar.as_mut_ptr(), self.dpar.as_mut_ptr(), &mut stat,
            );
        }
        check_tr(stat)
    }
}

impl<T: HelmholtzScalar> Drop for SphericalHelmholtz<T> {
    fn drop(&mut self) {
        if self.h1.is_null() && self.h2.is_null() {
            return;
        }
        let mut stat: c_int = 0;
        unsafe {
            sys::free_sph_p(&mut self.h1, &mut self.h2, self.ipar.as_mut_ptr(), &mut stat);
        }
    }
}

/// Helmholtz solver `Δ_S u − q·u = F` on a patch of the sphere surface,
/// `φ ∈ [ap, bp]`, `θ ∈ [at, bt]` with `0 < at < bt < π` (no pole), with
/// Dirichlet data on all four sides, wrapping MKL's non-periodic spherical
/// solver (`?_sph_np`).
///
/// Node `(i, j)` (`φ = ap + (bp−ap)·i/np`, `θ = at + (bt−at)·j/nt`) is
/// `f[i + j*(np+1)]`. On entry the interior nodes hold `F` and the boundary
/// nodes (`i = 0`, `i = np`, `j = 0`, `j = nt`) hold the Dirichlet data `g`;
/// on exit the interior holds `u` and the boundary is unchanged.
///
/// **Measured MKL behaviour** (oneMKL 2026.2.1, Linux x86_64, LP64): argument
/// order `(ap, bp, at, bt)`; the interior block (`1 ≤ i < np`, `1 ≤ j < nt`)
/// of the solver matrix is the inverse of
/// `hθ²·sinθⱼ·q + (sₚ + sₘ)` on the diagonal, `−sₚ`/`−sₘ` to the θ
/// neighbours (`sₚ`, `sₘ` the averaged node sines at `θⱼ ± hθ/2`) and
/// `2hθ²/(sinθⱼ·hφ²)` on the diagonal / `−hθ²/(sinθⱼ·hφ²)` to the φ
/// neighbours (agreement with a dense model 1e-15 on 2 domains at
/// np = 6, nt = 4). MKL treats the boundary rows and columns as decoupled
/// lines carrying no usable boundary condition (`j = 0, nt` pass `f`
/// through; `i = 0, np` are solved as θ-only problems), so the wrapper moves
/// the Dirichlet data into the right-hand side itself and never uses MKL's
/// boundary output. The mapping is `f = −hθ²·sinθⱼ·F` plus that lift.
pub struct SphericalPatchHelmholtz<T: HelmholtzScalar> {
    h: DFTI_DESCRIPTOR_HANDLE,
    ipar: Vec<c_int>,
    dpar: Vec<T>,
    np: usize,
    nt: usize,
    hp: f64,
    ht: f64,
    at: f64,
    committed: bool,
}

unsafe impl<T: HelmholtzScalar + Send> Send for SphericalPatchHelmholtz<T> {}

impl<T: HelmholtzScalar> SphericalPatchHelmholtz<T> {
    /// Build a solver for the patch `[ap, bp] × [at, bt]` with `np` × `nt`
    /// intervals (`np ≥ 2`, `nt ≥ 2`, `0 < at < bt < π`, `ap < bp`).
    pub fn new(ap: f64, bp: f64, at: f64, bt: f64, np: usize, nt: usize, q: T) -> Result<Self> {
        if np < 2 || nt < 2 {
            return Err(Error::InvalidArgument("np and nt must be at least 2"));
        }
        if !(ap < bp) || !(0.0 < at && at < bt && bt < core::f64::consts::PI) {
            return Err(Error::InvalidArgument("need ap < bp and 0 < at < bt < pi"));
        }
        let np_i: c_int = np.try_into().map_err(|_| Error::DimensionOverflow)?;
        let nt_i: c_int = nt.try_into().map_err(|_| Error::DimensionOverflow)?;
        let lim = [T::from_f64(ap), T::from_f64(bp), T::from_f64(at), T::from_f64(bt)];
        let mut ipar: Vec<c_int> = vec![0; 128];
        let mut dpar: Vec<T> = vec![T::default(); 4 * (5 * np / 2 + nt + 10) + 200];
        let mut stat: c_int = 0;
        unsafe {
            T::init_sph_np(
                [&lim[0], &lim[1], &lim[2], &lim[3]], &np_i, &nt_i, &q,
                ipar.as_mut_ptr(), dpar.as_mut_ptr(), &mut stat,
            );
        }
        check_tr(stat)?;
        Ok(Self {
            h: ptr::null_mut(),
            ipar,
            dpar,
            np,
            nt,
            hp: (bp - ap) / np as f64,
            ht: (bt - at) / nt as f64,
            at,
            committed: false,
        })
    }

    /// Required grid length, `(np + 1) * (nt + 1)`.
    #[must_use]
    pub fn grid_len(&self) -> usize {
        (self.np + 1) * (self.nt + 1)
    }

    /// Solve in place (see the type docs for what `f` holds on entry / exit).
    pub fn solve(&mut self, f: &mut [T]) -> Result<()> {
        if f.len() < self.grid_len() {
            return Err(Error::InvalidArgument("grid buffer too small"));
        }
        let (np, nt) = (self.np, self.nt);
        let w = np + 1;
        let (hp, ht) = (self.hp, self.ht);
        let sin_at = |j: usize| (self.at + j as f64 * ht).sin();
        let g: Vec<f64> = f[..self.grid_len()].iter().map(|v| v.to_f64()).collect();
        let mut rhs = vec![T::default(); self.grid_len()];
        for j in 1..nt {
            let sj = sin_at(j);
            let sp = 0.5 * (sj + sin_at(j + 1));
            let sm = 0.5 * (sj + sin_at(j - 1));
            let c = ht * ht / (sj * hp * hp);
            for i in 1..np {
                let mut v = -ht * ht * sj * g[i + j * w];
                if j == 1 { v += sm * g[i]; }
                if j == nt - 1 { v += sp * g[i + nt * w]; }
                if i == 1 { v += c * g[j * w]; }
                if i == np - 1 { v += c * g[np + j * w]; }
                rhs[i + j * w] = T::from_f64(v);
            }
        }
        let mut stat: c_int = 0;
        if !self.committed {
            let mut probe = rhs.clone();
            unsafe {
                T::commit_sph_np(probe.as_mut_ptr(), &mut self.h, self.ipar.as_mut_ptr(), self.dpar.as_mut_ptr(), &mut stat);
            }
            check_tr(stat)?;
            self.committed = true;
        }
        unsafe {
            T::solve_sph_np(rhs.as_mut_ptr(), &mut self.h, self.ipar.as_mut_ptr(), self.dpar.as_mut_ptr(), &mut stat);
        }
        check_tr(stat)?;
        for j in 1..nt {
            for i in 1..np {
                f[i + j * w] = rhs[i + j * w];
            }
        }
        Ok(())
    }
}

impl<T: HelmholtzScalar> Drop for SphericalPatchHelmholtz<T> {
    fn drop(&mut self) {
        if self.h.is_null() {
            return;
        }
        let mut stat: c_int = 0;
        unsafe { T::free_sph_np(&mut self.h, self.ipar.as_mut_ptr(), &mut stat) };
    }
}
