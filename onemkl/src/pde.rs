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
#[allow(missing_docs)]
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
}

macro_rules! impl_helmholtz_scalar {
    ($t:ty, $init:ident, $commit:ident, $solve:ident) => {
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
        }
    };
}
impl_helmholtz_scalar!(f32, s_init_helmholtz_2d, s_commit_helmholtz_2d, s_helmholtz_2d);
impl_helmholtz_scalar!(f64, d_init_helmholtz_2d, d_commit_helmholtz_2d, d_helmholtz_2d);

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
pub struct Helmholtz2D<T: HelmholtzScalar> {
    handle: DFTI_DESCRIPTOR_HANDLE,
    ipar: Vec<c_int>,
    dpar: Vec<T>,
    nx: usize,
    ny: usize,
    bc: [Boundary; 4],
    /// `(by - ay)² / ny²`, used to undo MKL's solution scaling.
    hx2: f64,
    hy2: f64,
    committed: bool,
}

unsafe impl<T: HelmholtzScalar + Send> Send for Helmholtz2D<T> {}

impl<T: HelmholtzScalar> Helmholtz2D<T> {
    /// Build a solver. Only all-`Dirichlet` boundaries are accepted for
    /// now. `bc` is ordered `[x = ax, x = bx, y = ay, y = by]`;
    /// a periodic pair must be periodic on both sides. `nx` and `ny`
    /// must be even (MKL requirement). `q` is the Helmholtz coefficient
    /// (`q ≥ 0`; `q = 0` with all-Neumann/periodic sides is singular and
    /// the solution is determined only up to a constant).
    pub fn new(
        (ax, bx): (T, T),
        (ay, by): (T, T),
        nx: usize,
        ny: usize,
        bc: [Boundary; 4],
        q: T,
    ) -> Result<Self> {
        if nx < 4 || ny < 4 || nx % 2 != 0 || ny % 2 != 0 {
            return Err(Error::InvalidArgument(
                "nx and ny must be even and at least 4",
            ));
        }
        if bc.iter().any(|b| *b != Boundary::Dirichlet) {
            return Err(Error::InvalidArgument(
                "only Dirichlet boundaries are supported so far (Neumann/periodic unverified)",
            ));
        }
        if (bc[0] == Boundary::Periodic) != (bc[1] == Boundary::Periodic)
            || (bc[2] == Boundary::Periodic) != (bc[3] == Boundary::Periodic)
        {
            return Err(Error::InvalidArgument(
                "periodic boundary must be set on both sides of the pair",
            ));
        }
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

        let handle: DFTI_DESCRIPTOR_HANDLE = ptr::null_mut();
        let hy = (by.to_f64() - ay.to_f64()) / ny as f64;
        let hx = (bx.to_f64() - ax.to_f64()) / nx as f64;
        let this = Self { handle, ipar, dpar, nx, ny, bc, hx2: hx * hx, hy2: hy * hy, committed: false };
        Ok(this)
    }

    /// Required length of the grid buffer, `(nx + 1) * (ny + 1)`.
    #[must_use]
    pub fn grid_len(&self) -> usize {
        (self.nx + 1) * (self.ny + 1)
    }

    /// Solve in place. `f` holds the right-hand side on entry and the
    /// solution on exit. `bd_ax`/`bd_bx` (length `ny + 1`) and
    /// `bd_ay`/`bd_by` (length `nx + 1`) carry the Dirichlet values or
    /// Neumann derivatives for the corresponding side; for a periodic
    /// side pass an empty slice (ignored).
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
        let sides = [
            (bd_ax, self.ny + 1),
            (bd_bx, self.ny + 1),
            (bd_ay, self.nx + 1),
            (bd_by, self.nx + 1),
        ];
        let zeros_x: Vec<T> = vec![T::default(); self.ny + 1];
        let zeros_y: Vec<T> = vec![T::default(); self.nx + 1];
        let mut ptrs: [*const T; 4] = [ptr::null(); 4];
        for (k, (s, need)) in sides.iter().enumerate() {
            if self.bc[k] == Boundary::Periodic {
                ptrs[k] = if k < 2 { zeros_x.as_ptr() } else { zeros_y.as_ptr() };
            } else if s.len() < *need {
                return Err(Error::InvalidArgument("boundary data slice too short"));
            } else {
                ptrs[k] = s.as_ptr();
            }
        }
        // MKL (measured on 2026.2) applies no Dirichlet boundary lift: it
        // solves the homogeneous problem on the interior. Move the known
        // boundary values to the right-hand side here: for an interior
        // node next to a Dirichlet side, f -= g / h^2.
        {
            let (nx, ny) = (self.nx, self.ny);
            let (hx2, hy2) = (self.hx2, self.hy2);
            let w = nx + 1;
            let sub = |v: &mut T, g: T, h2: f64| *v = T::from_f64(v.to_f64() - g.to_f64() / h2);
            for j in 1..ny {
                sub(&mut f[1 + j * w], bd_ax[j], hx2);
                sub(&mut f[nx - 1 + j * w], bd_bx[j], hx2);
            }
            for i in 1..nx {
                sub(&mut f[i + w], bd_ay[i], hy2);
                sub(&mut f[i + (ny - 1) * w], bd_by[i], hy2);
            }
        }
        let mut stat: c_int = 0;
        if !self.committed {
            // MKL needs the real boundary data (and a right-hand side of
            // the right shape) at commit time; commit on a copy of `f`.
            let mut probe: Vec<T> = f[..self.grid_len()].to_vec();
            unsafe {
                T::commit_2d(
                    probe.as_mut_ptr(), ptrs[0], ptrs[1], ptrs[2], ptrs[3],
                    &mut self.handle, self.ipar.as_mut_ptr(),
                    self.dpar.as_mut_ptr(), &mut stat,
                );
            }
            check_tr(stat)?;
            self.committed = true;
        }
        unsafe {
            T::solve_2d(
                f.as_mut_ptr(), ptrs[0], ptrs[1], ptrs[2], ptrs[3],
                &mut self.handle, self.ipar.as_mut_ptr(),
                self.dpar.as_ptr(), &mut stat,
            );
        }
        check_tr(stat)?;
        // Measured against MKL 2026.2 (pip, LP64): the library returns
        // (-Δ + q)⁻¹ f / hy² rather than the documented (Δ - q)⁻¹ f, so
        // map non-Dirichlet nodes back by -hy². MKL leaves Dirichlet
        // nodes untouched, so write the user's boundary data there.
        let nx = self.nx;
        for j in 0..=self.ny {
            for i in 0..=nx {
                let fixed = (i == 0 && self.bc[0] == Boundary::Dirichlet)
                    || (i == nx && self.bc[1] == Boundary::Dirichlet)
                    || (j == 0 && self.bc[2] == Boundary::Dirichlet)
                    || (j == self.ny && self.bc[3] == Boundary::Dirichlet);
                if !fixed {
                    let v = &mut f[i + j * (nx + 1)];
                    *v = T::from_f64(-self.hy2 * v.to_f64());
                }
            }
        }
        let ny = self.ny;
        if self.bc[0] == Boundary::Dirichlet {
            for j in 0..=ny { f[j * (nx + 1)] = bd_ax[j]; }
        }
        if self.bc[1] == Boundary::Dirichlet {
            for j in 0..=ny { f[nx + j * (nx + 1)] = bd_bx[j]; }
        }
        if self.bc[2] == Boundary::Dirichlet {
            for i in 0..=nx { f[i] = bd_ay[i]; }
        }
        if self.bc[3] == Boundary::Dirichlet {
            for i in 0..=nx { f[i + ny * (nx + 1)] = bd_by[i]; }
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
