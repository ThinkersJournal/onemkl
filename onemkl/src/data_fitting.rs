// SPDX-License-Identifier: MIT OR Apache-2.0
//! Data fitting — 1-D spline interpolation and integration.
//!
//! Wraps oneMKL's Data Fitting `df*` task interface in an owned
//! [`CubicSpline1d`] type that bundles the underlying `DFTaskPtr`
//! handle and the spline coefficient storage. The task is freed on
//! drop.
//!
//! Cubic spline subtypes covered: [`natural`](CubicSpline1d::natural),
//! [`bessel`](CubicSpline1d::bessel), [`akima`](CubicSpline1d::akima),
//! and [`hermite`](CubicSpline1d::hermite) (which takes user-supplied
//! first derivatives at each knot).

use core::ffi::{c_int, c_longlong, c_void};
use core::ptr;
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use onemkl_sys::{self as sys, DFTaskPtr};

use crate::error::{Error, Result};

// =====================================================================
// Trait wiring
// =====================================================================

/// Real scalar types supported by the data-fitting interface.
#[allow(missing_docs)]
pub trait DataFittingScalar: Copy + PartialOrd + 'static {
    /// Raw `df_new_task_1d` call.
    ///
    /// # Safety
    ///
    /// `task` must be valid for a write of one `DFTaskPtr`. `nx >= 2` is the number of breakpoints and `x` must hold `nx` strictly increasing values (or 2 values `[a, b]` when `xhint` says the partition is uniform). `ny >= 1` is the number of functions and `y` must hold `ny * nx` values laid out per `yhint` (row-major: `ny` rows of `nx`; column-major by hint).
    /// **Pointer retention:** the task stores the pointers passed to it; each such buffer must stay allocated at a stable address and be neither mutated nor aliased by a `&mut` while MKL may read it (outputs: not accessed at all) until the task is deleted (`dfDeleteTask`) or the pointer replaced.
    /// On non-zero return `*task` must not be used. The MKL library must be linked and usable.
    /// **Verified** by guard-page probe (smallest buffer for which the call did not fault: `x`: nx, `y`: nx * ny elements, with a non-uniform partition and no hints; oneMKL 2026.2, Linux x86_64). Extents not listed as verified in this block are unverified (from MKL documentation).
    unsafe fn df_new_task_1d(
        task: *mut DFTaskPtr,
        nx: c_int,
        x: *const Self,
        xhint: c_int,
        ny: c_int,
        y: *const Self,
        yhint: c_int,
    ) -> c_int;
    /// Raw `df_edit_pp_spline_1d` call.
    ///
    /// # Safety
    ///
    /// `task` must be a live `DFTaskPtr` from `df_new_task_1d` and not yet deleted, and not used by another thread during the call.
    /// `s_order`, `s_type`, `bc_type`, `ic_type` must be valid `DF_PP_*`/`DF_BC_*`/`DF_IC_*` constants. `bc` and `ic` must be readable for the lengths those types define (usually 0 to 2 values for `bc`, `nx - 1` or `ny * (nx-1)` for `ic` as MKL documents); `scoeff` must be valid for reads and writes of `ny * (nx - 1) * s_order` coefficients.
    /// **Pointer retention:** the task stores the pointers passed to it; each such buffer must stay allocated at a stable address and be neither mutated nor aliased by a `&mut` while MKL may read it (outputs: not accessed at all) until the task is deleted (`dfDeleteTask`) or the pointer replaced.
    /// The MKL library must be linked and usable.
    /// **Verified** by guard-page probe (smallest buffer for which the call did not fault: `scoeff`: ny * (nx - 1) * 4 elements for a cubic spline (s_order = 4), with `bc`/`ic` null for free-end/no-IC; oneMKL 2026.2, Linux x86_64). Extents not listed as verified in this block are unverified (from MKL documentation).
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the oneMKL C signature df?EditPPSpline1D argument for argument"
    )]
    unsafe fn df_edit_pp_spline_1d(
        task: DFTaskPtr,
        s_order: c_int,
        s_type: c_int,
        bc_type: c_int,
        bc: *const Self,
        ic_type: c_int,
        ic: *const Self,
        scoeff: *const Self,
        scoeffhint: c_int,
    ) -> c_int;
    /// Raw `df_construct_1d` call.
    ///
    /// # Safety
    ///
    /// `task` must be a live `DFTaskPtr` from `df_new_task_1d` and not yet deleted, and not used by another thread during the call.
    /// `s_format` and `method` must be `DF_PP_SPLINE` / `DF_METHOD_PP` as MKL requires; spline parameters must already have been set with `df_edit_pp_spline_1d`, whose registered buffers must still be valid.
    /// The MKL library must be linked and usable.
    /// Extents not listed as verified in this block are unverified (from MKL documentation). (no pointer arguments of its own; it reads the buffers registered earlier.)
    unsafe fn df_construct_1d(task: DFTaskPtr, s_format: c_int, method: c_int) -> c_int;
    /// Raw `df_interpolate_1d` call.
    ///
    /// # Safety
    ///
    /// `task` must be a live `DFTaskPtr` from `df_new_task_1d` and not yet deleted, and not used by another thread during the call.
    /// `nsite` is the number of evaluation sites and `site` must hold `nsite` values (2 for a uniform `sitehint`). `ndorder` is the number of derivative orders and `dorder` must hold `ndorder` entries (non-zero flag per requested derivative). `datahint` may be null or point to the 1-element hint array MKL documents. `r` must be valid for writes of `ny * nsite * (number of nonzero dorder entries)` elements laid out per `rhint`; `cell` may be null, otherwise it must be valid for writes of `nsite` `c_int` values.
    /// `r` and `cell` must not overlap `site`, `dorder` or any task-registered buffer. The MKL library must be linked and usable.
    /// **Verified** by guard-page probe (smallest buffer for which the call did not fault: `site`: nsite, `dorder`: ndorder entries, `r`: nsite * ny * ndorder for ndorder = 2 with both flags set, `cell` null; oneMKL 2026.2, Linux x86_64). Extents not listed as verified in this block are unverified (from MKL documentation).
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the oneMKL C signature df?Interpolate1D argument for argument"
    )]
    unsafe fn df_interpolate_1d(
        task: DFTaskPtr,
        estimation_type: c_int,
        method: c_int,
        nsite: c_int,
        site: *const Self,
        sitehint: c_int,
        ndorder: c_int,
        dorder: *const c_int,
        datahint: *const Self,
        r: *mut Self,
        rhint: c_int,
        cell: *mut c_int,
    ) -> c_int;
    /// Raw `df_integrate_1d` call.
    ///
    /// # Safety
    ///
    /// `task` must be a live `DFTaskPtr` from `df_new_task_1d` and not yet deleted, and not used by another thread during the call.
    /// `nlim` is the number of integration intervals; `llim` and `rlim` must each hold `nlim` values (or one value when the corresponding hint says so), with `llim <= rlim` inside the breakpoint range. `ldatahint` and `rdatahint` may be null or point to the hint arrays MKL documents. `r` must be valid for writes of `ny * nlim` elements laid out per `rhint`; `r` must not overlap any input or task-registered buffer.
    /// The MKL library must be linked and usable.
    /// **Verified** by guard-page probe (smallest buffer for which the call did not fault: `llim`: nlim, `r`: nlim * ny; oneMKL 2026.2, Linux x86_64). Extents not listed as verified in this block are unverified (from MKL documentation).
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the oneMKL C signature df?Integrate1D argument for argument"
    )]
    unsafe fn df_integrate_1d(
        task: DFTaskPtr,
        method: c_int,
        nlim: c_int,
        llim: *const Self,
        llimhint: c_int,
        rlim: *const Self,
        rlimhint: c_int,
        ldatahint: *const Self,
        rdatahint: *const Self,
        r: *mut Self,
        rhint: c_int,
    ) -> c_int;
    /// Raw `df_integrate_ex_1d` call with the integration call-backs of
    /// `args` (see [`IntegrateExArgs`]); a panic inside a call-back is caught
    /// at the C boundary and left in `args.panic` for the caller to resume.
    ///
    /// # Safety
    ///
    /// `task` must be a live `DFTaskPtr` from `df_new_task_1d`, not used by
    /// another thread during the call. `args.llim` and `args.rlim` must each
    /// hold `args.nlim` values and `args.r` must be valid for writes of
    /// `args.nlim` elements (`ny` is 1), none overlapping a task-registered
    /// buffer. The call-backs run on the calling thread.
    /// **Measured** on oneMKL 2026.2 (`tools/size-probes/df_probe3.c`): one
    /// call-back call per limit piece with `n = 1`, 1-based cell arguments, a
    /// positive status returned as the routine status with the result kept, a
    /// negative one returned with `r` untouched.
    unsafe fn df_integrate_ex_1d(task: DFTaskPtr, args: &mut IntegrateExArgs<'_, Self>) -> c_int;

    /// Raw `df?SearchCellsEx1D` call with one user-defined search call-back.
    ///
    /// # Safety
    ///
    /// `task` must be a live Data Fitting task with a constructed partition; `cells` must have exactly `sites.len()` elements. The call-back runs on the calling thread; a panic in it is stored in `args.panic`, not unwound through oneMKL.
    /// **Measured** on oneMKL 2026.2 (`tools/size-probes/df_search.c`): one call-back call with `n = sites.len()` and `library_params` NULL; cells are 1-based; a negative call-back status is returned with `cells` untouched, a positive one is returned with the cells filled.
    unsafe fn df_search_cells_ex_1d(
        task: DFTaskPtr,
        sites: &[Self],
        cells: &mut [c_int],
        args: &mut SearchExArgs<'_, Self>,
    ) -> c_int;
}

/// What a user-defined search call-back receives (`df?SearchCellsCallBack`).
/// Cells are 1-based, as oneMKL reports them: cell `k` is
/// `[x[k - 1], x[k])`, 0 is left of the first breakpoint and `nx` right of
/// the last.
#[non_exhaustive]
pub struct SearchRequest<'a, T> {
    /// The interpolation sites to locate.
    pub sites: &'a [T],
    /// Where to write the 1-based cell of each site.
    pub cells: &'a mut [i64],
    /// Per site: set 1 when `cells[i]` is exactly right, 0 when it is only a
    /// starting guess the library should refine. Ignored (and need not be
    /// written) if the call-back returns [`DF_STATUS_EXACT_RESULT`].
    pub exact: &'a mut [c_int],
}

// Cells the call-back declares exact (`exact[i] == 1`, or every cell when it
// returns `DF_STATUS_EXACT_RESULT`) are checked against the knots before
// oneMKL uses them: cell `c` must satisfy `0 <= c <= nx` and
// `x[c - 1] <= site <= x[c]` (no lower bound for `c == 0`, no upper for
// `c == nx`). A violation is `Error::SearchCell`; a positive status is
// `Error::SearchCallbackStatus`.

/// `DF_STATUS_EXACT_RESULT`: returned from a search call-back to declare every
/// cell exact.
pub const DF_STATUS_EXACT_RESULT: i32 = sys::DF_STATUS_EXACT_RESULT as i32;

/// A user-defined search call-back: return 0, [`DF_STATUS_EXACT_RESULT`], a
/// negative error code or a positive warning code.
pub type SearchCallback<'a, T> = &'a mut dyn FnMut(SearchRequest<'_, T>) -> i32;

#[doc(hidden)]
pub struct SearchExArgs<'a, T> {
    pub callback: SearchCallback<'a, T>,
    /// The breakpoints of the task, to check exact cells against.
    pub x: &'a [T],
    /// Set when the call-back panicked.
    pub panic: Option<Box<dyn Any + Send>>,
    /// First problem found in what the call-back returned.
    pub error: Option<Error>,
}

impl<T: DataFittingScalar> SearchExArgs<'_, T> {
    /// The error to report for this search, if any (a call-back problem
    /// found by the trampoline).
    pub fn take_error(&mut self) -> Option<Error> {
        self.error.take()
    }
}

/// Check one exact cell (1-based, see [`SearchRequest`]) against the knots.
fn check_exact_cell<T: DataFittingScalar>(
    x: &[T],
    index: usize,
    site: T,
    cell: i64,
) -> Option<Error> {
    let nx = x.len();
    let bad = || {
        Some(Error::SearchCell {
            index,
            cell,
            max_cell: nx,
        })
    };
    let Ok(c) = usize::try_from(cell) else {
        return bad();
    };
    if c > nx {
        return bad();
    }
    if c > 0 && !(x[c - 1] <= site) {
        return bad();
    }
    if c < nx && !(site <= x[c]) {
        return bad();
    }
    None
}

/// # Safety
///
/// `user` is the `SearchExArgs` passed with the call-back; `n`, `site`, `cell` and `flag` are the arrays oneMKL passes (`n` elements each).
unsafe fn run_search_callback<T: DataFittingScalar>(
    n: *mut c_longlong,
    site: *mut T,
    cell: *mut c_longlong,
    flag: *mut c_int,
    user: *mut c_void,
) -> c_int {
    // SAFETY: per the contract above.
    unsafe {
        let args = &mut *user.cast::<SearchExArgs<'_, T>>();
        let n = usize::try_from(*n).unwrap_or(0);
        let req = SearchRequest {
            sites: core::slice::from_raw_parts(site, n),
            cells: core::slice::from_raw_parts_mut(cell, n),
            exact: core::slice::from_raw_parts_mut(flag, n),
        };
        match catch_unwind(AssertUnwindSafe(|| (args.callback)(req))) {
            Ok(status) => {
                let all_exact = status == DF_STATUS_EXACT_RESULT;
                // SAFETY: the same arrays, re-read after the call-back.
                let (sites, cells, flags) = (
                    core::slice::from_raw_parts(site, n),
                    core::slice::from_raw_parts(cell, n),
                    core::slice::from_raw_parts(flag, n),
                );
                for i in 0..n {
                    if (all_exact || flags[i] == 1)
                        && let Some(e) = check_exact_cell(args.x, i, sites[i], cells[i])
                    {
                        args.error.get_or_insert(e);
                        // Make oneMKL stop without using the cells.
                        return -1;
                    }
                }
                if status > 0 && !all_exact {
                    args.error
                        .get_or_insert(Error::SearchCallbackStatus(status));
                }
                status
            }
            Err(payload) => {
                args.panic = Some(payload);
                -1
            }
        }
    }
}

/// The data a user-defined integration call-back receives for one piece of
/// an integration interval (`df?IntegrCallBack`). Cell indices are the 1-based
/// ones oneMKL passes (0 left of the interpolation interval, `nx` at or
/// right of its end).
#[non_exhaustive]
pub struct IntegrationPieces<'a, T> {
    /// Cells containing the left limits.
    pub left_cells: &'a [i64],
    /// Left integration limits.
    pub left_limits: &'a [T],
    /// Cells containing the right limits.
    pub right_cells: &'a [i64],
    /// Right integration limits.
    pub right_limits: &'a [T],
    /// Where to write one integral per limit pair.
    pub out: &'a mut [T],
}

/// A user-defined integration call-back: fill `out` and return 0, a negative
/// error code, or a positive warning code.
pub type IntegrationCallback<'a, T> = &'a mut dyn FnMut(IntegrationPieces<'_, T>) -> i32;

/// The optional call-backs of `df?IntegrateEx1D`. A `None` entry uses
/// the library's own spline integration (or cell search) for that region.
#[non_exhaustive]
pub struct IntegrateCallbacks<'a, T> {
    /// For the part of an interval left of the interpolation interval.
    pub left: Option<IntegrationCallback<'a, T>>,
    /// For the part right of it.
    pub right: Option<IntegrationCallback<'a, T>>,
    /// For the part inside it.
    pub interior: Option<IntegrationCallback<'a, T>>,
    /// A user-defined cell search for the integration limits. oneMKL calls
    /// it twice per `integrate_with_callbacks`: first for the left limits,
    /// then for the right ones, each time with every limit as a site. Cells
    /// are 1-based like [`SearchRequest`]'s and are checked before oneMKL
    /// uses them: see [`SearchRequest`]. (oneMKL 2026.2,
    /// `tools/size-probes/df_search3.c`: an unchecked out-of-range exact
    /// cell gives integrals of 0 with status 0, and a positive call-back
    /// status is swallowed; both are errors here.)
    pub search: Option<SearchCallback<'a, T>>,
}

impl<'a, T> IntegrateCallbacks<'a, T> {
    /// No call-backs: the library's own integration and search everywhere.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the call-back for the part left of the interpolation interval.
    #[must_use]
    pub fn left(mut self, f: IntegrationCallback<'a, T>) -> Self {
        self.left = Some(f);
        self
    }

    /// Set the call-back for the part right of the interpolation interval.
    #[must_use]
    pub fn right(mut self, f: IntegrationCallback<'a, T>) -> Self {
        self.right = Some(f);
        self
    }

    /// Set the call-back for the part inside the interpolation interval.
    #[must_use]
    pub fn interior(mut self, f: IntegrationCallback<'a, T>) -> Self {
        self.interior = Some(f);
        self
    }

    /// Set the user-defined cell search (see [`search`](Self::search) field).
    #[must_use]
    pub fn search(mut self, f: SearchCallback<'a, T>) -> Self {
        self.search = Some(f);
        self
    }
}

impl<T> Default for IntegrateCallbacks<'_, T> {
    fn default() -> Self {
        Self {
            left: None,
            right: None,
            interior: None,
            search: None,
        }
    }
}

/// Arguments of [`DataFittingScalar::df_integrate_ex_1d`], gathered in one
/// struct so the raw call stays within clippy's argument limit.
#[doc(hidden)]
pub struct IntegrateExArgs<'a, T: DataFittingScalar> {
    pub nlim: c_int,
    pub llim: *const T,
    pub rlim: *const T,
    pub r: *mut T,
    /// The breakpoints of the task (for checking search cells).
    pub x: &'a [T],
    pub callbacks: IntegrateCallbacks<'a, T>,
    /// Set when a call-back panicked.
    pub panic: Option<Box<dyn Any + Send>>,
    /// A problem found in what the search call-back returned.
    pub search_error: Option<Error>,
}

struct CbSlot<'a, T> {
    f: IntegrationCallback<'a, T>,
    panic: Option<Box<dyn Any + Send>>,
}

/// Shared body of the C trampolines.
///
/// # Safety
///
/// The pointers are those oneMKL passes to an integration call-back: `n`
/// points to the number of pairs and each array holds that many elements;
/// `user` is the `CbSlot` passed with the call-back.
unsafe fn run_integr_callback<T>(
    n: *mut c_longlong,
    lcell: *mut c_longlong,
    llim: *mut T,
    rcell: *mut c_longlong,
    rlim: *mut T,
    r: *mut T,
    user: *mut c_void,
) -> c_int {
    // SAFETY: per the contract above.
    unsafe {
        let slot = &mut *user.cast::<CbSlot<'_, T>>();
        let n = usize::try_from(*n).unwrap_or(0);
        let pieces = IntegrationPieces {
            left_cells: core::slice::from_raw_parts(lcell, n),
            left_limits: core::slice::from_raw_parts(llim, n),
            right_cells: core::slice::from_raw_parts(rcell, n),
            right_limits: core::slice::from_raw_parts(rlim, n),
            out: core::slice::from_raw_parts_mut(r, n),
        };
        match catch_unwind(AssertUnwindSafe(|| (slot.f)(pieces))) {
            Ok(status) => status,
            Err(payload) => {
                slot.panic = Some(payload);
                -1
            }
        }
    }
}

macro_rules! impl_data_fitting_real {
    ($ty:ty,
        new_task=$new:ident, edit_spline=$edit:ident,
        construct=$ctor:ident, interpolate=$interp:ident,
        integrate=$integ:ident,
        integrate_ex=$integ_ex:ident, integr_cb=$cbty:ident,
        search_ex=$search_ex:ident, search_cb=$scbty:ident,
    ) => {
        impl DataFittingScalar for $ty {
            unsafe fn df_new_task_1d(
                task: *mut DFTaskPtr,
                nx: c_int,
                x: *const Self,
                xhint: c_int,
                ny: c_int,
                y: *const Self,
                yhint: c_int,
            ) -> c_int {
                unsafe { sys::$new(task, nx, x, xhint, ny, y, yhint) }
            }
            unsafe fn df_edit_pp_spline_1d(
                task: DFTaskPtr,
                s_order: c_int,
                s_type: c_int,
                bc_type: c_int,
                bc: *const Self,
                ic_type: c_int,
                ic: *const Self,
                scoeff: *const Self,
                scoeffhint: c_int,
            ) -> c_int {
                unsafe {
                    sys::$edit(
                        task, s_order, s_type, bc_type, bc, ic_type, ic, scoeff, scoeffhint,
                    )
                }
            }
            unsafe fn df_construct_1d(task: DFTaskPtr, s_format: c_int, method: c_int) -> c_int {
                unsafe { sys::$ctor(task, s_format, method) }
            }
            unsafe fn df_interpolate_1d(
                task: DFTaskPtr,
                estimation_type: c_int,
                method: c_int,
                nsite: c_int,
                site: *const Self,
                sitehint: c_int,
                ndorder: c_int,
                dorder: *const c_int,
                datahint: *const Self,
                r: *mut Self,
                rhint: c_int,
                cell: *mut c_int,
            ) -> c_int {
                unsafe {
                    sys::$interp(
                        task,
                        estimation_type,
                        method,
                        nsite,
                        site,
                        sitehint,
                        ndorder,
                        dorder,
                        datahint,
                        r,
                        rhint,
                        cell,
                    )
                }
            }
            unsafe fn df_integrate_1d(
                task: DFTaskPtr,
                method: c_int,
                nlim: c_int,
                llim: *const Self,
                llimhint: c_int,
                rlim: *const Self,
                rlimhint: c_int,
                ldatahint: *const Self,
                rdatahint: *const Self,
                r: *mut Self,
                rhint: c_int,
            ) -> c_int {
                unsafe {
                    sys::$integ(
                        task, method, nlim, llim, llimhint, rlim, rlimhint, ldatahint, rdatahint,
                        r, rhint,
                    )
                }
            }
            unsafe fn df_integrate_ex_1d(
                task: DFTaskPtr,
                args: &mut IntegrateExArgs<'_, Self>,
            ) -> c_int {
                unsafe extern "C" fn tramp(
                    n: *mut c_longlong,
                    lcell: *mut c_longlong,
                    llim: *mut $ty,
                    rcell: *mut c_longlong,
                    rlim: *mut $ty,
                    r: *mut $ty,
                    user: *mut c_void,
                    _library_params: *mut sys::dfIntegrCallBackLibraryParams,
                ) -> c_int {
                    // SAFETY: oneMKL passes valid arguments and the slot pointer given below.
                    unsafe { run_integr_callback::<$ty>(n, lcell, llim, rcell, rlim, r, user) }
                }
                let mut left = args
                    .callbacks
                    .left
                    .take()
                    .map(|f| CbSlot { f, panic: None });
                let mut right = args
                    .callbacks
                    .right
                    .take()
                    .map(|f| CbSlot { f, panic: None });
                let mut inside = args
                    .callbacks
                    .interior
                    .take()
                    .map(|f| CbSlot { f, panic: None });
                unsafe extern "C" fn search_tramp(
                    n: *mut c_longlong,
                    site: *mut $ty,
                    cell: *mut c_longlong,
                    flag: *mut c_int,
                    user: *mut c_void,
                    _library_params: *mut sys::dfSearchCallBackLibraryParams,
                ) -> c_int {
                    // SAFETY: oneMKL passes valid arrays and the `SearchExArgs` pointer given below.
                    unsafe { run_search_callback::<$ty>(n, site, cell, flag, user) }
                }
                let mut search = args.callbacks.search.take().map(|callback| SearchExArgs {
                    callback,
                    x: args.x,
                    panic: None,
                    error: None,
                });
                let (scb, sp): (sys::$scbty, *const c_void) = match search.as_mut() {
                    Some(a) => (Some(search_tramp), ptr::from_mut(a).cast_const().cast()),
                    None => (None, ptr::null()),
                };
                fn slot_ptr<T>(s: &mut Option<CbSlot<'_, T>>) -> *const c_void {
                    s.as_mut()
                        .map_or(ptr::null(), |s| ptr::from_mut(s).cast_const().cast())
                }
                let cb = |present: bool| -> sys::$cbty { if present { Some(tramp) } else { None } };
                let (lcb, rcb, icb) = (
                    cb(left.is_some()),
                    cb(right.is_some()),
                    cb(inside.is_some()),
                );
                let (lp, rp, ip) = (
                    slot_ptr(&mut left),
                    slot_ptr(&mut right),
                    slot_ptr(&mut inside),
                );
                let status = unsafe {
                    sys::$integ_ex(
                        task,
                        sys::DF_METHOD_PP as c_int,
                        args.nlim,
                        args.llim,
                        sys::DF_NO_HINT as c_int,
                        args.rlim,
                        sys::DF_NO_HINT as c_int,
                        ptr::null(),
                        ptr::null(),
                        args.r,
                        sys::DF_NO_HINT as c_int,
                        lcb,
                        lp,
                        rcb,
                        rp,
                        icb,
                        ip,
                        scb,
                        sp,
                    )
                };
                args.panic = left
                    .and_then(|s| s.panic)
                    .or_else(|| right.and_then(|s| s.panic))
                    .or_else(|| inside.and_then(|s| s.panic))
                    .or_else(|| search.as_mut().and_then(|s| s.panic.take()));
                args.search_error = search.as_mut().and_then(SearchExArgs::take_error);
                status
            }
            unsafe fn df_search_cells_ex_1d(
                task: DFTaskPtr,
                sites: &[Self],
                cells: &mut [c_int],
                args: &mut SearchExArgs<'_, Self>,
            ) -> c_int {
                unsafe extern "C" fn tramp(
                    n: *mut c_longlong,
                    site: *mut $ty,
                    cell: *mut c_longlong,
                    flag: *mut c_int,
                    user: *mut c_void,
                    _library_params: *mut sys::dfSearchCallBackLibraryParams,
                ) -> c_int {
                    // SAFETY: oneMKL passes valid arrays and the `args` pointer given below.
                    unsafe { run_search_callback::<$ty>(n, site, cell, flag, user) }
                }
                let user: *const c_void = ptr::from_mut(args).cast_const().cast();
                unsafe {
                    sys::$search_ex(
                        task,
                        sys::DF_METHOD_STD as c_int,
                        sites.len() as c_int,
                        sites.as_ptr(),
                        sys::DF_NON_UNIFORM_PARTITION as c_int,
                        ptr::null(),
                        cells.as_mut_ptr(),
                        Some(tramp),
                        user,
                    )
                }
            }
        }
    };
}

impl_data_fitting_real!(
    f32,
    new_task = dfsNewTask1D,
    edit_spline = dfsEditPPSpline1D,
    construct = dfsConstruct1D,
    interpolate = dfsInterpolate1D,
    integrate = dfsIntegrate1D,
    integrate_ex = dfsIntegrateEx1D,
    integr_cb = dfsIntegrCallBack,
    search_ex = dfsSearchCellsEx1D,
    search_cb = dfsSearchCellsCallBack,
);
impl_data_fitting_real!(
    f64,
    new_task = dfdNewTask1D,
    edit_spline = dfdEditPPSpline1D,
    construct = dfdConstruct1D,
    interpolate = dfdInterpolate1D,
    integrate = dfdIntegrate1D,
    integrate_ex = dfdIntegrateEx1D,
    integr_cb = dfdIntegrCallBack,
    search_ex = dfdSearchCellsEx1D,
    search_cb = dfdSearchCellsCallBack,
);

// Note: oneMKL's data-fitting routines expose only real (`?s` / `?d`)
// variants — there are no complex spline interpolators — so
// [`DataFittingScalar`] is not implemented for `Complex32` / `Complex64`.

// =====================================================================
// CubicSpline1d
// =====================================================================

/// A natural cubic spline interpolant of a real-valued 1-D function.
///
/// The struct owns the spline coefficient storage and the underlying
/// task handle; both are released on drop.
pub struct CubicSpline1d<T: DataFittingScalar> {
    task: DFTaskPtr,
    // Inputs kept alive — MKL holds raw pointers into them while the
    // task is open.
    _x: Box<[T]>,
    _y: Box<[T]>,
    // Hermite splines need user-supplied derivatives at each knot; MKL
    // also keeps a raw pointer into this for the task's lifetime.
    _ic: Option<Box<[T]>>,
    // Boundary-condition values (Hermite: the end-point derivatives); MKL
    // also keeps a raw pointer into this.
    _bc: Option<Box<[T]>>,
    // Coefficient storage. For cubic spline of n knots, MKL writes
    // 4 * (n - 1) coefficients here.
    _coeffs: Box<[T]>,
}

unsafe impl<T: DataFittingScalar + Send> Send for CubicSpline1d<T> {}

impl<T: DataFittingScalar + num_traits::Zero + Default> CubicSpline1d<T> {
    /// Build a natural cubic spline interpolating `(x[i], y[i])`.
    /// `x` must be strictly increasing.
    pub fn natural(x: Vec<T>, y: Vec<T>) -> Result<Self> {
        Self::build(
            x,
            y,
            sys::DF_PP_NATURAL as c_int,
            sys::DF_BC_FREE_END as c_int,
            None,
            None,
        )
    }

    /// Build a Hyman cubic spline interpolating `(x[i], y[i])`; the slopes
    /// are limited so that monotone data stays monotone between knots.
    /// Needs at least 4 knots. Built with `DF_BC_FREE_END`: measured on oneMKL
    /// 2026.2 (`tools/size-probes/df_probe3.c`), that keeps plateau data
    /// monotone, while `DF_BC_NOT_A_KNOT` does not and `DF_NO_BC` fails with
    /// status -1014 even though the Intel reference lists it as supported.
    pub fn hyman(x: Vec<T>, y: Vec<T>) -> Result<Self> {
        if x.len() < 4 {
            return Err(Error::InvalidArgument(
                "Hyman spline needs at least 4 knots",
            ));
        }
        Self::build(
            x,
            y,
            sys::DF_PP_HYMAN as c_int,
            sys::DF_BC_FREE_END as c_int,
            None,
            None,
        )
    }

    /// Build a Bessel cubic spline interpolating `(x[i], y[i])`. The
    /// derivative at each knot is set to the slope of the parabola
    /// through that knot and its two neighbors.
    pub fn bessel(x: Vec<T>, y: Vec<T>) -> Result<Self> {
        Self::build(
            x,
            y,
            sys::DF_PP_BESSEL as c_int,
            sys::DF_BC_NOT_A_KNOT as c_int,
            None,
            None,
        )
    }

    /// Build an Akima cubic spline interpolating `(x[i], y[i])`. Akima
    /// splines are more robust to outliers than natural cubics
    /// because the derivative at each knot is a weighted average of
    /// nearby slopes.
    pub fn akima(x: Vec<T>, y: Vec<T>) -> Result<Self> {
        Self::build(
            x,
            y,
            sys::DF_PP_AKIMA as c_int,
            sys::DF_BC_NOT_A_KNOT as c_int,
            None,
            None,
        )
    }

    /// Build a Hermite cubic spline interpolating `(x[i], y[i])` with
    /// the supplied first derivatives at each knot. `derivatives` must
    /// have the same length as `x` and `y`.
    ///
    /// oneMKL takes the Hermite slopes in two parts (see
    /// `DF_PP_HERMITE`, `DF_BC_1ST_LEFT_DER` / `DF_BC_1ST_RIGHT_DER` and
    /// `DF_IC_1ST_DER` in `mkl_df_defines.h`; behaviour measured on oneMKL
    /// 2026.2): the end-point slopes go in as boundary-condition values and
    /// `ic` carries only the `n - 2` interior slopes. Passing all `n`
    /// slopes as `ic` makes MKL read them shifted by one knot and return
    /// wrong coefficients (published versions 0.1.0 to 0.3.10 did that).
    pub fn hermite(x: Vec<T>, y: Vec<T>, derivatives: Vec<T>) -> Result<Self> {
        if derivatives.len() != x.len() {
            return Err(Error::InvalidArgument(
                "derivatives must have the same length as x and y",
            ));
        }
        let n = derivatives.len();
        // oneMKL rejects a two-knot Hermite spline at construction (status
        // -1004; measured on oneMKL 2026.2), so refuse it here rather than
        // surface an opaque status.
        if n < 3 {
            return Err(Error::InvalidArgument(
                "Hermite spline needs at least 3 knots (oneMKL rejects 2)",
            ));
        }
        let bc = vec![derivatives[0], derivatives[n - 1]];
        let interior: Option<Vec<T>> = Some(derivatives[1..n - 1].to_vec());
        Self::build(
            x,
            y,
            sys::DF_PP_HERMITE as c_int,
            (sys::DF_BC_1ST_LEFT_DER | sys::DF_BC_1ST_RIGHT_DER) as c_int,
            Some(bc),
            interior,
        )
    }

    fn build(
        x: Vec<T>,
        y: Vec<T>,
        s_type: c_int,
        bc_type: c_int,
        bc: Option<Vec<T>>,
        ic: Option<Vec<T>>,
    ) -> Result<Self> {
        if x.len() != y.len() {
            return Err(Error::InvalidArgument("x and y must have the same length"));
        }
        let n = x.len();
        if n < 2 {
            return Err(Error::InvalidArgument(
                "cubic spline needs at least 2 knots",
            ));
        }

        let x_box = x.into_boxed_slice();
        let y_box = y.into_boxed_slice();
        let ic_box: Option<Box<[T]>> = ic.map(|v| v.into_boxed_slice());
        let bc_box: Option<Box<[T]>> = bc.map(|v| v.into_boxed_slice());
        let coeffs_box: Box<[T]> = (0..(n - 1) * sys::DF_PP_CUBIC as usize)
            .map(|_| T::default())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let nx: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ny: c_int = 1;

        let mut task: DFTaskPtr = ptr::null_mut();
        let status = unsafe {
            T::df_new_task_1d(
                &mut task,
                nx,
                x_box.as_ptr(),
                sys::DF_NON_UNIFORM_PARTITION as c_int,
                ny,
                y_box.as_ptr(),
                sys::DF_NO_HINT as c_int,
            )
        };
        check_df(status)?;

        let this = Self {
            task,
            _x: x_box,
            _y: y_box,
            _ic: ic_box,
            _bc: bc_box,
            _coeffs: coeffs_box,
        };

        let (ic_type, ic_ptr) = match this._ic.as_ref() {
            Some(buf) => (sys::DF_IC_1ST_DER as c_int, buf.as_ptr()),
            None => (sys::DF_NO_IC as c_int, ptr::null()),
        };
        let bc_ptr = match this._bc.as_ref() {
            Some(buf) => buf.as_ptr(),
            None => ptr::null(),
        };
        let status = unsafe {
            T::df_edit_pp_spline_1d(
                this.task,
                sys::DF_PP_CUBIC as c_int,
                s_type,
                bc_type,
                bc_ptr,
                ic_type,
                ic_ptr,
                this._coeffs.as_ptr(),
                sys::DF_NO_HINT as c_int,
            )
        };
        check_df(status)?;

        let status = unsafe {
            T::df_construct_1d(
                this.task,
                sys::DF_PP_SPLINE as c_int,
                sys::DF_METHOD_STD as c_int,
            )
        };
        check_df(status)?;
        Ok(this)
    }

    /// Evaluate the spline at the given sites, returning their function
    /// values.
    pub fn interpolate(&self, sites: &[T]) -> Result<Vec<T>>
    where
        T: Default,
    {
        let nsite: c_int = sites
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let mut out: Vec<T> = (0..sites.len()).map(|_| T::default()).collect();
        let dorder: [c_int; 1] = [1]; // request the function value (not derivatives)
        let status = unsafe {
            T::df_interpolate_1d(
                self.task,
                sys::DF_INTERP as c_int,
                sys::DF_METHOD_PP as c_int,
                nsite,
                sites.as_ptr(),
                sys::DF_NO_HINT as c_int,
                1,
                dorder.as_ptr(),
                ptr::null(),
                out.as_mut_ptr(),
                sys::DF_NO_HINT as c_int,
                ptr::null_mut(),
            )
        };
        check_df(status)?;
        Ok(out)
    }

    /// Evaluate derivatives of the spline at `sites`. `orders` lists the
    /// derivative orders wanted (`0` is the function value, up to `3` for a
    /// cubic); the result holds one vector per entry of `orders`, in the
    /// order requested, each with one value per site. Duplicate orders are
    /// rejected.
    pub fn interpolate_derivatives(&self, sites: &[T], orders: &[usize]) -> Result<Vec<Vec<T>>>
    where
        T: Default + Copy,
    {
        let max_order = sys::DF_PP_CUBIC as usize - 1;
        let mut flags = [0 as c_int; sys::DF_PP_CUBIC as usize];
        for &o in orders {
            if o > max_order {
                return Err(Error::InvalidArgument(
                    "derivative order exceeds the spline degree (3 for a cubic)",
                ));
            }
            if flags[o] != 0 {
                return Err(Error::InvalidArgument("duplicate derivative order"));
            }
            flags[o] = 1;
        }
        if orders.is_empty() {
            return Ok(Vec::new());
        }
        let nsite: c_int = sites
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let nd = orders.len();
        let mut out: Vec<T> = vec![T::default(); sites.len() * nd];
        let status = unsafe {
            T::df_interpolate_1d(
                self.task,
                sys::DF_INTERP as c_int,
                sys::DF_METHOD_PP as c_int,
                nsite,
                sites.as_ptr(),
                sys::DF_NO_HINT as c_int,
                flags.len() as c_int,
                flags.as_ptr(),
                ptr::null(),
                out.as_mut_ptr(),
                sys::DF_NO_HINT as c_int,
                ptr::null_mut(),
            )
        };
        check_df(status)?;
        // MKL orders the requested derivatives ascending; map back to
        // the caller's request order.
        let mut ascending: Vec<usize> = orders.to_vec();
        ascending.sort_unstable();
        let result = orders
            .iter()
            .map(|o| {
                let col = ascending.iter().position(|a| a == o).unwrap();
                (0..sites.len()).map(|i| out[i * nd + col]).collect()
            })
            .collect();
        Ok(result)
    }

    /// For each site, the index (0-based) of the knot interval
    /// `[x[k], x[k+1])` that contains it, as reported by MKL's cell
    /// search. MKL reports cells 1-based; this returns them 0-based.
    pub fn find_cells(&self, sites: &[T]) -> Result<Vec<usize>>
    where
        T: Default,
    {
        let nsite: c_int = sites
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let mut vals: Vec<T> = (0..sites.len()).map(|_| T::default()).collect();
        let mut cells: Vec<c_int> = vec![0; sites.len()];
        let dorder: [c_int; 1] = [1];
        let status = unsafe {
            T::df_interpolate_1d(
                self.task,
                sys::DF_CELL as c_int,
                sys::DF_METHOD_PP as c_int,
                nsite,
                sites.as_ptr(),
                sys::DF_NO_HINT as c_int,
                1,
                dorder.as_ptr(),
                ptr::null(),
                vals.as_mut_ptr(),
                sys::DF_NO_HINT as c_int,
                cells.as_mut_ptr(),
            )
        };
        check_df(status)?;
        Ok(cells
            .into_iter()
            .map(|c| (c as usize).saturating_sub(1))
            .collect())
    }

    /// Integrate the spline over `[a, b]` for each pair `(a, b)` in
    /// `(left, right)`. Returns one result per pair.
    pub fn integrate(&self, left: &[T], right: &[T]) -> Result<Vec<T>>
    where
        T: Default,
    {
        if left.len() != right.len() {
            return Err(Error::InvalidArgument(
                "left and right limit slices must have the same length",
            ));
        }
        let nlim: c_int = left
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let mut out: Vec<T> = (0..left.len()).map(|_| T::default()).collect();
        let status = unsafe {
            T::df_integrate_1d(
                self.task,
                sys::DF_METHOD_PP as c_int,
                nlim,
                left.as_ptr(),
                sys::DF_NO_HINT as c_int,
                right.as_ptr(),
                sys::DF_NO_HINT as c_int,
                ptr::null(),
                ptr::null(),
                out.as_mut_ptr(),
                sys::DF_NO_HINT as c_int,
            )
        };
        check_df(status)?;
        Ok(out)
    }

    /// Integrate like [`integrate`](Self::integrate), but with user-defined
    /// call-backs for the part of each interval left of, right of and inside
    /// the interpolation interval (`df?IntegrateEx1D`); a `None` call-back
    /// uses the library's own integration for that region. A negative
    /// call-back status becomes an error, a positive one is a warning and the
    /// result is kept; a panic in a call-back is resumed here once oneMKL has
    /// returned. See `docs/DATA-FITTING-FACTS.md` for the measured details.
    pub fn integrate_with_callbacks<'a>(
        &'a self,
        left: &[T],
        right: &[T],
        callbacks: IntegrateCallbacks<'a, T>,
    ) -> Result<Vec<T>>
    where
        T: Default,
    {
        integrate_ex_impl(self.task, &self._x, left, right, callbacks)
    }

    /// Locate `sites` with a user-defined search call-back
    /// (`df?SearchCellsEx1D`). The call-back works in oneMKL's 1-based
    /// cells (see [`SearchRequest`]); the result is 0-based like
    /// [`find_cells`](Self::find_cells). A call-back that sets `exact[i] = 1`
    /// (or returns [`DF_STATUS_EXACT_RESULT`]) has its cells used as given;
    /// otherwise oneMKL refines them. A negative status becomes an error, a
    /// positive one is `Error::SearchCallbackStatus`; an exact cell that does
    /// not contain its site is `Error::SearchCell` (checked before oneMKL uses
    /// it); a panic in the call-back is resumed here. Measured details: `docs/DATA-FITTING-FACTS.md`.
    pub fn find_cells_with_callback(
        &self,
        sites: &[T],
        callback: SearchCallback<'_, T>,
    ) -> Result<Vec<usize>> {
        search_cells_impl(self.task, &self._x, sites, callback)
    }
}

impl<T: DataFittingScalar> Drop for CubicSpline1d<T> {
    fn drop(&mut self) {
        if !self.task.is_null() {
            unsafe {
                let _ = sys::dfDeleteTask(&mut self.task);
            }
        }
    }
}

// =====================================================================
// LinearSpline1d
// =====================================================================

/// A piecewise-linear interpolant (`DF_PP_LINEAR`, default spline type,
/// no boundary or internal conditions: the combination oneMKL 2026.2
/// accepts, measured with `tools/size-probes/df_probe3.c`). `x` must be
/// strictly increasing.
pub struct LinearSpline1d<T: DataFittingScalar> {
    task: DFTaskPtr,
    // Inputs and coefficient storage; MKL keeps raw pointers into them
    // while the task is open. Two coefficients per cell.
    _x: Box<[T]>,
    _y: Box<[T]>,
    _coeffs: Box<[T]>,
}

unsafe impl<T: DataFittingScalar + Send> Send for LinearSpline1d<T> {}

impl<T: DataFittingScalar + Default> LinearSpline1d<T> {
    /// Build the piecewise-linear interpolant through `(x[i], y[i])`.
    pub fn new(x: Vec<T>, y: Vec<T>) -> Result<Self> {
        if x.len() != y.len() {
            return Err(Error::InvalidArgument("x and y must have the same length"));
        }
        let n = x.len();
        if n < 2 {
            return Err(Error::InvalidArgument(
                "linear spline needs at least 2 knots",
            ));
        }
        let x_box = x.into_boxed_slice();
        let y_box = y.into_boxed_slice();
        let coeffs_box: Box<[T]> = (0..(n - 1) * sys::DF_PP_LINEAR as usize)
            .map(|_| T::default())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let nx: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
        let mut task: DFTaskPtr = ptr::null_mut();
        check_df(unsafe {
            T::df_new_task_1d(
                &mut task,
                nx,
                x_box.as_ptr(),
                sys::DF_NON_UNIFORM_PARTITION as c_int,
                1,
                y_box.as_ptr(),
                sys::DF_NO_HINT as c_int,
            )
        })?;
        let this = Self {
            task,
            _x: x_box,
            _y: y_box,
            _coeffs: coeffs_box,
        };
        check_df(unsafe {
            T::df_edit_pp_spline_1d(
                this.task,
                sys::DF_PP_LINEAR as c_int,
                sys::DF_PP_DEFAULT as c_int,
                sys::DF_NO_BC as c_int,
                ptr::null(),
                sys::DF_NO_IC as c_int,
                ptr::null(),
                this._coeffs.as_ptr(),
                sys::DF_NO_HINT as c_int,
            )
        })?;
        check_df(unsafe {
            T::df_construct_1d(
                this.task,
                sys::DF_PP_SPLINE as c_int,
                sys::DF_METHOD_STD as c_int,
            )
        })?;
        Ok(this)
    }

    /// Evaluate the interpolant at `sites`.
    pub fn interpolate(&self, sites: &[T]) -> Result<Vec<T>> {
        let nsite: c_int = sites
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let mut out: Vec<T> = (0..sites.len()).map(|_| T::default()).collect();
        let dorder: [c_int; 1] = [1];
        check_df(unsafe {
            T::df_interpolate_1d(
                self.task,
                sys::DF_INTERP as c_int,
                sys::DF_METHOD_PP as c_int,
                nsite,
                sites.as_ptr(),
                sys::DF_NO_HINT as c_int,
                1,
                dorder.as_ptr(),
                ptr::null(),
                out.as_mut_ptr(),
                sys::DF_NO_HINT as c_int,
                ptr::null_mut(),
            )
        })?;
        Ok(out)
    }

    /// Integrate over `[left[i], right[i]]` for each pair.
    pub fn integrate(&self, left: &[T], right: &[T]) -> Result<Vec<T>> {
        if left.len() != right.len() {
            return Err(Error::InvalidArgument(
                "left and right limit slices must have the same length",
            ));
        }
        let nlim: c_int = left
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let mut out: Vec<T> = (0..left.len()).map(|_| T::default()).collect();
        check_df(unsafe {
            T::df_integrate_1d(
                self.task,
                sys::DF_METHOD_PP as c_int,
                nlim,
                left.as_ptr(),
                sys::DF_NO_HINT as c_int,
                right.as_ptr(),
                sys::DF_NO_HINT as c_int,
                ptr::null(),
                ptr::null(),
                out.as_mut_ptr(),
                sys::DF_NO_HINT as c_int,
            )
        })?;
        Ok(out)
    }

    /// Integrate like [`integrate`](Self::integrate), but with user-defined
    /// call-backs for the part of each interval left of, right of and inside
    /// the interpolation interval (`df?IntegrateEx1D`); a `None` call-back
    /// uses the library's own integration for that region. A negative
    /// call-back status becomes an error, a positive one is a warning and the
    /// result is kept; a panic in a call-back is resumed here once oneMKL has
    /// returned. See `docs/DATA-FITTING-FACTS.md` for the measured details.
    pub fn integrate_with_callbacks<'a>(
        &'a self,
        left: &[T],
        right: &[T],
        callbacks: IntegrateCallbacks<'a, T>,
    ) -> Result<Vec<T>>
    where
        T: Default,
    {
        integrate_ex_impl(self.task, &self._x, left, right, callbacks)
    }

    /// Locate `sites` with a user-defined search call-back
    /// (`df?SearchCellsEx1D`). The call-back works in oneMKL's 1-based
    /// cells (see [`SearchRequest`]); the result is 0-based like
    /// `CubicSpline1d::find_cells`. A call-back that sets `exact[i] = 1`
    /// (or returns [`DF_STATUS_EXACT_RESULT`]) has its cells used as given;
    /// otherwise oneMKL refines them. A negative status becomes an error, a
    /// positive one is `Error::SearchCallbackStatus`; an exact cell that does
    /// not contain its site is `Error::SearchCell` (checked before oneMKL uses
    /// it); a panic in the call-back is resumed here. Measured details: `docs/DATA-FITTING-FACTS.md`.
    pub fn find_cells_with_callback(
        &self,
        sites: &[T],
        callback: SearchCallback<'_, T>,
    ) -> Result<Vec<usize>> {
        search_cells_impl(self.task, &self._x, sites, callback)
    }
}

impl<T: DataFittingScalar> Drop for LinearSpline1d<T> {
    fn drop(&mut self) {
        if !self.task.is_null() {
            unsafe {
                let _ = sys::dfDeleteTask(&mut self.task);
            }
        }
    }
}

// =====================================================================
// QuadraticSpline1d
// =====================================================================

/// A piecewise-quadratic interpolant: the default quadratic
/// (`DF_PP_QUADRATIC`, `DF_PP_DEFAULT`) fixed by one extra value
/// (`DF_BC_Q_VAL`), or the Subbotin quadratic (`DF_PP_SUBBOTIN`) with free
/// ends and a knot array (`DF_IC_Q_KNOT`). The conditions are those measured
/// on oneMKL 2026.2 (`tools/size-probes/df_probe3.c`, `docs/DATA-FITTING-FACTS.md`):
/// `DF_NO_BC` is rejected for the default quadratic (-1014) and Subbotin is
/// rejected without a knot array (-1012). `x` must be strictly increasing.
pub struct QuadraticSpline1d<T: DataFittingScalar> {
    task: DFTaskPtr,
    // MKL keeps raw pointers into all of these while the task is open.
    _x: Box<[T]>,
    _y: Box<[T]>,
    _coeffs: Box<[T]>,
    _bc: Option<Box<[T]>>,
    _ic: Option<Box<[T]>>,
}

unsafe impl<T: DataFittingScalar + Send> Send for QuadraticSpline1d<T> {}

impl<T: DataFittingScalar + Default + PartialOrd> QuadraticSpline1d<T> {
    /// The default quadratic spline through `(x[i], y[i])`, fixed by its
    /// value `q_value` at the midpoint `(x[0] + x[1]) / 2`.
    pub fn with_midpoint_value(x: Vec<T>, y: Vec<T>, q_value: T) -> Result<Self> {
        Self::build(
            x,
            y,
            sys::DF_PP_DEFAULT as c_int,
            sys::DF_BC_Q_VAL as c_int,
            Some(vec![q_value]),
            sys::DF_NO_IC as c_int,
            None,
        )
    }

    /// A Subbotin quadratic spline through `(x[i], y[i])` with free ends and
    /// the knots at the midpoints of the cells, the library's own default
    /// (`t_0 = x_0`, `t_n = x_{n-1}`, `t_i = (x_i + x_{i-1}) / 2`). The
    /// knots are passed explicitly because oneMKL rejects the call without a
    /// knot array (-1012, measured).
    pub fn subbotin(x: Vec<T>, y: Vec<T>) -> Result<Self>
    where
        T: core::ops::Add<Output = T> + core::ops::Div<Output = T> + From<u8>,
    {
        let n = x.len();
        if n < 2 {
            return Err(Error::InvalidArgument(
                "Subbotin spline needs at least 2 knots",
            ));
        }
        let two = T::from(2u8);
        let mut knots = Vec::with_capacity(n + 1);
        knots.push(x[0]);
        for i in 1..n {
            knots.push((x[i] + x[i - 1]) / two);
        }
        knots.push(x[n - 1]);
        Self::subbotin_with_knots(x, y, knots)
    }

    /// A Subbotin quadratic spline with caller-chosen knots: `x.len() + 1`
    /// of them, `t_0 = x_0`, `t_n = x_{n-1}` and each interior `t_i` strictly
    /// inside `(x_{i-1}, x_i)`.
    pub fn subbotin_with_knots(x: Vec<T>, y: Vec<T>, knots: Vec<T>) -> Result<Self> {
        let n = x.len();
        if n < 2 {
            return Err(Error::InvalidArgument(
                "Subbotin spline needs at least 2 knots",
            ));
        }
        if knots.len() != n + 1 {
            return Err(Error::InvalidArgument(
                "Subbotin knot array must have x.len() + 1 entries",
            ));
        }
        let ends_ok = knots[0] == x[0] && knots[n] == x[n - 1];
        let inside_ok = (1..n).all(|i| knots[i] > x[i - 1] && knots[i] < x[i]);
        if !(ends_ok && inside_ok) {
            return Err(Error::InvalidArgument(
                "Subbotin knots need t_0 = x_0, t_n = x_{n-1} and t_i inside (x_{i-1}, x_i)",
            ));
        }
        Self::build(
            x,
            y,
            sys::DF_PP_SUBBOTIN as c_int,
            sys::DF_BC_FREE_END as c_int,
            None,
            sys::DF_IC_Q_KNOT as c_int,
            Some(knots),
        )
    }

    fn build(
        x: Vec<T>,
        y: Vec<T>,
        s_type: c_int,
        bc_type: c_int,
        bc: Option<Vec<T>>,
        ic_type: c_int,
        ic: Option<Vec<T>>,
    ) -> Result<Self> {
        if x.len() != y.len() {
            return Err(Error::InvalidArgument("x and y must have the same length"));
        }
        let n = x.len();
        if n < 2 {
            return Err(Error::InvalidArgument(
                "quadratic spline needs at least 2 knots",
            ));
        }
        let x_box = x.into_boxed_slice();
        let y_box = y.into_boxed_slice();
        // oneMKL writes order * (n - 1) coefficients for every spline type
        // except the Subbotin quadratic, which writes order * n (measured on
        // 2026.2 with `tools/size-probes/df_coeff_extent.c`; the Intel
        // reference says n - 1 for all).
        let cells = if s_type == sys::DF_PP_SUBBOTIN as c_int {
            n
        } else {
            n - 1
        };
        let coeffs_box: Box<[T]> = (0..cells * sys::DF_PP_QUADRATIC as usize)
            .map(|_| T::default())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let bc_box = bc.map(Vec::into_boxed_slice);
        let ic_box = ic.map(Vec::into_boxed_slice);
        let nx: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
        let mut task: DFTaskPtr = ptr::null_mut();
        check_df(unsafe {
            T::df_new_task_1d(
                &mut task,
                nx,
                x_box.as_ptr(),
                sys::DF_NON_UNIFORM_PARTITION as c_int,
                1,
                y_box.as_ptr(),
                sys::DF_NO_HINT as c_int,
            )
        })?;
        let this = Self {
            task,
            _x: x_box,
            _y: y_box,
            _coeffs: coeffs_box,
            _bc: bc_box,
            _ic: ic_box,
        };
        let bc_ptr = this._bc.as_ref().map_or(ptr::null(), |b| b.as_ptr());
        let ic_ptr = this._ic.as_ref().map_or(ptr::null(), |b| b.as_ptr());
        check_df(unsafe {
            T::df_edit_pp_spline_1d(
                this.task,
                sys::DF_PP_QUADRATIC as c_int,
                s_type,
                bc_type,
                bc_ptr,
                ic_type,
                ic_ptr,
                this._coeffs.as_ptr(),
                sys::DF_NO_HINT as c_int,
            )
        })?;
        check_df(unsafe {
            T::df_construct_1d(
                this.task,
                sys::DF_PP_SPLINE as c_int,
                sys::DF_METHOD_STD as c_int,
            )
        })?;
        Ok(this)
    }

    /// Evaluate the interpolant at `sites`.
    pub fn interpolate(&self, sites: &[T]) -> Result<Vec<T>> {
        let nsite: c_int = sites
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let mut out: Vec<T> = (0..sites.len()).map(|_| T::default()).collect();
        let dorder: [c_int; 1] = [1];
        check_df(unsafe {
            T::df_interpolate_1d(
                self.task,
                sys::DF_INTERP as c_int,
                sys::DF_METHOD_PP as c_int,
                nsite,
                sites.as_ptr(),
                sys::DF_NO_HINT as c_int,
                1,
                dorder.as_ptr(),
                ptr::null(),
                out.as_mut_ptr(),
                sys::DF_NO_HINT as c_int,
                ptr::null_mut(),
            )
        })?;
        Ok(out)
    }

    /// Integrate over `[left[i], right[i]]` for each pair.
    pub fn integrate(&self, left: &[T], right: &[T]) -> Result<Vec<T>> {
        if left.len() != right.len() {
            return Err(Error::InvalidArgument(
                "left and right limit slices must have the same length",
            ));
        }
        let nlim: c_int = left
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let mut out: Vec<T> = (0..left.len()).map(|_| T::default()).collect();
        check_df(unsafe {
            T::df_integrate_1d(
                self.task,
                sys::DF_METHOD_PP as c_int,
                nlim,
                left.as_ptr(),
                sys::DF_NO_HINT as c_int,
                right.as_ptr(),
                sys::DF_NO_HINT as c_int,
                ptr::null(),
                ptr::null(),
                out.as_mut_ptr(),
                sys::DF_NO_HINT as c_int,
            )
        })?;
        Ok(out)
    }

    /// Locate `sites` with a user-defined search call-back
    /// (`df?SearchCellsEx1D`). Same contract as
    /// `CubicSpline1d::find_cells_with_callback`: 1-based cells in the
    /// call-back, 0-based result, `exact[i] = 1` or
    /// [`DF_STATUS_EXACT_RESULT`] to have cells used as given, a negative
    /// status is an error, a positive one `Error::SearchCallbackStatus`, an exact
    /// cell that does not contain its site `Error::SearchCell`, a panic is resumed.
    /// Measured: `docs/DATA-FITTING-FACTS.md`.
    pub fn find_cells_with_callback(
        &self,
        sites: &[T],
        callback: SearchCallback<'_, T>,
    ) -> Result<Vec<usize>> {
        search_cells_impl(self.task, &self._x, sites, callback)
    }
}

impl<T: DataFittingScalar> Drop for QuadraticSpline1d<T> {
    fn drop(&mut self) {
        if !self.task.is_null() {
            unsafe {
                let _ = sys::dfDeleteTask(&mut self.task);
            }
        }
    }
}

#[inline]
fn integrate_ex_impl<'a, T: DataFittingScalar + Default>(
    task: DFTaskPtr,
    x: &'a [T],
    left: &[T],
    right: &[T],
    callbacks: IntegrateCallbacks<'a, T>,
) -> Result<Vec<T>> {
    if left.len() != right.len() {
        return Err(Error::InvalidArgument(
            "left and right limit slices must have the same length",
        ));
    }
    let nlim: c_int = left
        .len()
        .try_into()
        .map_err(|_| Error::DimensionOverflow)?;
    let mut out: Vec<T> = (0..left.len()).map(|_| T::default()).collect();
    let mut args = IntegrateExArgs {
        nlim,
        llim: left.as_ptr(),
        rlim: right.as_ptr(),
        r: out.as_mut_ptr(),
        x,
        callbacks,
        panic: None,
        search_error: None,
    };
    let status = unsafe { T::df_integrate_ex_1d(task, &mut args) };
    if let Some(payload) = args.panic.take() {
        resume_unwind(payload);
    }
    if let Some(e) = args.search_error.take() {
        return Err(e);
    }
    // A positive status is a warning from a call-back; the result is kept.
    if status < 0 {
        return Err(Error::VslStatus(status));
    }
    Ok(out)
}

fn search_cells_impl<'a, T: DataFittingScalar>(
    task: DFTaskPtr,
    x: &'a [T],
    sites: &[T],
    callback: SearchCallback<'a, T>,
) -> Result<Vec<usize>> {
    let _: c_int = sites
        .len()
        .try_into()
        .map_err(|_| Error::DimensionOverflow)?;
    let mut cells: Vec<c_int> = vec![0; sites.len()];
    let mut args = SearchExArgs {
        callback,
        x,
        panic: None,
        error: None,
    };
    let status = unsafe { T::df_search_cells_ex_1d(task, sites, &mut cells, &mut args) };
    if let Some(payload) = args.panic.take() {
        resume_unwind(payload);
    }
    if let Some(e) = args.take_error() {
        return Err(e);
    }
    if status < 0 {
        return Err(Error::VslStatus(status));
    }
    Ok(cells
        .into_iter()
        .map(|c| usize::try_from(c).unwrap_or(0).saturating_sub(1))
        .collect())
}

fn check_df(status: c_int) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::VslStatus(status))
    }
}
