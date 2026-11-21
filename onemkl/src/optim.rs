//! Nonlinear optimization — trust-region nonlinear least-squares
//! (TRNLS) and bound-constrained variant (TRNLSPBC).
//!
//! Solves problems of the form
//!
//! ```text
//! minimize  ½ ‖F(x)‖²    subject to  l ≤ x ≤ u  (optional)
//! ```
//!
//! where `F: ℝⁿ → ℝᵐ` is supplied by the caller along with its
//! Jacobian. The wrappers run oneMKL's reverse-communication loop
//! internally, calling the user's closures when MKL requests a
//! residual or Jacobian evaluation.
//!
//! Currently double-precision only.

use core::ffi::{c_int, c_void};
use core::ptr;
use core::slice;
use std::any::Any;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::Mutex;

use onemkl_sys as sys;

use crate::error::{Error, Result};

/// `m * n` with overflow reported as an error (a wrapped product would let an
/// undersized buffer pass the length check on 32-bit targets).
fn checked_len(m: usize, n: usize) -> Result<usize> {
    m.checked_mul(n).ok_or(Error::DimensionOverflow)
}

/// Tunable options for the trust-region NLLS solver.
#[derive(Debug, Clone, Copy)]
pub struct TrnlsOptions {
    /// Maximum number of outer iterations.
    pub max_iterations: i32,
    /// Maximum number of trial step refinements per iteration.
    pub max_trial_steps: i32,
    /// Initial trust region size.
    pub initial_trust_region: f64,
    /// Stopping tolerances `eps[0..=5]`. Each defaults to a reasonable
    /// value if left at `0.0`. See the oneMKL reference for the
    /// individual meanings.
    pub eps: [f64; 6],
}

impl Default for TrnlsOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            max_trial_steps: 100,
            initial_trust_region: 100.0,
            // 1e-5 is the value used in oneMKL's bundled examples;
            // tightening below ~1e-7 typically trips TR_INVALID_OPTION.
            eps: [1.0e-5; 6],
        }
    }
}

/// Outcome of a successful TRNLS solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrnlsResult {
    /// Number of outer iterations performed.
    pub iterations: i32,
    /// Stopping criterion code (`st_cr` from `dtrnlsp_get`).
    /// Mirrors oneMKL's six-valued enumeration.
    pub stopping_criterion: i32,
    /// `‖F(x₀)‖₂` — Euclidean norm of the residual at the initial
    /// guess.
    pub initial_residual_norm: f64,
    /// `‖F(x*)‖₂` — Euclidean norm of the residual at the returned
    /// solution.
    pub final_residual_norm: f64,
}

/// Unconstrained trust-region NLLS solver.
///
/// `x` (length `n`) is the initial guess on entry and the optimum on
/// return. `m` is the number of residual equations.
/// `residual(x, f)` writes `F(x)` into `f` (length `m`).
/// `jacobian(x, j)` writes the Jacobian into `j` (length `m * n`,
/// column-major).
pub fn solve_trnls<R, J>(
    n: usize,
    m: usize,
    x: &mut [f64],
    opts: TrnlsOptions,
    mut residual: R,
    mut jacobian: J,
) -> Result<TrnlsResult>
where
    R: FnMut(&[f64], &mut [f64]),
    J: FnMut(&[f64], &mut [f64]),
{
    if x.len() != n {
        return Err(Error::InvalidArgument(
            "x must have length n",
        ));
    }
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let m_i: c_int = m.try_into().map_err(|_| Error::DimensionOverflow)?;

    let mut handle: sys::_TRNSP_HANDLE_t = ptr::null_mut();
    let status = unsafe {
        sys::dtrnlsp_init(
            &mut handle,
            &n_i,
            &m_i,
            x.as_mut_ptr(),
            opts.eps.as_ptr(),
            &opts.max_iterations,
            &opts.max_trial_steps,
            &opts.initial_trust_region,
        )
    };
    if status != sys::TR_SUCCESS as c_int {
        return Err(Error::LapackComputationFailure { info: status });
    }

    let mut fvec = vec![0.0_f64; m];
    let mut fjac = vec![0.0_f64; checked_len(m, n)?];
    let mut rci: c_int = 0;

    let result = (|| -> Result<TrnlsResult> {
        loop {
            let status = unsafe {
                sys::dtrnlsp_solve(
                    &mut handle,
                    fvec.as_mut_ptr(),
                    fjac.as_mut_ptr(),
                    &mut rci,
                )
            };
            if status != sys::TR_SUCCESS as c_int {
                return Err(Error::LapackComputationFailure { info: status });
            }
            match rci {
                -1 | -2 | -3 | -4 | -5 | -6 => break,
                0 => break,
                1 => residual(x, &mut fvec),
                2 => jacobian(x, &mut fjac),
                other => {
                    return Err(Error::LapackComputationFailure { info: other });
                }
            }
        }

        let mut iters: c_int = 0;
        let mut st_cr: c_int = 0;
        let mut r1: f64 = 0.0;
        let mut r2: f64 = 0.0;
        let status = unsafe {
            sys::dtrnlsp_get(&mut handle, &mut iters, &mut st_cr, &mut r1, &mut r2)
        };
        if status != sys::TR_SUCCESS as c_int {
            return Err(Error::LapackComputationFailure { info: status });
        }
        Ok(TrnlsResult {
            iterations: iters,
            stopping_criterion: st_cr,
            initial_residual_norm: r1,
            final_residual_norm: r2,
        })
    })();

    unsafe {
        let _ = sys::dtrnlsp_delete(&mut handle);
    }
    result
}

/// Bound-constrained trust-region NLLS solver. Adds element-wise
/// lower / upper bound enforcement: `lower[i] ≤ x[i] ≤ upper[i]`.
pub fn solve_trnls_bounded<R, J>(
    n: usize,
    m: usize,
    x: &mut [f64],
    lower: &[f64],
    upper: &[f64],
    opts: TrnlsOptions,
    mut residual: R,
    mut jacobian: J,
) -> Result<TrnlsResult>
where
    R: FnMut(&[f64], &mut [f64]),
    J: FnMut(&[f64], &mut [f64]),
{
    if x.len() != n || lower.len() != n || upper.len() != n {
        return Err(Error::InvalidArgument(
            "x, lower, and upper must each have length n",
        ));
    }
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let m_i: c_int = m.try_into().map_err(|_| Error::DimensionOverflow)?;

    let mut handle: sys::_TRNSPBC_HANDLE_t = ptr::null_mut();
    let status = unsafe {
        sys::dtrnlspbc_init(
            &mut handle,
            &n_i,
            &m_i,
            x.as_mut_ptr(),
            lower.as_ptr(),
            upper.as_ptr(),
            opts.eps.as_ptr(),
            &opts.max_iterations,
            &opts.max_trial_steps,
            &opts.initial_trust_region,
        )
    };
    if status != sys::TR_SUCCESS as c_int {
        return Err(Error::LapackComputationFailure { info: status });
    }

    let mut fvec = vec![0.0_f64; m];
    let mut fjac = vec![0.0_f64; checked_len(m, n)?];
    let mut rci: c_int = 0;

    let result = (|| -> Result<TrnlsResult> {
        loop {
            let status = unsafe {
                sys::dtrnlspbc_solve(
                    &mut handle,
                    fvec.as_mut_ptr(),
                    fjac.as_mut_ptr(),
                    &mut rci,
                )
            };
            if status != sys::TR_SUCCESS as c_int {
                return Err(Error::LapackComputationFailure { info: status });
            }
            match rci {
                -1 | -2 | -3 | -4 | -5 | -6 => break,
                0 => break,
                1 => residual(x, &mut fvec),
                2 => jacobian(x, &mut fjac),
                other => {
                    return Err(Error::LapackComputationFailure { info: other });
                }
            }
        }

        let mut iters: c_int = 0;
        let mut st_cr: c_int = 0;
        let mut r1: f64 = 0.0;
        let mut r2: f64 = 0.0;
        let status = unsafe {
            sys::dtrnlspbc_get(&mut handle, &mut iters, &mut st_cr, &mut r1, &mut r2)
        };
        if status != sys::TR_SUCCESS as c_int {
            return Err(Error::LapackComputationFailure { info: status });
        }
        Ok(TrnlsResult {
            iterations: iters,
            stopping_criterion: st_cr,
            initial_residual_norm: r1,
            final_residual_norm: r2,
        })
    })();

    unsafe {
        let _ = sys::dtrnlspbc_delete(&mut handle);
    }
    result
}

/// Numerical Jacobian via forward differences. `residual(x, f)` is
/// called repeatedly with perturbed `x`; the resulting `m × n`
/// (column-major) Jacobian is written into `jac`.
///
/// `step` is the perturbation step size; pass `1e-6` for a typical
/// double-precision balance between truncation and round-off error.
pub fn numerical_jacobian<R>(
    n: usize,
    m: usize,
    x: &[f64],
    step: f64,
    jac: &mut [f64],
    mut residual: R,
) -> Result<()>
where
    R: FnMut(&[f64], &mut [f64]),
{
    if x.len() != n {
        return Err(Error::InvalidArgument(
            "x must have length n",
        ));
    }
    if jac.len() < checked_len(m, n)? {
        return Err(Error::InvalidArgument(
            "jac buffer is smaller than m * n",
        ));
    }
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let m_i: c_int = m.try_into().map_err(|_| Error::DimensionOverflow)?;

    // MKL's RCI loop perturbs `x` in place and the closure reads it between
    // calls, so MKL must get an owned, mutable buffer (writing through a
    // pointer derived from the shared `&[f64]` would be undefined behaviour).
    let mut xv = x.to_vec();
    let mut handle = JacobiHandle(ptr::null_mut());
    let status = unsafe {
        sys::djacobi_init(
            &mut handle.0, &n_i, &m_i,
            xv.as_mut_ptr(), jac.as_mut_ptr(), &step,
        )
    };
    if status != sys::TR_SUCCESS as c_int {
        return Err(Error::LapackComputationFailure { info: status });
    }

    // Working buffers MKL will manage internally; we provide the
    // residual closure via the RCI loop.
    let mut f1 = vec![0.0_f64; m];
    let mut f2 = vec![0.0_f64; m];
    let mut rci: c_int = 0;

    // `handle` is an RAII guard: a panic in `residual` unwinds through here and
    // still frees the MKL handle.
    let result = (|| -> Result<()> {
        loop {
            let status = unsafe {
                sys::djacobi_solve(&mut handle.0, f1.as_mut_ptr(), f2.as_mut_ptr(), &mut rci)
            };
            if status != sys::TR_SUCCESS as c_int {
                return Err(Error::LapackComputationFailure { info: status });
            }
            match rci {
                0 => break,
                1 => {
                    // Need F(x) at the current sample — MKL has
                    // already adjusted x internally; we call the
                    // user's residual on whatever they passed.
                    residual(&xv, &mut f1);
                }
                2 => {
                    residual(&xv, &mut f2);
                }
                other if other < 0 => {
                    return Err(Error::LapackComputationFailure { info: other });
                }
                _ => {}
            }
        }
        Ok(())
    })();

    drop(handle);
    result
}

/// Owns an initialised `djacobi` RCI handle; `djacobi_delete` runs on drop,
/// including during unwinding.
struct JacobiHandle(sys::_JACOBIMATRIX_HANDLE_t);

impl Drop for JacobiHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let _ = sys::djacobi_delete(&mut self.0);
            }
        }
    }
}

/// Function-pointer type for the direct-callback form of `djacobi`.
/// MKL invokes this once per Jacobian column, passing pointers to
/// dimensions, the perturbed `x`, and the output residual.
pub type DjacobiCallback = unsafe extern "C" fn(
    n: *mut core::ffi::c_int,
    m: *mut core::ffi::c_int,
    x: *mut f64,
    f: *mut f64,
);

/// Compute a numerical Jacobian using oneMKL's direct-callback form
/// (`djacobi`). Unlike [`numerical_jacobian`], which drives MKL's
/// reverse-communication interface from a Rust closure, this variant
/// accepts a raw `extern "C"` function pointer that MKL calls
/// internally — useful when the residual is implemented in C / C++ /
/// Fortran or already exposed as an FFI-compatible function.
///
/// `jac` must be at least `m * n` elements (column-major). `step` is
/// the perturbation size for forward-difference Jacobian estimation.
///
/// # Safety
///
/// `fcn` must remain valid for the duration of this call and must
/// not unwind across the FFI boundary.
pub unsafe fn djacobi_with_callback(
    n: usize,
    m: usize,
    x: &[f64],
    step: f64,
    jac: &mut [f64],
    fcn: DjacobiCallback,
) -> Result<()> {
    if x.len() != n {
        return Err(Error::InvalidArgument("x must have length n"));
    }
    if jac.len() < checked_len(m, n)? {
        return Err(Error::InvalidArgument(
            "jac buffer is smaller than m * n",
        ));
    }
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let m_i: c_int = m.try_into().map_err(|_| Error::DimensionOverflow)?;
    // MKL temporarily perturbs each element of `x` in place, so it gets an
    // owned buffer (writing through a pointer derived from the shared `&[f64]`
    // would be undefined behaviour).
    let mut xv = x.to_vec();
    let mut step_mut = step;
    let status = unsafe {
        sys::djacobi(
            Some(fcn),
            &n_i,
            &m_i,
            jac.as_mut_ptr(),
            xv.as_mut_ptr(),
            &mut step_mut,
        )
    };
    if status != sys::TR_SUCCESS as c_int {
        return Err(Error::LapackComputationFailure { info: status });
    }
    Ok(())
}

struct JacobixState<'a> {
    n: usize,
    m: usize,
    f: &'a mut dyn FnMut(&[f64], &mut [f64]),
    panic: Option<Box<dyn Any + Send>>,
}

unsafe extern "C" fn jacobix_trampoline(
    _m: *mut c_int,
    _n: *mut c_int,
    x: *mut f64,
    f: *mut f64,
    user: *mut c_void,
) {
    // SAFETY: `user` is the `Mutex<JacobixState>` that
    // `numerical_jacobian_central` passed to `djacobix` and keeps alive for the
    // whole call. Only a shared reference is formed; the mutex serialises
    // callbacks, so a threaded MKL layer cannot alias the `FnMut` state.
    let cell = unsafe { &*(user as *const Mutex<JacobixState<'_>>) };
    let mut guard = cell.lock().unwrap_or_else(|e| e.into_inner());
    let st = &mut *guard;
    if st.panic.is_some() {
        return;
    }
    // The dimensions come from our own state, not from MKL's (Fortran-order)
    // arguments. MKL passes `x` of length `n` and `f` of length `m`.
    let xs = unsafe { slice::from_raw_parts(x, st.n) };
    // `f` is an output buffer MKL may not have initialised: zero it through the
    // raw pointer before forming a `&mut [f64]` over it.
    unsafe { ptr::write_bytes(f, 0, st.m) };
    let fs = unsafe { slice::from_raw_parts_mut(f, st.m) };
    if let Err(p) = catch_unwind(AssertUnwindSafe(|| (st.f)(xs, fs))) {
        st.panic = Some(p);
    }
}

/// Numerical Jacobian of `residual: ℝⁿ → ℝᵐ` by central differences
/// (`djacobix`). `residual(x, f)` writes `F(x)` into `f` (length `m`) and
/// is called with perturbed copies of `x` (length `n`). The `m × n`
/// Jacobian is written into `jac` column-major (column `j` = ∂F/∂xⱼ);
/// measured in `tests/optim_jacobix.rs`.
///
/// `eps` is the requested precision of the Jacobian (`1e-6` is typical).
/// A panic in `residual` is caught inside the callback and re-raised here
/// once MKL has returned, never unwound through MKL. `residual` must be
/// `Send` and is serialised by a mutex, because a threaded oneMKL layer may
/// call back from worker threads.
pub fn numerical_jacobian_central<R>(
    n: usize,
    m: usize,
    x: &[f64],
    eps: f64,
    jac: &mut [f64],
    mut residual: R,
) -> Result<()>
where
    R: FnMut(&[f64], &mut [f64]) + Send,
{
    if x.len() != n {
        return Err(Error::InvalidArgument("x must have length n"));
    }
    if jac.len() < checked_len(m, n)? {
        return Err(Error::InvalidArgument("jac buffer is smaller than m * n"));
    }
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Error::InvalidArgument("eps must be finite and positive"));
    }
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let m_i: c_int = m.try_into().map_err(|_| Error::DimensionOverflow)?;

    // MKL perturbs and restores `x` in place, so give it an owned copy.
    let mut xv = x.to_vec();
    let mut eps_mut = eps;
    let state = Mutex::new(JacobixState { n, m, f: &mut residual, panic: None });
    let status = unsafe {
        sys::djacobix(
            Some(jacobix_trampoline),
            &n_i,
            &m_i,
            jac.as_mut_ptr(),
            xv.as_mut_ptr(),
            &mut eps_mut,
            &state as *const Mutex<JacobixState<'_>> as *mut c_void,
        )
    };
    let panic = state.into_inner().unwrap_or_else(|e| e.into_inner()).panic.take();
    if let Some(p) = panic {
        resume_unwind(p);
    }
    if status != sys::TR_SUCCESS as c_int {
        return Err(Error::LapackComputationFailure { info: status });
    }
    Ok(())
}

