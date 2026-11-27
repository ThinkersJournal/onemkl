// SPDX-License-Identifier: MIT OR Apache-2.0
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

use core::ffi::c_void;
use core::ptr;
use core::slice;
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Mutex;

use onemkl_sys::{self as sys, MKL_INT};

use crate::error::{Error, Result};
use crate::util::status_i32;

/// `m * n` with overflow reported as an error (a wrapped product would let an
/// undersized buffer pass the length check on 32-bit targets).
fn checked_len(m: usize, n: usize) -> Result<usize> {
    m.checked_mul(n).ok_or(Error::DimensionOverflow)
}

/// Tunable options for the trust-region NLLS solver.
#[derive(Debug, Clone, Copy)]
pub struct TrnlsOptions {
    /// Maximum number of outer iterations.
    pub max_iterations: MKL_INT,
    /// Maximum number of trial step refinements per iteration.
    pub max_trial_steps: MKL_INT,
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
    pub iterations: MKL_INT,
    /// Stopping criterion code (`st_cr` from `dtrnlsp_get`).
    /// Mirrors oneMKL's six-valued enumeration.
    pub stopping_criterion: MKL_INT,
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
        return Err(Error::InvalidArgument("x must have length n"));
    }
    let n_i: MKL_INT = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let m_i: MKL_INT = m.try_into().map_err(|_| Error::DimensionOverflow)?;

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
    if status != sys::TR_SUCCESS as MKL_INT {
        return Err(Error::LapackComputationFailure {
            info: status_i32(status),
        });
    }

    let mut fvec = vec![0.0_f64; m];
    let mut fjac = vec![0.0_f64; checked_len(m, n)?];
    let mut rci: MKL_INT = 0;

    let result = (|| -> Result<TrnlsResult> {
        loop {
            let status = unsafe {
                sys::dtrnlsp_solve(&mut handle, fvec.as_mut_ptr(), fjac.as_mut_ptr(), &mut rci)
            };
            if status != sys::TR_SUCCESS as MKL_INT {
                return Err(Error::LapackComputationFailure {
                    info: status_i32(status),
                });
            }
            match rci {
                -6..=-1 => break,
                0 => break,
                1 => residual(x, &mut fvec),
                2 => jacobian(x, &mut fjac),
                other => {
                    return Err(Error::LapackComputationFailure {
                        info: status_i32(other),
                    });
                }
            }
        }

        let mut iters: MKL_INT = 0;
        let mut st_cr: MKL_INT = 0;
        let mut r1: f64 = 0.0;
        let mut r2: f64 = 0.0;
        let status =
            unsafe { sys::dtrnlsp_get(&mut handle, &mut iters, &mut st_cr, &mut r1, &mut r2) };
        if status != sys::TR_SUCCESS as MKL_INT {
            return Err(Error::LapackComputationFailure {
                info: status_i32(status),
            });
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

/// Element-wise bounds `lower[i] ≤ x[i] ≤ upper[i]` for
/// [`solve_trnls_bounded`]; both slices must have length `n`.
#[derive(Debug, Clone, Copy)]
pub struct TrnlsBounds<'a> {
    /// Lower bounds, length `n`.
    pub lower: &'a [f64],
    /// Upper bounds, length `n`.
    pub upper: &'a [f64],
}

/// Bound-constrained trust-region NLLS solver. Adds element-wise
/// lower / upper bound enforcement: `lower[i] ≤ x[i] ≤ upper[i]`.
pub fn solve_trnls_bounded<R, J>(
    n: usize,
    m: usize,
    x: &mut [f64],
    bounds: TrnlsBounds<'_>,
    opts: TrnlsOptions,
    mut residual: R,
    mut jacobian: J,
) -> Result<TrnlsResult>
where
    R: FnMut(&[f64], &mut [f64]),
    J: FnMut(&[f64], &mut [f64]),
{
    let TrnlsBounds { lower, upper } = bounds;
    if x.len() != n || lower.len() != n || upper.len() != n {
        return Err(Error::InvalidArgument(
            "x, lower, and upper must each have length n",
        ));
    }
    let n_i: MKL_INT = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let m_i: MKL_INT = m.try_into().map_err(|_| Error::DimensionOverflow)?;

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
    if status != sys::TR_SUCCESS as MKL_INT {
        return Err(Error::LapackComputationFailure {
            info: status_i32(status),
        });
    }

    let mut fvec = vec![0.0_f64; m];
    let mut fjac = vec![0.0_f64; checked_len(m, n)?];
    let mut rci: MKL_INT = 0;

    let result = (|| -> Result<TrnlsResult> {
        loop {
            let status = unsafe {
                sys::dtrnlspbc_solve(&mut handle, fvec.as_mut_ptr(), fjac.as_mut_ptr(), &mut rci)
            };
            if status != sys::TR_SUCCESS as MKL_INT {
                return Err(Error::LapackComputationFailure {
                    info: status_i32(status),
                });
            }
            match rci {
                -6..=-1 => break,
                0 => break,
                1 => residual(x, &mut fvec),
                2 => jacobian(x, &mut fjac),
                other => {
                    return Err(Error::LapackComputationFailure {
                        info: status_i32(other),
                    });
                }
            }
        }

        let mut iters: MKL_INT = 0;
        let mut st_cr: MKL_INT = 0;
        let mut r1: f64 = 0.0;
        let mut r2: f64 = 0.0;
        let status =
            unsafe { sys::dtrnlspbc_get(&mut handle, &mut iters, &mut st_cr, &mut r1, &mut r2) };
        if status != sys::TR_SUCCESS as MKL_INT {
            return Err(Error::LapackComputationFailure {
                info: status_i32(status),
            });
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
        return Err(Error::InvalidArgument("x must have length n"));
    }
    if jac.len() < checked_len(m, n)? {
        return Err(Error::InvalidArgument("jac buffer is smaller than m * n"));
    }
    let n_i: MKL_INT = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let m_i: MKL_INT = m.try_into().map_err(|_| Error::DimensionOverflow)?;

    // MKL's RCI loop perturbs `x` in place and the closure reads it between
    // calls, so MKL must get an owned, mutable buffer (writing through a
    // pointer derived from the shared `&[f64]` would be undefined behaviour).
    let mut xv = x.to_vec();
    let mut handle = JacobiHandle(ptr::null_mut());
    let status = unsafe {
        sys::djacobi_init(
            &mut handle.0,
            &n_i,
            &m_i,
            xv.as_mut_ptr(),
            jac.as_mut_ptr(),
            &step,
        )
    };
    if status != sys::TR_SUCCESS as MKL_INT {
        return Err(Error::LapackComputationFailure {
            info: status_i32(status),
        });
    }

    // Working buffers MKL will manage internally; we provide the
    // residual closure via the RCI loop.
    let mut f1 = vec![0.0_f64; m];
    let mut f2 = vec![0.0_f64; m];
    let mut rci: MKL_INT = 0;

    // `handle` is an RAII guard: a panic in `residual` unwinds through here and
    // still frees the MKL handle.
    let result = (|| -> Result<()> {
        loop {
            let status = unsafe {
                sys::djacobi_solve(&mut handle.0, f1.as_mut_ptr(), f2.as_mut_ptr(), &mut rci)
            };
            if status != sys::TR_SUCCESS as MKL_INT {
                return Err(Error::LapackComputationFailure {
                    info: status_i32(status),
                });
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
                    return Err(Error::LapackComputationFailure {
                        info: status_i32(other),
                    });
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
pub type DjacobiCallback =
    unsafe extern "C" fn(n: *mut MKL_INT, m: *mut MKL_INT, x: *mut f64, f: *mut f64);

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
        return Err(Error::InvalidArgument("jac buffer is smaller than m * n"));
    }
    let n_i: MKL_INT = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let m_i: MKL_INT = m.try_into().map_err(|_| Error::DimensionOverflow)?;
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
    if status != sys::TR_SUCCESS as MKL_INT {
        return Err(Error::LapackComputationFailure {
            info: status_i32(status),
        });
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
    _m: *mut MKL_INT,
    _n: *mut MKL_INT,
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
    let n_i: MKL_INT = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let m_i: MKL_INT = m.try_into().map_err(|_| Error::DimensionOverflow)?;

    // MKL perturbs and restores `x` in place, so give it an owned copy.
    let mut xv = x.to_vec();
    let mut eps_mut = eps;
    let state = Mutex::new(JacobixState {
        n,
        m,
        f: &mut residual,
        panic: None,
    });
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
    let panic = state
        .into_inner()
        .unwrap_or_else(|e| e.into_inner())
        .panic
        .take();
    if let Some(p) = panic {
        resume_unwind(p);
    }
    if status != sys::TR_SUCCESS as MKL_INT {
        return Err(Error::LapackComputationFailure {
            info: status_i32(status),
        });
    }
    Ok(())
}

/// Next action requested by a [`TrnlsSession`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrnlsStep {
    /// Evaluate `F(x)` at [`TrnlsSession::x`] into [`TrnlsSession::residual_mut`], then call `step` again.
    NeedResidual,
    /// Evaluate the Jacobian at [`TrnlsSession::x`] into [`TrnlsSession::jacobian_mut`]
    /// (column-major `m x n`), then call `step` again.
    NeedJacobian,
    /// The solver stopped; the optimum is in [`TrnlsSession::x`].
    Done(TrnlsResult),
}

/// Result of [`TrnlsSession::check`] (`dtrnlsp_check`): the raw six-entry
/// `info` array. All zero means the Jacobian, residual and tolerances are
/// valid; `info[0]` names the first invalid object and the others locate the
/// offending entries (see the oneMKL reference).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrnlsCheck {
    /// Raw `info` array returned by `dtrnlsp_check`.
    pub info: [MKL_INT; 6],
}

impl TrnlsCheck {
    /// `true` when every `info` entry is zero.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.info.iter().all(|&i| i == 0)
    }
}

/// Step-by-step driver of the unconstrained trust-region solver: the caller
/// owns the reverse-communication loop (unlike [`solve_trnls`], which runs it
/// with closures). The session owns `x`, the residual and the Jacobian
/// buffers, since MKL keeps pointers to `x` for the life of the handle.
pub struct TrnlsSession {
    /// Boxed so the handle's address survives moves of the session.
    handle: Box<sys::_TRNSP_HANDLE_t>,
    n: usize,
    m: usize,
    x: Vec<f64>,
    fvec: Vec<f64>,
    fjac: Vec<f64>,
    opts: Box<TrnlsOptions>,
    /// Reverse-communication state: MKL reads it back on the next call, so it must persist.
    rci: MKL_INT,
    done: Option<TrnlsResult>,
}

impl TrnlsSession {
    /// Start a session with `n` variables, `m` residuals and initial guess `x0`.
    pub fn new(n: usize, m: usize, x0: &[f64], opts: TrnlsOptions) -> Result<Self> {
        if x0.len() != n {
            return Err(Error::InvalidArgument("x0 must have length n"));
        }
        let n_i: MKL_INT = n.try_into().map_err(|_| Error::DimensionOverflow)?;
        let m_i: MKL_INT = m.try_into().map_err(|_| Error::DimensionOverflow)?;
        let mut s = Self {
            handle: Box::new(ptr::null_mut()),
            n,
            m,
            x: x0.to_vec(),
            fvec: vec![0.0; m],
            fjac: vec![0.0; checked_len(m, n)?],
            opts: Box::new(opts),
            rci: 0,
            done: None,
        };
        let status = unsafe {
            sys::dtrnlsp_init(
                &mut *s.handle,
                &n_i,
                &m_i,
                s.x.as_mut_ptr(),
                s.opts.eps.as_ptr(),
                &s.opts.max_iterations,
                &s.opts.max_trial_steps,
                &s.opts.initial_trust_region,
            )
        };
        if status != sys::TR_SUCCESS as MKL_INT {
            // No handle was created: Drop must not delete it.
            *s.handle = ptr::null_mut();
            return Err(Error::LapackComputationFailure {
                info: status_i32(status),
            });
        }
        Ok(s)
    }

    /// Current iterate; evaluate residual / Jacobian here.
    #[must_use]
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// Residual buffer (length `m`) to fill on [`TrnlsStep::NeedResidual`].
    pub fn residual_mut(&mut self) -> &mut [f64] {
        &mut self.fvec
    }

    /// Jacobian buffer (column-major `m x n`) to fill on [`TrnlsStep::NeedJacobian`].
    pub fn jacobian_mut(&mut self) -> &mut [f64] {
        &mut self.fjac
    }

    /// Validate the Jacobian, residual and tolerances currently in the
    /// session's buffers (`dtrnlsp_check`). Call after filling both buffers.
    pub fn check(&mut self) -> Result<TrnlsCheck> {
        let n_i = self.n as MKL_INT;
        let m_i = self.m as MKL_INT;
        let mut info = [0 as MKL_INT; 6];
        let status = unsafe {
            sys::dtrnlsp_check(
                &mut *self.handle,
                &n_i,
                &m_i,
                self.fjac.as_ptr(),
                self.fvec.as_ptr(),
                self.opts.eps.as_ptr(),
                info.as_mut_ptr(),
            )
        };
        if status != sys::TR_SUCCESS as MKL_INT {
            return Err(Error::LapackComputationFailure {
                info: status_i32(status),
            });
        }
        Ok(TrnlsCheck { info })
    }

    /// Advance the solver one reverse-communication step. After `Done`,
    /// further calls return the same result.
    pub fn step(&mut self) -> Result<TrnlsStep> {
        if let Some(r) = self.done {
            return Ok(TrnlsStep::Done(r));
        }
        let status = unsafe {
            sys::dtrnlsp_solve(
                &mut *self.handle,
                self.fvec.as_mut_ptr(),
                self.fjac.as_mut_ptr(),
                &mut self.rci,
            )
        };
        if status != sys::TR_SUCCESS as MKL_INT {
            return Err(Error::LapackComputationFailure {
                info: status_i32(status),
            });
        }
        match self.rci {
            1 => Ok(TrnlsStep::NeedResidual),
            2 => Ok(TrnlsStep::NeedJacobian),
            -6..=0 => {
                let (mut iters, mut st_cr, mut r1, mut r2) =
                    (0 as MKL_INT, 0 as MKL_INT, 0.0_f64, 0.0_f64);
                let status = unsafe {
                    sys::dtrnlsp_get(&mut *self.handle, &mut iters, &mut st_cr, &mut r1, &mut r2)
                };
                if status != sys::TR_SUCCESS as MKL_INT {
                    return Err(Error::LapackComputationFailure {
                        info: status_i32(status),
                    });
                }
                let r = TrnlsResult {
                    iterations: iters,
                    stopping_criterion: st_cr,
                    initial_residual_norm: r1,
                    final_residual_norm: r2,
                };
                self.done = Some(r);
                Ok(TrnlsStep::Done(r))
            }
            other => Err(Error::LapackComputationFailure {
                info: status_i32(other),
            }),
        }
    }
}

impl Drop for TrnlsSession {
    fn drop(&mut self) {
        if !(*self.handle).is_null() {
            unsafe {
                let _ = sys::dtrnlsp_delete(&mut *self.handle);
            }
        }
    }
}

/// Step-by-step driver of the bound-constrained trust-region solver
/// (`dtrnlspbc_*`); the bounded counterpart of [`TrnlsSession`]. The session
/// owns `x`, the bounds and the residual / Jacobian buffers.
pub struct TrnlsBoundedSession {
    /// Boxed so the handle's address survives moves of the session.
    handle: Box<sys::_TRNSPBC_HANDLE_t>,
    n: usize,
    m: usize,
    x: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    fvec: Vec<f64>,
    fjac: Vec<f64>,
    opts: Box<TrnlsOptions>,
    /// Reverse-communication state: MKL reads it back on the next call.
    rci: MKL_INT,
    done: Option<TrnlsResult>,
}

impl TrnlsBoundedSession {
    /// Start a session with `n` variables, `m` residuals, initial guess `x0`
    /// and element-wise bounds `lower[i] <= x[i] <= upper[i]`.
    pub fn new(
        n: usize,
        m: usize,
        x0: &[f64],
        lower: &[f64],
        upper: &[f64],
        opts: TrnlsOptions,
    ) -> Result<Self> {
        if x0.len() != n || lower.len() != n || upper.len() != n {
            return Err(Error::InvalidArgument(
                "x0, lower, and upper must each have length n",
            ));
        }
        let n_i: MKL_INT = n.try_into().map_err(|_| Error::DimensionOverflow)?;
        let m_i: MKL_INT = m.try_into().map_err(|_| Error::DimensionOverflow)?;
        let mut s = Self {
            handle: Box::new(ptr::null_mut()),
            n,
            m,
            x: x0.to_vec(),
            lower: lower.to_vec(),
            upper: upper.to_vec(),
            fvec: vec![0.0; m],
            fjac: vec![0.0; checked_len(m, n)?],
            opts: Box::new(opts),
            rci: 0,
            done: None,
        };
        let status = unsafe {
            sys::dtrnlspbc_init(
                &mut *s.handle,
                &n_i,
                &m_i,
                s.x.as_mut_ptr(),
                s.lower.as_ptr(),
                s.upper.as_ptr(),
                s.opts.eps.as_ptr(),
                &s.opts.max_iterations,
                &s.opts.max_trial_steps,
                &s.opts.initial_trust_region,
            )
        };
        if status != sys::TR_SUCCESS as MKL_INT {
            *s.handle = ptr::null_mut();
            return Err(Error::LapackComputationFailure {
                info: status_i32(status),
            });
        }
        Ok(s)
    }

    /// Current iterate; evaluate residual / Jacobian here.
    #[must_use]
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// Residual buffer (length `m`) to fill on [`TrnlsStep::NeedResidual`].
    pub fn residual_mut(&mut self) -> &mut [f64] {
        &mut self.fvec
    }

    /// Jacobian buffer (column-major `m x n`) to fill on [`TrnlsStep::NeedJacobian`].
    pub fn jacobian_mut(&mut self) -> &mut [f64] {
        &mut self.fjac
    }

    /// Validate the buffers and bounds (`dtrnlspbc_check`); see [`TrnlsCheck`].
    pub fn check(&mut self) -> Result<TrnlsCheck> {
        let n_i = self.n as MKL_INT;
        let m_i = self.m as MKL_INT;
        let mut info = [0 as MKL_INT; 6];
        let status = unsafe {
            sys::dtrnlspbc_check(
                &mut *self.handle,
                &n_i,
                &m_i,
                self.fjac.as_ptr(),
                self.fvec.as_ptr(),
                self.lower.as_ptr(),
                self.upper.as_ptr(),
                self.opts.eps.as_ptr(),
                info.as_mut_ptr(),
            )
        };
        if status != sys::TR_SUCCESS as MKL_INT {
            return Err(Error::LapackComputationFailure {
                info: status_i32(status),
            });
        }
        Ok(TrnlsCheck { info })
    }

    /// Advance the solver one reverse-communication step. After `Done`,
    /// further calls return the same result.
    pub fn step(&mut self) -> Result<TrnlsStep> {
        if let Some(r) = self.done {
            return Ok(TrnlsStep::Done(r));
        }
        let status = unsafe {
            sys::dtrnlspbc_solve(
                &mut *self.handle,
                self.fvec.as_mut_ptr(),
                self.fjac.as_mut_ptr(),
                &mut self.rci,
            )
        };
        if status != sys::TR_SUCCESS as MKL_INT {
            return Err(Error::LapackComputationFailure {
                info: status_i32(status),
            });
        }
        match self.rci {
            1 => Ok(TrnlsStep::NeedResidual),
            2 => Ok(TrnlsStep::NeedJacobian),
            -6..=0 => {
                let (mut iters, mut st_cr, mut r1, mut r2) =
                    (0 as MKL_INT, 0 as MKL_INT, 0.0_f64, 0.0_f64);
                let status = unsafe {
                    sys::dtrnlspbc_get(&mut *self.handle, &mut iters, &mut st_cr, &mut r1, &mut r2)
                };
                if status != sys::TR_SUCCESS as MKL_INT {
                    return Err(Error::LapackComputationFailure {
                        info: status_i32(status),
                    });
                }
                let r = TrnlsResult {
                    iterations: iters,
                    stopping_criterion: st_cr,
                    initial_residual_norm: r1,
                    final_residual_norm: r2,
                };
                self.done = Some(r);
                Ok(TrnlsStep::Done(r))
            }
            other => Err(Error::LapackComputationFailure {
                info: status_i32(other),
            }),
        }
    }
}

impl Drop for TrnlsBoundedSession {
    fn drop(&mut self) {
        if !(*self.handle).is_null() {
            unsafe {
                let _ = sys::dtrnlspbc_delete(&mut *self.handle);
            }
        }
    }
}
