// SPDX-License-Identifier: MIT OR Apache-2.0
//! Summary statistics — `vsl?SS*` task family.
//!
//! Computes per-variable statistics (mean, variance, min, max, sum)
//! over a `(p × n)` data matrix where `p` is the number of variables
//! and `n` is the number of observations.
//!
//! Data is row-major in the natural sense: variable `i`'s
//! observations live at `data[i * n + 0 .. i * n + n]`.
//!
//! ```no_run
//! use onemkl::rng::summary_stats::SummaryStats;
//!
//! // 1 variable, 5 observations.
//! let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
//! let mut ss = SummaryStats::<f64>::new(&data, 1, 5).unwrap();
//! let mean = ss.mean().unwrap();
//! assert!((mean[0] - 3.0).abs() < 1e-12);
//! ```

use core::ffi::c_int;
use core::marker::PhantomData;
use core::ptr;

use onemkl_sys::{self as sys, VSLSSTaskPtr};

use crate::error::{Error, Result};

/// Real scalar types supported by the summary-statistics task family.
#[allow(missing_docs)]
pub trait SsScalar: Copy + Default + 'static {
    /// The bit pattern oneMKL's `VSL_SS_SNAN` / `VSL_SS_DNAN` uses to mark a missing observation (a quiet NaN with the sign bit set: `0xFFC00000` for `f32`, `0xFFF8000000000000` for `f64`). oneMKL compares bit patterns, so an ordinary `NAN` is not recognised.
    const MISSING: Self;
    /// Whether `self` has exactly the bit pattern of [`Self::MISSING`].
    fn is_missing(self) -> bool;
    /// Lossy conversion from `f64` (parameter counts).
    fn from_f64(v: f64) -> Self;
    /// `self` as an `f64` when it is a usable observation weight (finite and not negative), else `None`.
    fn weight_value(self) -> Option<f64>;

    /// Raw `ss_new_task` call.
    ///
    /// # Safety
    ///
    /// `task` must be valid for a write of one `VSLSSTaskPtr`. `p`, `n` and `xstorage` must each point to one readable `c_int` (dimension `p >= 1`, observation count `n >= 1`, and a `VSL_SS_MATRIX_STORAGE_*` constant); they are read during the call.
    /// `x` must point to the `p x n` observation matrix, `p * n` elements, stored per `*xstorage`. `w` may be null (equal weights) or point to `n` weights. `indices` may be null or point to `p` entries of 0/1 selecting which variables are used.
    /// **Pointer retention:** the task stores the pointers it is given (it does not copy the data). Each buffer must stay allocated, at a stable address, and not be mutated through Rust references (outputs: not accessed at all) from this call until the task has been deleted or the pointer replaced, including across every `ss_compute` call.
    /// On a non-zero return `*task` must not be used. A non-zero return is the VSL status. The MKL library must be linked and usable.
    /// **Verified** by guard-page probe (smallest buffer for which the call did not fault: `x`: p * n and `w`: n elements, row storage; oneMKL 2026.2, Linux x86_64). Extents not listed as verified in this block are unverified (from MKL documentation).
    unsafe fn ss_new_task(
        task: *mut VSLSSTaskPtr,
        p: *const c_int,
        n: *const c_int,
        xstorage: *const c_int,
        x: *const Self,
        w: *const Self,
        indices: *const c_int,
    ) -> c_int;

    /// Raw `ss_edit_task` call.
    ///
    /// # Safety
    ///
    /// `task` must be a live `VSLSSTaskPtr` from `ss_new_task` that has not been deleted, and must not be used from another thread during the call.
    /// `parameter` must be a `VSL_SS_ED_*` constant that expects a `Self`-typed address (or the matching integer type, which this method does not provide); `address` must point to storage of the extent MKL documents for that parameter (for example `p` elements for a per-variable vector, `p * p` for a covariance matrix).
    /// **Pointer retention:** the task stores the pointers it is given (it does not copy the data). Each buffer must stay allocated, at a stable address, and not be mutated through Rust references (outputs: not accessed at all) from this call until the task has been deleted or the pointer replaced, including across every `ss_compute` call.
    /// A non-zero return is the VSL status. The MKL library must be linked and usable.
    /// Extents not listed as verified in this block are unverified (from MKL documentation).
    unsafe fn ss_edit_task(task: VSLSSTaskPtr, parameter: c_int, address: *const Self) -> c_int;

    /// Raw `ss_edit_moments` call.
    ///
    /// # Safety
    ///
    /// `task` must be a live `VSLSSTaskPtr` from `ss_new_task` that has not been deleted, and must not be used from another thread during the call.
    /// `mean`, `r2m`, `r3m`, `r4m`, `c2m`, `c3m` and `c4m` are output vectors of `p` elements each (the task dimension); each pointer must be non-null and valid for writes of `p` elements when the corresponding estimate is requested in `ss_compute`, and the buffers must not overlap each other or the task data.
    /// **Pointer retention:** the task stores the pointers it is given (it does not copy the data). Each buffer must stay allocated, at a stable address, and not be mutated through Rust references (outputs: not accessed at all) from this call until the task has been deleted or the pointer replaced, including across every `ss_compute` call.
    /// A non-zero return is the VSL status. The MKL library must be linked and usable.
    /// **Verified** by guard-page probe (smallest buffer for which the call did not fault: `mean` and `r2m` (raw 2nd moment): p elements each; note 2C_MOM needs the 2R_MOM output registered, status -4007 otherwise; oneMKL 2026.2, Linux x86_64). Extents not listed as verified in this block are unverified (from MKL documentation).
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the oneMKL C signature vsl?SSEditMoments argument for argument"
    )]
    unsafe fn ss_edit_moments(
        task: VSLSSTaskPtr,
        mean: *mut Self,
        r2m: *mut Self,
        r3m: *mut Self,
        r4m: *mut Self,
        c2m: *mut Self,
        c3m: *mut Self,
        c4m: *mut Self,
    ) -> c_int;

    /// Raw `ss_compute` call.
    ///
    /// # Safety
    ///
    /// `task` must be a live `VSLSSTaskPtr` from `ss_new_task` that has not been deleted, and must not be used from another thread during the call.
    /// All pointers previously registered with the task via `ss_new_task` and the `ss_edit_*` calls must still be valid per their own contracts. `estimates` must be a bitwise OR of `VSL_SS_*` estimate flags for which the corresponding output buffers were registered; otherwise MKL writes through unset pointers. `method` must be a valid `VSL_SS_METHOD_*`.
    /// A non-zero return is the VSL status. The MKL library must be linked and usable.
    /// Extents not listed as verified in this block are unverified (from MKL documentation).
    unsafe fn ss_compute(task: VSLSSTaskPtr, estimates: u64, method: c_int) -> c_int;

    /// Raw `ss_edit_quantiles` call.
    ///
    /// # Safety
    ///
    /// `task` must be a live `VSLSSTaskPtr` from `ss_new_task` that has not been deleted, and must not be used from another thread during the call.
    /// `quant_order_n` points to one `c_int` (number of quantile orders `m`); `quant_orders` points to `m` values in `[0, 1]`; `quants_out` must be valid for writes of `p * m` elements; `order_stats_out` and `order_stats_storage` may be null together, otherwise `order_stats_out` must be valid for writes of `p * n` elements stored per `*order_stats_storage`.
    /// **Pointer retention:** the task stores the pointers it is given (it does not copy the data). Each buffer must stay allocated, at a stable address, and not be mutated through Rust references (outputs: not accessed at all) from this call until the task has been deleted or the pointer replaced, including across every `ss_compute` call.
    /// A non-zero return is the VSL status. The MKL library must be linked and usable.
    /// **Verified** by guard-page probe (smallest buffer for which the call did not fault: `quants_out`: p * q elements; oneMKL 2026.2, Linux x86_64). Extents not listed as verified in this block are unverified (from MKL documentation).
    unsafe fn ss_edit_quantiles(
        task: VSLSSTaskPtr,
        quant_order_n: *const c_int,
        quant_orders: *const Self,
        quants_out: *mut Self,
        order_stats_out: *mut Self,
        order_stats_storage: *const c_int,
    ) -> c_int;

    /// Raw `ss_edit_stream_quantiles` call.
    ///
    /// # Safety
    ///
    /// `task` must be a live `VSLSSTaskPtr` from `ss_new_task` that has not been deleted, and must not be used from another thread during the call.
    /// `quant_order_n` points to one `c_int` (`m`); `quant_orders` points to `m` values in `[0, 1]`; `quants_out` must be valid for writes of `p * m` elements; `params_n` points to one `c_int` (the number of parameters, at least 1) and `params` to that many values (the accuracy parameter).
    /// **Pointer retention:** the task stores the pointers it is given (it does not copy the data). Each buffer must stay allocated, at a stable address, and not be mutated through Rust references (outputs: not accessed at all) from this call until the task has been deleted or the pointer replaced, including across every `ss_compute` call.
    /// A non-zero return is the VSL status. The MKL library must be linked and usable.
    /// **Verified** by guard-page probe (`tools/size-probes/ss2.c`, smallest buffer for which the call did not fault, p = 3, n = 40, row storage; oneMKL 2026.2, Linux x86_64): `quants_out`: p * m elements (m = 3 orders; ZW method, accuracy 0.01, computed over one batch of n observations, not a multi-block stream).
    /// Extents not listed as verified in this block are unverified (from MKL documentation).
    unsafe fn ss_edit_stream_quantiles(
        task: VSLSSTaskPtr,
        quant_order_n: *const c_int,
        quant_orders: *const Self,
        quants_out: *mut Self,
        params_n: *const c_int,
        params: *const Self,
    ) -> c_int;

    /// Raw `ss_edit_robust_covariance` call.
    ///
    /// # Safety
    ///
    /// `task` must be a live `VSLSSTaskPtr` from `ss_new_task` that has not been deleted, and must not be used from another thread during the call.
    /// `cov_storage` points to one `c_int` giving the storage of `cov_out` (a `VSL_SS_MATRIX_STORAGE_*`); `n_params` points to one `c_int` and `params` to that many parameters; `mean_out` must be valid for writes of `p` elements and `cov_out` of `p * p` elements (or the packed size for packed storage).
    /// **Pointer retention:** the task stores the pointers it is given (it does not copy the data). Each buffer must stay allocated, at a stable address, and not be mutated through Rust references (outputs: not accessed at all) from this call until the task has been deleted or the pointer replaced, including across every `ss_compute` call.
    /// A non-zero return is the VSL status. The MKL library must be linked and usable.
    /// **Verified** by guard-page probe (`tools/size-probes/ss2.c`, smallest buffer for which the call did not fault, p = 3, n = 40, row storage; oneMKL 2026.2, Linux x86_64): `mean_out`: p and `cov_out`: p * p elements (full storage, TBS method with 4 parameters).
    /// Extents not listed as verified in this block are unverified (from MKL documentation).
    unsafe fn ss_edit_robust_covariance(
        task: VSLSSTaskPtr,
        cov_storage: *const c_int,
        n_params: *const c_int,
        params: *const Self,
        mean_out: *mut Self,
        cov_out: *mut Self,
    ) -> c_int;

    /// Raw `ss_edit_outliers_detection` call.
    ///
    /// # Safety
    ///
    /// `task` must be a live `VSLSSTaskPtr` from `ss_new_task` that has not been deleted, and must not be used from another thread during the call.
    /// `n_params` points to one `c_int` and `params` to that many parameters (3 as MKL documents); `weights_out` must be valid for writes of `n` elements (one weight per observation).
    /// **Pointer retention:** the task stores the pointers it is given (it does not copy the data). Each buffer must stay allocated, at a stable address, and not be mutated through Rust references (outputs: not accessed at all) from this call until the task has been deleted or the pointer replaced, including across every `ss_compute` call.
    /// A non-zero return is the VSL status. The MKL library must be linked and usable.
    /// **Verified** by guard-page probe (`tools/size-probes/ss2.c`, smallest buffer for which the call did not fault, p = 3, n = 40, row storage; oneMKL 2026.2, Linux x86_64): `weights_out`: n elements (BACON method with 3 parameters).
    /// Extents not listed as verified in this block are unverified (from MKL documentation).
    unsafe fn ss_edit_outliers_detection(
        task: VSLSSTaskPtr,
        n_params: *const c_int,
        params: *const Self,
        weights_out: *mut Self,
    ) -> c_int;

    /// Raw `ss_edit_missing_values` call.
    ///
    /// # Safety
    ///
    /// `task` must be a live `VSLSSTaskPtr` from `ss_new_task` that has not been deleted, and must not be used from another thread during the call.
    /// `n_params`/`params`, `init_estimates_n`/`init_estimates`, `prior_n`/`prior` and `simul_vals_n`/`estimates_n` are (count, data) pairs: each count points to one `c_int` and the paired data to that many `Self` values (null where MKL allows a default). `simul_vals_out` must be valid for writes of `simul_vals * p` elements and `estimates_out` of the estimates the task defines (the mean `p` and covariance `p * p` as requested).
    /// **Pointer retention:** the task stores the pointers it is given (it does not copy the data). Each buffer must stay allocated, at a stable address, and not be mutated through Rust references (outputs: not accessed at all) from this call until the task has been deleted or the pointer replaced, including across every `ss_compute` call.
    /// A non-zero return is the VSL status. The MKL library must be linked and usable.
    /// **Not verifiable so far:** `tools/size-probes/ss3.c` and a parameter sweep on oneMKL 2026.2 could not get `VSL_SS_METHOD_MI` to compute: with a null `simul_vals_out` (what `SummaryStats::impute_missing` passes) the compute step returns `VSL_SS_ERROR_BAD_MI_SIMUL_VALS_ADDR` (-4054), and with a buffer of `k * m` elements every parameter set tried returned `VSL_SS_ERROR_BAD_MI_PARAMS` (-4051) or `VSL_SS_ERROR_BAD_MI_MISSING_VALS_N` (-4063). The meaning of the five method parameters is not in the Developer Reference (it points to the separate Summary Statistics Application Notes), so no extent here is verified; treat the extents above as the documentation's.
    /// Extents not listed as verified in this block are unverified (from MKL documentation).
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the oneMKL C signature vsl?SSEditMissingValues argument for argument"
    )]
    unsafe fn ss_edit_missing_values(
        task: VSLSSTaskPtr,
        n_params: *const c_int,
        params: *const Self,
        init_estimates_n: *const c_int,
        init_estimates: *const Self,
        prior_n: *const c_int,
        prior: *const Self,
        simul_vals_n: *const c_int,
        simul_vals_out: *mut Self,
        estimates_n: *const c_int,
        estimates_out: *mut Self,
    ) -> c_int;
}

macro_rules! impl_ss_scalar {
    ($ty:ty,
        new=$new:ident, edit=$edit:ident,
        moments=$moments:ident, compute=$compute:ident,
        quantiles=$quantiles:ident, stream_quantiles=$stream_quantiles:ident,
        robust_cov=$robust_cov:ident, outliers=$outliers:ident,
        missing=$missing:ident,
        marker=$marker:expr,
    ) => {
        impl SsScalar for $ty {
            const MISSING: Self = <$ty>::from_bits($marker);
            fn is_missing(self) -> bool {
                self.to_bits() == $marker
            }
            fn from_f64(v: f64) -> Self {
                v as $ty
            }
            fn weight_value(self) -> Option<f64> {
                (self.is_finite() && self >= 0.0).then_some(f64::from(self))
            }
            unsafe fn ss_new_task(
                task: *mut VSLSSTaskPtr,
                p: *const c_int,
                n: *const c_int,
                xstorage: *const c_int,
                x: *const Self,
                w: *const Self,
                indices: *const c_int,
            ) -> c_int {
                unsafe { sys::$new(task, p, n, xstorage, x, w, indices) }
            }
            unsafe fn ss_edit_task(
                task: VSLSSTaskPtr,
                parameter: c_int,
                address: *const Self,
            ) -> c_int {
                unsafe { sys::$edit(task, parameter, address) }
            }
            unsafe fn ss_edit_moments(
                task: VSLSSTaskPtr,
                mean: *mut Self,
                r2m: *mut Self,
                r3m: *mut Self,
                r4m: *mut Self,
                c2m: *mut Self,
                c3m: *mut Self,
                c4m: *mut Self,
            ) -> c_int {
                unsafe { sys::$moments(task, mean, r2m, r3m, r4m, c2m, c3m, c4m) }
            }
            unsafe fn ss_compute(task: VSLSSTaskPtr, estimates: u64, method: c_int) -> c_int {
                unsafe { sys::$compute(task, estimates, method) }
            }
            unsafe fn ss_edit_quantiles(
                task: VSLSSTaskPtr,
                quant_order_n: *const c_int,
                quant_orders: *const Self,
                quants_out: *mut Self,
                order_stats_out: *mut Self,
                order_stats_storage: *const c_int,
            ) -> c_int {
                unsafe {
                    sys::$quantiles(
                        task,
                        quant_order_n,
                        quant_orders,
                        quants_out,
                        order_stats_out,
                        order_stats_storage,
                    )
                }
            }
            unsafe fn ss_edit_stream_quantiles(
                task: VSLSSTaskPtr,
                quant_order_n: *const c_int,
                quant_orders: *const Self,
                quants_out: *mut Self,
                params_n: *const c_int,
                params: *const Self,
            ) -> c_int {
                unsafe {
                    sys::$stream_quantiles(
                        task,
                        quant_order_n,
                        quant_orders,
                        quants_out,
                        params_n,
                        params,
                    )
                }
            }
            unsafe fn ss_edit_robust_covariance(
                task: VSLSSTaskPtr,
                cov_storage: *const c_int,
                n_params: *const c_int,
                params: *const Self,
                mean_out: *mut Self,
                cov_out: *mut Self,
            ) -> c_int {
                unsafe { sys::$robust_cov(task, cov_storage, n_params, params, mean_out, cov_out) }
            }
            unsafe fn ss_edit_outliers_detection(
                task: VSLSSTaskPtr,
                n_params: *const c_int,
                params: *const Self,
                weights_out: *mut Self,
            ) -> c_int {
                unsafe { sys::$outliers(task, n_params, params, weights_out) }
            }
            unsafe fn ss_edit_missing_values(
                task: VSLSSTaskPtr,
                n_params: *const c_int,
                params: *const Self,
                init_estimates_n: *const c_int,
                init_estimates: *const Self,
                prior_n: *const c_int,
                prior: *const Self,
                simul_vals_n: *const c_int,
                simul_vals_out: *mut Self,
                estimates_n: *const c_int,
                estimates_out: *mut Self,
            ) -> c_int {
                unsafe {
                    sys::$missing(
                        task,
                        n_params,
                        params,
                        init_estimates_n,
                        init_estimates,
                        prior_n,
                        prior,
                        simul_vals_n,
                        simul_vals_out,
                        estimates_n,
                        estimates_out,
                    )
                }
            }
        }
    };
}

impl_ss_scalar!(
    f32,
    new = vslsSSNewTask,
    edit = vslsSSEditTask,
    moments = vslsSSEditMoments,
    compute = vslsSSCompute,
    quantiles = vslsSSEditQuantiles,
    stream_quantiles = vslsSSEditStreamQuantiles,
    robust_cov = vslsSSEditRobustCovariance,
    outliers = vslsSSEditOutliersDetection,
    missing = vslsSSEditMissingValues,
    marker = 0xFFC0_0000,
);
impl_ss_scalar!(
    f64,
    new = vsldSSNewTask,
    edit = vsldSSEditTask,
    moments = vsldSSEditMoments,
    compute = vsldSSCompute,
    quantiles = vsldSSEditQuantiles,
    stream_quantiles = vsldSSEditStreamQuantiles,
    robust_cov = vsldSSEditRobustCovariance,
    outliers = vsldSSEditOutliersDetection,
    missing = vsldSSEditMissingValues,
    marker = 0xFFF8_0000_0000_0000,
);

/// Owned summary-statistics task. Each compute method registers output
/// buffers internally and runs `vsl?SSCompute`.
pub struct SummaryStats<'data, T: SsScalar> {
    task: VSLSSTaskPtr,
    p: usize,
    _n: usize,
    // MKL stores raw pointers to these inside the task, so they must
    // outlive every Edit/Compute call. Boxed so the addresses are
    // stable across moves of `Self`.
    _p_holder: Box<c_int>,
    _n_holder: Box<c_int>,
    _xstorage_holder: Box<c_int>,
    _data: &'data [T],
    _weights: Option<&'data [T]>,
    _marker: PhantomData<T>,
}

unsafe impl<T: SsScalar + Send> Send for SummaryStats<'_, T> {}

impl<'data, T: SsScalar> SummaryStats<'data, T> {
    /// Build a task over `num_variables × num_observations` data laid
    /// out so variable `i` occupies `data[i * num_observations ..]`.
    pub fn new(data: &'data [T], num_variables: usize, num_observations: usize) -> Result<Self> {
        Self::build(data, num_variables, num_observations, None)
    }

    /// Like [`new`](Self::new), with one weight per observation
    /// (`weights[j]` is the weight of observation `j` of every variable).
    /// `mean` and `variance` then use them. Measured on oneMKL 2026.2
    /// (`tools/size-probes/ss7.c`): the mean is `sum(w x) / W`, the variance
    /// (2nd central moment) is `sum(w (x - mean)^2) / (W - sum(w^2) / W)`
    /// with `W = sum(w)` (reliability weights). oneMKL itself accepts a
    /// negative weight (computing with it) and all-zero weights (mean 0)
    /// without an error, so this checks the weights first: each must be
    /// finite and not negative, and they must not all be zero.
    pub fn with_weights(
        data: &'data [T],
        num_variables: usize,
        num_observations: usize,
        weights: &'data [T],
    ) -> Result<Self> {
        if weights.len() < num_observations {
            return Err(Error::InvalidArgument(
                "weights must have one entry per observation",
            ));
        }
        let mut total = 0.0_f64;
        for w in &weights[..num_observations] {
            total += w.weight_value().ok_or(Error::InvalidArgument(
                "weights must be finite and not negative",
            ))?;
        }
        if total <= 0.0 {
            return Err(Error::InvalidArgument("weights must not all be zero"));
        }
        Self::build(data, num_variables, num_observations, Some(weights))
    }

    fn build(
        data: &'data [T],
        num_variables: usize,
        num_observations: usize,
        weights: Option<&'data [T]>,
    ) -> Result<Self> {
        if data.len() < num_variables * num_observations {
            return Err(Error::InvalidArgument(
                "data buffer is smaller than num_variables * num_observations",
            ));
        }
        let p_box: Box<c_int> = Box::new(
            num_variables
                .try_into()
                .map_err(|_| Error::DimensionOverflow)?,
        );
        let n_box: Box<c_int> = Box::new(
            num_observations
                .try_into()
                .map_err(|_| Error::DimensionOverflow)?,
        );
        let xstorage_box: Box<c_int> = Box::new(sys::VSL_SS_MATRIX_STORAGE_ROWS as c_int);
        let mut task: VSLSSTaskPtr = ptr::null_mut();
        let status = unsafe {
            T::ss_new_task(
                &mut task,
                &*p_box,
                &*n_box,
                &*xstorage_box,
                data.as_ptr(),
                weights.map_or(ptr::null(), <[T]>::as_ptr),
                ptr::null(),
            )
        };
        check_vsl(status)?;
        Ok(Self {
            task,
            p: num_variables,
            _n: num_observations,
            _p_holder: p_box,
            _n_holder: n_box,
            _xstorage_holder: xstorage_box,
            _data: data,
            _weights: weights,
            _marker: PhantomData,
        })
    }

    /// Per-variable mean. Length is `num_variables`.
    pub fn mean(&mut self) -> Result<Vec<T>> {
        let mut mean = vec![T::default(); self.p];
        let status =
            unsafe { T::ss_edit_task(self.task, sys::VSL_SS_ED_MEAN as c_int, mean.as_mut_ptr()) };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_MEAN as u64,
                sys::VSL_SS_METHOD_FAST as c_int,
            )
        };
        check_vsl(status)?;
        Ok(mean)
    }

    /// Per-variable variance (2nd central moment). Length is
    /// `num_variables`. Computing variance also requires mean and the
    /// raw 2nd moment to be registered; the wrapper allocates
    /// temporary buffers for both and discards them.
    pub fn variance(&mut self) -> Result<Vec<T>> {
        let mut mean = vec![T::default(); self.p];
        let mut r2m = vec![T::default(); self.p];
        let mut c2m = vec![T::default(); self.p];
        let status =
            unsafe { T::ss_edit_task(self.task, sys::VSL_SS_ED_MEAN as c_int, mean.as_mut_ptr()) };
        check_vsl(status)?;
        let status =
            unsafe { T::ss_edit_task(self.task, sys::VSL_SS_ED_2R_MOM as c_int, r2m.as_mut_ptr()) };
        check_vsl(status)?;
        let status =
            unsafe { T::ss_edit_task(self.task, sys::VSL_SS_ED_2C_MOM as c_int, c2m.as_mut_ptr()) };
        check_vsl(status)?;
        let estimates = (sys::VSL_SS_MEAN | sys::VSL_SS_2R_MOM | sys::VSL_SS_2C_MOM) as u64;
        let status =
            unsafe { T::ss_compute(self.task, estimates, sys::VSL_SS_METHOD_FAST as c_int) };
        check_vsl(status)?;
        Ok(c2m)
    }

    /// Per-variable minimum. Length is `num_variables`. The MKL min
    /// estimator must be seeded from an observation, so the wrapper
    /// initializes the result with the first observation of each
    /// variable before computing.
    pub fn min(&mut self) -> Result<Vec<T>> {
        let mut min = self.seed_from_first_observation();
        let status =
            unsafe { T::ss_edit_task(self.task, sys::VSL_SS_ED_MIN as c_int, min.as_mut_ptr()) };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_MIN as u64,
                sys::VSL_SS_METHOD_FAST as c_int,
            )
        };
        check_vsl(status)?;
        Ok(min)
    }

    /// Per-variable maximum. See [`min`](Self::min) for the seeding
    /// note.
    pub fn max(&mut self) -> Result<Vec<T>> {
        let mut max = self.seed_from_first_observation();
        let status =
            unsafe { T::ss_edit_task(self.task, sys::VSL_SS_ED_MAX as c_int, max.as_mut_ptr()) };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_MAX as u64,
                sys::VSL_SS_METHOD_FAST as c_int,
            )
        };
        check_vsl(status)?;
        Ok(max)
    }

    /// Build a length-`p` vector seeded from the first observation of
    /// each variable. Used to initialize min/max estimators.
    fn seed_from_first_observation(&self) -> Vec<T> {
        let n = self._n;
        (0..self.p).map(|i| self._data[i * n]).collect()
    }

    /// Per-variable sum. Length is `num_variables`.
    pub fn sum(&mut self) -> Result<Vec<T>> {
        let mut sum = vec![T::default(); self.p];
        let status =
            unsafe { T::ss_edit_task(self.task, sys::VSL_SS_ED_SUM as c_int, sum.as_mut_ptr()) };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_SUM as u64,
                sys::VSL_SS_METHOD_FAST as c_int,
            )
        };
        check_vsl(status)?;
        Ok(sum)
    }

    /// Per-variable quantiles at the requested `orders` (e.g.
    /// `[0.25, 0.5, 0.75]` for Q1 / median / Q3). Returns a flat
    /// vector laid out as `num_variables × orders.len()`, where
    /// `result[i * orders.len() + j]` is variable `i`'s `orders[j]`
    /// quantile.
    ///
    /// All order values must lie in `(0, 1)`.
    pub fn quantiles(&mut self, orders: &[T]) -> Result<Vec<T>>
    where
        T: PartialOrd,
    {
        if orders.is_empty() {
            return Err(Error::InvalidArgument("orders must be non-empty"));
        }
        let order_n: c_int = orders
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let mut out = vec![T::default(); self.p * orders.len()];
        let status = unsafe {
            T::ss_edit_quantiles(
                self.task,
                &order_n,
                orders.as_ptr(),
                out.as_mut_ptr(),
                ptr::null_mut(),
                ptr::null(),
            )
        };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_QUANTS as u64,
                sys::VSL_SS_METHOD_FAST as c_int,
            )
        };
        check_vsl(status)?;
        Ok(out)
    }

    /// Streaming-quantile estimator (Zhang-Wang, `VSL_SS_METHOD_SQUANTS_ZW`).
    /// Returns the same `num_variables × orders.len()` layout as
    /// [`Self::quantiles`].
    ///
    /// `params` must hold exactly one value, the accuracy of the
    /// estimate, in `(0, 1)` (typical 0.01). Measured on oneMKL 2026.2
    /// (`tools/size-probes/ss6.c`): 0 or 2 parameters and an accuracy of
    /// 0 or 2 are refused by oneMKL (-4064 / -4066 / -4067), and the
    /// `VSL_SS_METHOD_FAST` method that earlier versions of this wrapper
    /// passed is refused with -4029, so earlier versions always failed.
    /// With accuracy 0.01 on 1000 distinct values the orders 0.25, 0.5,
    /// 0.9 came back as 0.25, 0.497, 0.898. The Summary Statistics
    /// Application Notes, which would define the parameter, were not
    /// consulted: the layout is from the probe.
    pub fn stream_quantiles(&mut self, orders: &[T], params: &[T]) -> Result<Vec<T>> {
        if orders.is_empty() {
            return Err(Error::InvalidArgument("orders must be non-empty"));
        }
        if params.len() != 1 {
            return Err(Error::InvalidArgument(
                "stream quantiles take exactly one parameter, the accuracy",
            ));
        }
        let order_n: c_int = orders
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let params_n: c_int = params
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let mut out = vec![T::default(); self.p * orders.len()];
        let status = unsafe {
            T::ss_edit_stream_quantiles(
                self.task,
                &order_n,
                orders.as_ptr(),
                out.as_mut_ptr(),
                &params_n,
                params.as_ptr(),
            )
        };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_STREAM_QUANTS as u64,
                sys::VSL_SS_METHOD_SQUANTS_ZW as c_int,
            )
        };
        check_vsl(status)?;
        Ok(out)
    }

    /// Robust covariance + mean via Tukey's bi-weight S-estimator
    /// (`VSL_SS_METHOD_TBS` in MKL).
    ///
    /// Returns `(mean, cov)` where `cov` is the `p × p` covariance
    /// stored in full layout. `params` holds up to four values, in this
    /// order (oneMKL 2026.0 C reference, `vslSSEditRobustCovariance`):
    /// breakdown point (default `(n - p) / (2 n)`), asymptotic
    /// rejection probability (default 0.001), stopping tolerance
    /// (default 0.001), maximum iterations (default 10; 0 returns the
    /// Maronna initial estimate only). An empty slice accepts every
    /// default; a shorter slice sets the leading values. oneMKL
    /// requires `n > 2 p`: otherwise this returns `InvalidArgument`
    /// (measured on oneMKL 2026.2: the call itself fails with -4002 for
    /// `p = 3`, `n = 6`; `n = 7` works; counts 0 to 4 are accepted;
    /// `tools/size-probes/ss5.c`).
    ///
    /// Note: MKL exposes BACON only for *outlier weights*
    /// (see [`Self::outliers_bacon`]); for a full robust covariance
    /// estimate the supported algorithm is TBS.
    pub fn robust_covariance_tbs(&mut self, params: &[T]) -> Result<(Vec<T>, Vec<T>)> {
        if self._n <= 2 * self.p {
            return Err(Error::InvalidArgument(
                "robust covariance (TBS) needs more than 2 * num_variables observations",
            ));
        }
        let cov_storage: c_int = sys::VSL_SS_MATRIX_STORAGE_FULL as c_int;
        let n_params: c_int = params
            .len()
            .try_into()
            .map_err(|_| Error::DimensionOverflow)?;
        let mut mean = vec![T::default(); self.p];
        let mut cov = vec![T::default(); self.p * self.p];
        let status = unsafe {
            T::ss_edit_robust_covariance(
                self.task,
                &cov_storage,
                &n_params,
                if params.is_empty() {
                    ptr::null()
                } else {
                    params.as_ptr()
                },
                mean.as_mut_ptr(),
                cov.as_mut_ptr(),
            )
        };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_ROBUST_COV as u64,
                sys::VSL_SS_METHOD_TBS as c_int,
            )
        };
        check_vsl(status)?;
        Ok((mean, cov))
    }

    /// Per-observation outlier weights via BACON (Blocked Adaptive
    /// Computationally-efficient Outlier Nominators). Returns a
    /// vector of length `num_observations`: `1.0` for inliers,
    /// `0.0` for flagged outliers.
    ///
    /// `params` is BACON's 3-element configuration:
    ///
    /// - `params[0]` = initialization method
    ///   (`VSL_SS_METHOD_BACON_MEDIAN_INIT` or
    ///   `VSL_SS_METHOD_BACON_MAHALANOBIS_INIT` from `onemkl-sys`),
    /// - `params[1]` = significance level (e.g. `0.05`),
    /// - `params[2]` = stopping criterion: the algorithm ends when the
    ///   size of the basic subset changes by less than this (oneMKL
    ///   default 0.005; the reference describes it so, an earlier
    ///   version of this comment called it an initial subset fraction).
    ///   The reference recommends `params[1]` = α / n.
    ///
    /// Use [`SummaryStats::<f64>::bacon_default_params`] for sensible
    /// defaults.
    pub fn outliers_bacon(&mut self, params: &[T; 3]) -> Result<Vec<T>> {
        let n_params: c_int = 3;
        let mut weights = vec![T::default(); self._n];
        let status = unsafe {
            T::ss_edit_outliers_detection(
                self.task,
                &n_params,
                params.as_ptr(),
                weights.as_mut_ptr(),
            )
        };
        check_vsl(status)?;
        let status = unsafe {
            T::ss_compute(
                self.task,
                sys::VSL_SS_OUTLIERS as u64,
                sys::VSL_SS_METHOD_BACON as c_int,
            )
        };
        check_vsl(status)?;
        Ok(weights)
    }
}

impl SummaryStats<'_, f64> {
    /// Default BACON parameters: median initialization, 0.05
    /// probability, 0.005 stopping criterion (oneMKL's documented
    /// defaults).
    #[inline]
    #[must_use]
    pub fn bacon_default_params() -> [f64; 3] {
        [sys::VSL_SS_METHOD_BACON_MEDIAN_INIT as f64, 0.05, 0.05]
    }
}

impl SummaryStats<'_, f32> {
    /// `f32` counterpart of [`SummaryStats::<f64>::bacon_default_params`].
    #[inline]
    #[must_use]
    pub fn bacon_default_params() -> [f32; 3] {
        [sys::VSL_SS_METHOD_BACON_MEDIAN_INIT as f32, 0.05, 0.05]
    }
}

/// Settings for [`impute_missing`] (the multiple-imputation method, `VSL_SS_METHOD_MI`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImputeOptions<T> {
    /// Maximum iterations of the EM algorithm that starts the sampler (oneMKL default 50).
    pub em_iterations: usize,
    /// Iterations of the data-augmentation sampler (oneMKL default 30).
    pub da_iterations: usize,
    /// EM stopping tolerance (oneMKL default 0.001).
    pub tolerance: T,
    /// Number of imputed sets `m` to draw.
    pub sets: usize,
}

impl<T: SsScalar + From<f32>> Default for ImputeOptions<T> {
    /// oneMKL's documented defaults (50 EM iterations, 30 DA iterations, tolerance 0.001) and two sets.
    fn default() -> Self {
        Self {
            em_iterations: 50,
            da_iterations: 30,
            tolerance: T::from(0.001_f32),
            sets: 2,
        }
    }
}

/// Result of [`impute_missing`].
#[derive(Debug, Clone, PartialEq)]
pub struct Imputation<T> {
    /// Number of missing cells `k` found in the data.
    pub missing_count: usize,
    /// `k * sets` imputed values: for each missing cell in order of appearance, its `sets` draws are consecutive.
    pub simulated: Vec<T>,
    /// `sets * da_iterations` parameter estimates drawn by the sampler, each `p` means followed by the upper triangle of the `p x p` covariance (`p + (p*p + p) / 2` values).
    pub estimates: Vec<T>,
}

/// Multiple imputation of missing observations (`vsl?SSEditMissingValues`, method `VSL_SS_METHOD_MI`).
///
/// `data` is `num_variables x num_observations`, variable `i` at `data[i * num_observations ..]` as in [`SummaryStats::new`]. Mark each missing cell with [`SsScalar::MISSING`] (oneMKL compares bit patterns; `f64::NAN` is not recognised). The cells are counted here and oneMKL is told the count.
///
/// **`data` is modified**: oneMKL rewrites the missing cells (so `data` is taken by `&mut`). Re-mark them to run again.
///
/// Facts below come from the oneMKL 2026.0 C reference (`vslSSEditMissingValues`: parameter layout `{em_iter, da_iter, tolerance, m, missing_count}`, `simul` size `m * missing_count`, estimates size `m * da_iter * (p + (p*p + p)/2)`) and a run on oneMKL 2026.2, Linux x86_64 (`tools/size-probes/ss4.c`): the call succeeds with that layout. The initial-estimate and prior parameters are left at oneMKL's defaults. The earlier `SummaryStats::impute_missing` never worked (it passed no marker, a wrong parameter layout and no simulated-values buffer) and was removed.
///
/// # Errors
///
/// `InvalidArgument` if `data` is shorter than `p * n`, `sets` or `da_iterations` is 0, or no cell is marked missing; otherwise the oneMKL status.
pub fn impute_missing<T: SsScalar>(
    data: &mut [T],
    num_variables: usize,
    num_observations: usize,
    options: &ImputeOptions<T>,
) -> Result<Imputation<T>> {
    let cells = num_variables
        .checked_mul(num_observations)
        .ok_or(Error::DimensionOverflow)?;
    if data.len() < cells {
        return Err(Error::InvalidArgument(
            "data buffer is smaller than num_variables * num_observations",
        ));
    }
    if options.sets == 0 || options.da_iterations == 0 {
        return Err(Error::InvalidArgument(
            "sets and da_iterations must be at least 1",
        ));
    }
    let missing_count = data[..cells].iter().filter(|v| v.is_missing()).count();
    if missing_count == 0 {
        return Err(Error::InvalidArgument(
            "no cell is marked with SsScalar::MISSING",
        ));
    }
    let conv = |v: usize| -> Result<T> { Ok(T::from_f64(v as f64)) };
    let params = [
        conv(options.em_iterations)?,
        conv(options.da_iterations)?,
        options.tolerance,
        conv(options.sets)?,
        conv(missing_count)?,
    ];
    let p = num_variables;
    let simul_len = options
        .sets
        .checked_mul(missing_count)
        .ok_or(Error::DimensionOverflow)?;
    let est_len = options
        .sets
        .checked_mul(options.da_iterations)
        .and_then(|v| v.checked_mul(p + (p * p + p) / 2))
        .ok_or(Error::DimensionOverflow)?;
    let to_int = |v: usize| -> Result<c_int> { v.try_into().map_err(|_| Error::DimensionOverflow) };
    let (p_i, n_i) = (to_int(p)?, to_int(num_observations)?);
    let xstorage = sys::VSL_SS_MATRIX_STORAGE_ROWS as c_int;
    let n_params: c_int = 5;
    let zero: c_int = 0;
    let (simul_n, est_n) = (to_int(simul_len)?, to_int(est_len)?);
    let mut simulated = vec![T::default(); simul_len];
    let mut estimates = vec![T::default(); est_len];
    let mut task: VSLSSTaskPtr = ptr::null_mut();
    let status = unsafe {
        T::ss_new_task(
            &mut task,
            &p_i,
            &n_i,
            &xstorage,
            data.as_ptr(),
            ptr::null(),
            ptr::null(),
        )
    };
    check_vsl(status)?;
    let status = unsafe {
        T::ss_edit_missing_values(
            task,
            &n_params,
            params.as_ptr(),
            &zero,
            ptr::null(),
            &zero,
            ptr::null(),
            &simul_n,
            simulated.as_mut_ptr(),
            &est_n,
            estimates.as_mut_ptr(),
        )
    };
    let status = if status == 0 {
        unsafe {
            T::ss_compute(
                task,
                sys::VSL_SS_MISSING_VALS as u64,
                sys::VSL_SS_METHOD_MI as c_int,
            )
        }
    } else {
        status
    };
    unsafe {
        let _ = sys::vslSSDeleteTask(&mut task);
    }
    check_vsl(status)?;
    Ok(Imputation {
        missing_count,
        simulated,
        estimates,
    })
}

impl<T: SsScalar> Drop for SummaryStats<'_, T> {
    fn drop(&mut self) {
        if !self.task.is_null() {
            unsafe {
                let _ = sys::vslSSDeleteTask(&mut self.task);
            }
        }
    }
}

#[inline]
fn check_vsl(status: c_int) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::PardisoStatus(status))
    }
}
