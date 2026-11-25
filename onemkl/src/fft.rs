//! Discrete Fourier Transform (DFT) — fast Fourier transform via the
//! oneMKL DFTI interface.
//!
//! This module exposes three plan types:
//!
//! - [`FftPlan`] — in-place complex-to-complex FFT, 1-D, 2-D, 3-D, or
//!   arbitrary dimensionality.
//! - [`FftPlanOutOfPlace`] — out-of-place complex-to-complex FFT.
//! - [`RealFftPlan`] — real-to-complex (forward) and complex-to-real
//!   (backward) using the CCE (conjugate-even) storage format.
//!
//! All plans own a committed DFTI descriptor and free it on drop.
//!
//! ```no_run
//! use num_complex::Complex64;
//! use onemkl::fft::FftPlan;
//!
//! // 1-D in-place FFT.
//! let mut plan = FftPlan::<f64>::complex_1d(8).unwrap();
//! let mut buf: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); 8];
//! buf[0] = Complex64::new(1.0, 0.0);
//! plan.forward_in_place(&mut buf).unwrap();
//! ```

use core::ffi::c_long;
use core::marker::PhantomData;
use core::ptr;

use num_complex::Complex;
use onemkl_sys::{self as sys, DFTI_DESCRIPTOR_HANDLE};

use crate::error::{Error, Result};

/// Precision of an FFT plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FftPrecision {
    /// Single precision (32-bit).
    Single,
    /// Double precision (64-bit).
    Double,
}

impl FftPrecision {
    #[inline]
    fn as_dfti(self) -> sys::DFTI_CONFIG_VALUE::Type {
        match self {
            Self::Single => sys::DFTI_CONFIG_VALUE::DFTI_SINGLE,
            Self::Double => sys::DFTI_CONFIG_VALUE::DFTI_DOUBLE,
        }
    }
}

/// Trait associating a Rust real scalar type with its DFTI precision
/// code.
pub trait FftScalar: Copy + 'static {
    /// DFTI precision code corresponding to this real type.
    const PRECISION: FftPrecision;
    /// Convert to `f64` for variadic FFI calls. DFTI's variadic
    /// `DftiSetValue` reads the value with `va_arg(ap, double)`
    /// regardless of descriptor precision, then converts internally
    /// — so callers must pass `double` even for `DFTI_SINGLE`.
    fn to_f64(self) -> f64;
}
impl FftScalar for f32 {
    const PRECISION: FftPrecision = FftPrecision::Single;
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl FftScalar for f64 {
    const PRECISION: FftPrecision = FftPrecision::Double;
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

// =====================================================================
// Internal helpers
// =====================================================================

#[inline]
fn check_dfti(status: c_long) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::DftiStatus(status as i64))
    }
}

#[inline]
fn apply_scales<T: FftScalar>(
    handle: DFTI_DESCRIPTOR_HANDLE,
    scales: Option<(T, T)>,
) -> Result<()> {
    if let Some((forward, backward)) = scales {
        // DFTI's variadic interface reads scales as `double`
        // regardless of descriptor precision, so cast f32 → f64.
        let status = unsafe {
            sys::DftiSetValue(
                handle,
                sys::DFTI_CONFIG_PARAM::DFTI_FORWARD_SCALE,
                forward.to_f64(),
            )
        };
        check_dfti(status)?;
        let status = unsafe {
            sys::DftiSetValue(
                handle,
                sys::DFTI_CONFIG_PARAM::DFTI_BACKWARD_SCALE,
                backward.to_f64(),
            )
        };
        check_dfti(status)?;
    }
    Ok(())
}

#[inline]
fn create_complex_descriptor<T: FftScalar>(
    dims: &[usize],
    placement: sys::DFTI_CONFIG_VALUE::Type,
    scales: Option<(T, T)>,
) -> Result<DFTI_DESCRIPTOR_HANDLE> {
    if dims.is_empty() {
        return Err(Error::InvalidArgument(
            "FFT plan needs at least one dimension",
        ));
    }
    let dims_long: Vec<c_long> = dims
        .iter()
        .map(|&d| {
            c_long::try_from(d).map_err(|_| Error::DimensionOverflow)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut handle: DFTI_DESCRIPTOR_HANDLE = ptr::null_mut();
    let ndim_long: c_long = c_long::try_from(dims.len())
        .map_err(|_| Error::DimensionOverflow)?;
    let precision = T::PRECISION.as_dfti();
    let domain = sys::DFTI_CONFIG_VALUE::DFTI_COMPLEX;

    let status = unsafe {
        if dims.len() == 1 {
            sys::DftiCreateDescriptor(&mut handle, precision, domain, ndim_long, dims_long[0])
        } else {
            sys::DftiCreateDescriptor(
                &mut handle,
                precision,
                domain,
                ndim_long,
                dims_long.as_ptr(),
            )
        }
    };
    check_dfti(status)?;

    let status = unsafe {
        sys::DftiSetValue(
            handle,
            sys::DFTI_CONFIG_PARAM::DFTI_PLACEMENT,
            placement,
        )
    };
    if let Err(e) = check_dfti(status) {
        unsafe {
            let mut h = handle;
            let _ = sys::DftiFreeDescriptor(&mut h);
        }
        return Err(e);
    }

    if let Err(e) = apply_scales(handle, scales) {
        unsafe {
            let mut h = handle;
            let _ = sys::DftiFreeDescriptor(&mut h);
        }
        return Err(e);
    }

    let status = unsafe { sys::DftiCommitDescriptor(handle) };
    if let Err(e) = check_dfti(status) {
        unsafe {
            let mut h = handle;
            let _ = sys::DftiFreeDescriptor(&mut h);
        }
        return Err(e);
    }
    Ok(handle)
}

#[inline]
fn create_real_descriptor<T: FftScalar>(
    dims: &[usize],
    placement: sys::DFTI_CONFIG_VALUE::Type,
    scales: Option<(T, T)>,
) -> Result<DFTI_DESCRIPTOR_HANDLE> {
    if dims.is_empty() {
        return Err(Error::InvalidArgument(
            "FFT plan needs at least one dimension",
        ));
    }
    let dims_long: Vec<c_long> = dims
        .iter()
        .map(|&d| c_long::try_from(d).map_err(|_| Error::DimensionOverflow))
        .collect::<Result<Vec<_>>>()?;

    let mut handle: DFTI_DESCRIPTOR_HANDLE = ptr::null_mut();
    let ndim_long: c_long = c_long::try_from(dims.len())
        .map_err(|_| Error::DimensionOverflow)?;
    let precision = T::PRECISION.as_dfti();
    let domain = sys::DFTI_CONFIG_VALUE::DFTI_REAL;

    let status = unsafe {
        if dims.len() == 1 {
            sys::DftiCreateDescriptor(&mut handle, precision, domain, ndim_long, dims_long[0])
        } else {
            sys::DftiCreateDescriptor(
                &mut handle,
                precision,
                domain,
                ndim_long,
                dims_long.as_ptr(),
            )
        }
    };
    check_dfti(status)?;

    // Pack CCE output as one contiguous Complex<T> array of size
    // floor(n0/2)+1 along the first dimension (MKL's fastest-varying
    // dimension, since DFTI defaults to Fortran-style strides) and
    // full size along the rest.
    let status = unsafe {
        sys::DftiSetValue(
            handle,
            sys::DFTI_CONFIG_PARAM::DFTI_CONJUGATE_EVEN_STORAGE,
            sys::DFTI_CONFIG_VALUE::DFTI_COMPLEX_COMPLEX,
        )
    };
    if let Err(e) = check_dfti(status) {
        unsafe {
            let mut h = handle;
            let _ = sys::DftiFreeDescriptor(&mut h);
        }
        return Err(e);
    }

    // MKL's default INPUT_STRIDES are computed for in-place storage,
    // which assumes the real buffer has been padded to hold the
    // complex output. For not-in-place real ↔ complex transforms we
    // must override both stride arrays so MKL walks each side with
    // its own contiguous layout.
    if placement == sys::DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE && dims.len() > 1 {
        // MKL DFTI uses Fortran-style strides where stride[i+1] is the
        // step along dimension i. CCE shrinks the LAST physical
        // dimension of the spectrum (the slowest-varying one in
        // column-major), not the first.
        let last = dims.len() - 1;
        let mut input_strides: Vec<c_long> = Vec::with_capacity(dims.len() + 1);
        input_strides.push(0);
        let mut acc: c_long = 1;
        for &d in &dims_long {
            input_strides.push(acc);
            acc = acc.checked_mul(d).ok_or(Error::DimensionOverflow)?;
        }

        let mut output_strides: Vec<c_long> = Vec::with_capacity(dims.len() + 1);
        output_strides.push(0);
        let cce_last: c_long = c_long::try_from(cce_complex_len(dims[last]))
            .map_err(|_| Error::DimensionOverflow)?;
        let mut acc: c_long = 1;
        for (i, &d) in dims_long.iter().enumerate() {
            output_strides.push(acc);
            let step = if i == last { cce_last } else { d };
            acc = acc.checked_mul(step).ok_or(Error::DimensionOverflow)?;
        }

        let status = unsafe {
            sys::DftiSetValue(
                handle,
                sys::DFTI_CONFIG_PARAM::DFTI_INPUT_STRIDES,
                input_strides.as_ptr(),
            )
        };
        if let Err(e) = check_dfti(status) {
            unsafe {
                let mut h = handle;
                let _ = sys::DftiFreeDescriptor(&mut h);
            }
            return Err(e);
        }
        let status = unsafe {
            sys::DftiSetValue(
                handle,
                sys::DFTI_CONFIG_PARAM::DFTI_OUTPUT_STRIDES,
                output_strides.as_ptr(),
            )
        };
        if let Err(e) = check_dfti(status) {
            unsafe {
                let mut h = handle;
                let _ = sys::DftiFreeDescriptor(&mut h);
            }
            return Err(e);
        }
    }

    let status = unsafe {
        sys::DftiSetValue(
            handle,
            sys::DFTI_CONFIG_PARAM::DFTI_PLACEMENT,
            placement,
        )
    };
    if let Err(e) = check_dfti(status) {
        unsafe {
            let mut h = handle;
            let _ = sys::DftiFreeDescriptor(&mut h);
        }
        return Err(e);
    }

    if let Err(e) = apply_scales(handle, scales) {
        unsafe {
            let mut h = handle;
            let _ = sys::DftiFreeDescriptor(&mut h);
        }
        return Err(e);
    }

    let status = unsafe { sys::DftiCommitDescriptor(handle) };
    if let Err(e) = check_dfti(status) {
        unsafe {
            let mut h = handle;
            let _ = sys::DftiFreeDescriptor(&mut h);
        }
        return Err(e);
    }
    Ok(handle)
}

/// Number of complex elements in a CCE forward output for an `n`-point
/// real input along the FFT dimension: `n/2 + 1`.
#[inline]
#[must_use]
pub fn cce_complex_len(real_n: usize) -> usize {
    real_n / 2 + 1
}

// =====================================================================
// FftPlan — in-place complex-to-complex
// =====================================================================

/// In-place complex-to-complex FFT plan.
pub struct FftPlan<T: FftScalar> {
    handle: DFTI_DESCRIPTOR_HANDLE,
    total_complex_len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: FftScalar> Send for FftPlan<T> {}

impl<T: FftScalar> FftPlan<T> {
    /// 1-D in-place plan of length `n`.
    pub fn complex_1d(n: usize) -> Result<Self> {
        Self::complex_nd(&[n])
    }

    /// 2-D in-place plan of size `n0 × n1`.
    pub fn complex_2d(n0: usize, n1: usize) -> Result<Self> {
        Self::complex_nd(&[n0, n1])
    }

    /// 3-D in-place plan of size `n0 × n1 × n2`.
    pub fn complex_3d(n0: usize, n1: usize, n2: usize) -> Result<Self> {
        Self::complex_nd(&[n0, n1, n2])
    }

    /// N-D in-place plan with the supplied dimensions.
    pub fn complex_nd(dims: &[usize]) -> Result<Self> {
        let handle = create_complex_descriptor::<T>(
            dims,
            sys::DFTI_CONFIG_VALUE::DFTI_INPLACE,
            None,
        )?;
        let total: usize = dims.iter().product();
        Ok(Self {
            handle,
            total_complex_len: total,
            _marker: PhantomData,
        })
    }

    /// N-D in-place plan with explicit forward / backward scales. Set
    /// `backward_scale = 1.0 / N` (where `N = dims.iter().product()`)
    /// to make `IFFT(FFT(x)) = x`. Set both to `1.0 / sqrt(N)` for a
    /// unitary transform pair.
    pub fn complex_nd_with_scales(
        dims: &[usize],
        forward_scale: T,
        backward_scale: T,
    ) -> Result<Self> {
        let handle = create_complex_descriptor::<T>(
            dims,
            sys::DFTI_CONFIG_VALUE::DFTI_INPLACE,
            Some((forward_scale, backward_scale)),
        )?;
        let total: usize = dims.iter().product();
        Ok(Self {
            handle,
            total_complex_len: total,
            _marker: PhantomData,
        })
    }

    /// Forward transform in place.
    pub fn forward_in_place(&mut self, x: &mut [Complex<T>]) -> Result<()> {
        if x.len() < self.total_complex_len {
            return Err(Error::InvalidArgument(
                "buffer too small for FFT plan size",
            ));
        }
        let status = unsafe {
            sys::DftiComputeForward(self.handle, x.as_mut_ptr().cast())
        };
        check_dfti(status)
    }

    /// Backward transform in place (un-normalized).
    pub fn backward_in_place(&mut self, x: &mut [Complex<T>]) -> Result<()> {
        if x.len() < self.total_complex_len {
            return Err(Error::InvalidArgument(
                "buffer too small for FFT plan size",
            ));
        }
        let status = unsafe {
            sys::DftiComputeBackward(self.handle, x.as_mut_ptr().cast())
        };
        check_dfti(status)
    }
}

impl<T: FftScalar> Drop for FftPlan<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = sys::DftiFreeDescriptor(&mut self.handle);
            }
        }
    }
}

// =====================================================================
// FftPlanOutOfPlace — out-of-place complex-to-complex
// =====================================================================

/// Out-of-place complex-to-complex FFT plan.
pub struct FftPlanOutOfPlace<T: FftScalar> {
    handle: DFTI_DESCRIPTOR_HANDLE,
    total_complex_len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: FftScalar> Send for FftPlanOutOfPlace<T> {}

impl<T: FftScalar> FftPlanOutOfPlace<T> {
    /// 1-D out-of-place plan.
    pub fn complex_1d(n: usize) -> Result<Self> {
        Self::complex_nd(&[n])
    }

    /// 2-D out-of-place plan.
    pub fn complex_2d(n0: usize, n1: usize) -> Result<Self> {
        Self::complex_nd(&[n0, n1])
    }

    /// 3-D out-of-place plan.
    pub fn complex_3d(n0: usize, n1: usize, n2: usize) -> Result<Self> {
        Self::complex_nd(&[n0, n1, n2])
    }

    /// N-D out-of-place plan.
    pub fn complex_nd(dims: &[usize]) -> Result<Self> {
        let handle = create_complex_descriptor::<T>(
            dims,
            sys::DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE,
            None,
        )?;
        let total: usize = dims.iter().product();
        Ok(Self {
            handle,
            total_complex_len: total,
            _marker: PhantomData,
        })
    }

    /// N-D out-of-place plan with explicit forward / backward scales.
    /// See [`FftPlan::complex_nd_with_scales`] for normalization
    /// guidance.
    pub fn complex_nd_with_scales(
        dims: &[usize],
        forward_scale: T,
        backward_scale: T,
    ) -> Result<Self> {
        let handle = create_complex_descriptor::<T>(
            dims,
            sys::DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE,
            Some((forward_scale, backward_scale)),
        )?;
        let total: usize = dims.iter().product();
        Ok(Self {
            handle,
            total_complex_len: total,
            _marker: PhantomData,
        })
    }

    /// Forward transform `out ← FFT(in)`. The two buffers must have the
    /// same length (≥ the plan's total size).
    pub fn forward(&mut self, input: &[Complex<T>], out: &mut [Complex<T>]) -> Result<()> {
        if input.len() < self.total_complex_len || out.len() < self.total_complex_len {
            return Err(Error::InvalidArgument(
                "buffer too small for FFT plan size",
            ));
        }
        let status = unsafe {
            sys::DftiComputeForward(
                self.handle,
                input.as_ptr() as *mut core::ffi::c_void,
                out.as_mut_ptr().cast::<core::ffi::c_void>(),
            )
        };
        check_dfti(status)
    }

    /// Backward transform `out ← IFFT(in)` (un-normalized).
    pub fn backward(&mut self, input: &[Complex<T>], out: &mut [Complex<T>]) -> Result<()> {
        if input.len() < self.total_complex_len || out.len() < self.total_complex_len {
            return Err(Error::InvalidArgument(
                "buffer too small for FFT plan size",
            ));
        }
        let status = unsafe {
            sys::DftiComputeBackward(
                self.handle,
                input.as_ptr() as *mut core::ffi::c_void,
                out.as_mut_ptr().cast::<core::ffi::c_void>(),
            )
        };
        check_dfti(status)
    }
}

impl<T: FftScalar> Drop for FftPlanOutOfPlace<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = sys::DftiFreeDescriptor(&mut self.handle);
            }
        }
    }
}

// =====================================================================
// RealFftPlan — real-to-complex / complex-to-real (CCE format)
// =====================================================================

/// Real-input FFT plan. Forward maps a real `n0 × n1 × ...` array to a
/// `(n0/2 + 1) × n1 × ...` complex array (CCE format); backward does
/// the reverse.
///
/// The plan is always out-of-place because the real and complex
/// buffers have different sizes.
pub struct RealFftPlan<T: FftScalar> {
    handle: DFTI_DESCRIPTOR_HANDLE,
    real_total_len: usize,
    complex_total_len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: FftScalar> Send for RealFftPlan<T> {}

impl<T: FftScalar> RealFftPlan<T> {
    /// 1-D real FFT plan of length `n`. Real buffer has `n` elements;
    /// CCE complex buffer has `n / 2 + 1` elements.
    pub fn real_1d(n: usize) -> Result<Self> {
        Self::real_nd(&[n])
    }

    /// 2-D real FFT plan of size `n0 × n1`.
    pub fn real_2d(n0: usize, n1: usize) -> Result<Self> {
        Self::real_nd(&[n0, n1])
    }

    /// 3-D real FFT plan of size `n0 × n1 × n2`.
    pub fn real_3d(n0: usize, n1: usize, n2: usize) -> Result<Self> {
        Self::real_nd(&[n0, n1, n2])
    }

    /// N-D real FFT plan with the supplied dimensions.
    ///
    /// CCE storage shrinks the last dimension to
    /// `dims[dims.len() - 1] / 2 + 1` in the complex output.
    pub fn real_nd(dims: &[usize]) -> Result<Self> {
        Self::real_nd_inner(dims, None)
    }

    /// N-D real FFT plan with explicit forward / backward scales. See
    /// [`FftPlan::complex_nd_with_scales`] for normalization
    /// guidance.
    pub fn real_nd_with_scales(
        dims: &[usize],
        forward_scale: T,
        backward_scale: T,
    ) -> Result<Self> {
        Self::real_nd_inner(dims, Some((forward_scale, backward_scale)))
    }

    fn real_nd_inner(dims: &[usize], scales: Option<(T, T)>) -> Result<Self> {
        let handle = create_real_descriptor::<T>(
            dims,
            sys::DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE,
            scales,
        )?;
        let real_total: usize = dims.iter().product();
        let mut complex_dims = dims.to_vec();
        let last = complex_dims.len() - 1;
        complex_dims[last] = cce_complex_len(complex_dims[last]);
        let complex_total: usize = complex_dims.iter().product();
        Ok(Self {
            handle,
            real_total_len: real_total,
            complex_total_len: complex_total,
            _marker: PhantomData,
        })
    }

    /// Number of real elements per transform.
    #[inline]
    #[must_use]
    pub fn real_len(&self) -> usize {
        self.real_total_len
    }

    /// Number of complex elements per transform (CCE format).
    #[inline]
    #[must_use]
    pub fn complex_len(&self) -> usize {
        self.complex_total_len
    }

    /// Forward transform `out ← FFT(in)`.
    pub fn forward(&mut self, input: &[T], out: &mut [Complex<T>]) -> Result<()> {
        if input.len() < self.real_total_len {
            return Err(Error::InvalidArgument(
                "real input buffer too small for FFT plan size",
            ));
        }
        if out.len() < self.complex_total_len {
            return Err(Error::InvalidArgument(
                "complex output buffer too small (need n/2 + 1 along first dim)",
            ));
        }
        let status = unsafe {
            sys::DftiComputeForward(
                self.handle,
                input.as_ptr() as *mut core::ffi::c_void,
                out.as_mut_ptr().cast::<core::ffi::c_void>(),
            )
        };
        check_dfti(status)
    }

    /// Backward transform `out ← IFFT(in)` (un-normalized).
    ///
    /// The complex input must satisfy CCE Hermitian symmetry —
    /// otherwise the output is not real and the result is undefined.
    pub fn backward(&mut self, input: &[Complex<T>], out: &mut [T]) -> Result<()> {
        if input.len() < self.complex_total_len {
            return Err(Error::InvalidArgument(
                "complex input buffer too small (need n/2 + 1 along first dim)",
            ));
        }
        if out.len() < self.real_total_len {
            return Err(Error::InvalidArgument(
                "real output buffer too small for FFT plan size",
            ));
        }
        let status = unsafe {
            sys::DftiComputeBackward(
                self.handle,
                input.as_ptr() as *mut core::ffi::c_void,
                out.as_mut_ptr().cast::<core::ffi::c_void>(),
            )
        };
        check_dfti(status)
    }
}

impl<T: FftScalar> Drop for RealFftPlan<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = sys::DftiFreeDescriptor(&mut self.handle);
            }
        }
    }
}

// =====================================================================
// StridedFftPlan — N-D in-place complex FFT over explicit element strides
// =====================================================================

/// N-D in-place complex-to-complex FFT over a strided layout.
///
/// `strides[i]` is the distance, in complex elements, between consecutive
/// indices of dimension `i` (so a padded row-major `n0 x n1` array with
/// leading dimension `ld` is `dims = [n0, n1]`, `strides = [ld, 1]`, and a
/// column-major one is `strides = [1, n0]`). Elements between the strided
/// positions are never touched. Strides must be positive and the layout
/// must not overlap itself.
pub struct StridedFftPlan<T: FftScalar> {
    handle: DFTI_DESCRIPTOR_HANDLE,
    buffer_len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: FftScalar> Send for StridedFftPlan<T> {}

impl<T: FftScalar> StridedFftPlan<T> {
    /// Unscaled plan over `dims` with element `strides`.
    pub fn new(dims: &[usize], strides: &[usize]) -> Result<Self> {
        Self::build(dims, strides, None)
    }

    /// Plan with explicit forward / backward scales.
    pub fn with_scales(
        dims: &[usize],
        strides: &[usize],
        forward_scale: T,
        backward_scale: T,
    ) -> Result<Self> {
        Self::build(dims, strides, Some((forward_scale, backward_scale)))
    }

    fn build(dims: &[usize], strides: &[usize], scales: Option<(T, T)>) -> Result<Self> {
        if dims.is_empty() {
            return Err(Error::InvalidArgument("FFT plan needs at least one dimension"));
        }
        if dims.len() != strides.len() {
            return Err(Error::InvalidArgument("dims and strides must have the same length"));
        }
        if dims.iter().any(|&d| d == 0) || strides.iter().any(|&s| s == 0) {
            return Err(Error::InvalidArgument("dimensions and strides must be positive"));
        }
        // Non-overlap: ordered by stride, each stride must clear the
        // extent spanned by the next-faster dimension.
        let mut order: Vec<usize> = (0..dims.len()).collect();
        order.sort_by_key(|&i| strides[i]);
        for w in order.windows(2) {
            let (a, b) = (w[0], w[1]);
            let span = strides[a].checked_mul(dims[a]).ok_or(Error::DimensionOverflow)?;
            if dims[a] > 1 && strides[b] < span {
                return Err(Error::InvalidArgument(
                    "strides overlap: a stride is smaller than the extent of a faster dimension",
                ));
            }
        }
        let mut buffer_len = 1usize;
        for (&d, &s) in dims.iter().zip(strides) {
            buffer_len = (d - 1)
                .checked_mul(s)
                .and_then(|v| buffer_len.checked_add(v))
                .ok_or(Error::DimensionOverflow)?;
        }
        let to_long = |v: usize| c_long::try_from(v).map_err(|_| Error::DimensionOverflow);
        let dims_l: Vec<c_long> = dims.iter().map(|&d| to_long(d)).collect::<Result<_>>()?;
        // MKL layout: [offset, stride_0, stride_1, ...].
        let mut strides_l: Vec<c_long> = Vec::with_capacity(strides.len() + 1);
        strides_l.push(0);
        for &s in strides {
            strides_l.push(to_long(s)?);
        }
        let ndim = to_long(dims.len())?;

        let mut handle: DFTI_DESCRIPTOR_HANDLE = ptr::null_mut();
        check_dfti(unsafe {
            if dims.len() == 1 {
                sys::DftiCreateDescriptor(
                    &mut handle,
                    T::PRECISION.as_dfti(),
                    sys::DFTI_CONFIG_VALUE::DFTI_COMPLEX,
                    ndim,
                    dims_l[0],
                )
            } else {
                sys::DftiCreateDescriptor(
                    &mut handle,
                    T::PRECISION.as_dfti(),
                    sys::DFTI_CONFIG_VALUE::DFTI_COMPLEX,
                    ndim,
                    dims_l.as_ptr(),
                )
            }
        })?;
        let configure = |handle: DFTI_DESCRIPTOR_HANDLE| -> Result<()> {
            use sys::DFTI_CONFIG_PARAM as P;
            check_dfti(unsafe {
                sys::DftiSetValue(handle, P::DFTI_PLACEMENT, sys::DFTI_CONFIG_VALUE::DFTI_INPLACE)
            })?;
            check_dfti(unsafe {
                sys::DftiSetValue(handle, P::DFTI_INPUT_STRIDES, strides_l.as_ptr())
            })?;
            check_dfti(unsafe {
                sys::DftiSetValue(handle, P::DFTI_OUTPUT_STRIDES, strides_l.as_ptr())
            })?;
            apply_scales(handle, scales)?;
            check_dfti(unsafe { sys::DftiCommitDescriptor(handle) })
        };
        if let Err(e) = configure(handle) {
            unsafe {
                let mut h = handle;
                let _ = sys::DftiFreeDescriptor(&mut h);
            }
            return Err(e);
        }
        Ok(Self { handle, buffer_len, _marker: PhantomData })
    }

    /// Minimum buffer length: `1 + sum((dims[i] - 1) * strides[i])`.
    #[inline]
    #[must_use]
    pub fn buffer_len(&self) -> usize {
        self.buffer_len
    }

    /// Forward transform in place.
    pub fn forward_in_place(&mut self, x: &mut [Complex<T>]) -> Result<()> {
        if x.len() < self.buffer_len {
            return Err(Error::InvalidArgument("buffer too small for strided FFT plan"));
        }
        check_dfti(unsafe { sys::DftiComputeForward(self.handle, x.as_mut_ptr().cast()) })
    }

    /// Backward transform in place (un-normalized unless scaled).
    pub fn backward_in_place(&mut self, x: &mut [Complex<T>]) -> Result<()> {
        if x.len() < self.buffer_len {
            return Err(Error::InvalidArgument("buffer too small for strided FFT plan"));
        }
        check_dfti(unsafe { sys::DftiComputeBackward(self.handle, x.as_mut_ptr().cast()) })
    }
}

impl<T: FftScalar> Drop for StridedFftPlan<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = sys::DftiFreeDescriptor(&mut self.handle);
            }
        }
    }
}

// =====================================================================
// PackedRealFftPlan — 1-D in-place real FFT in CCS / PACK / PERM storage
// =====================================================================

/// Packed storage format of the spectrum of a 1-D real transform
/// (`DFTI_PACKED_FORMAT`). For an `n`-point transform with
/// `X[k] = re_k + i im_k`:
///
/// - [`Ccs`](Self::Ccs): `re_0, 0, re_1, im_1, ..., re_{n/2}, im_{n/2}`
///   (`2 * (n/2 + 1)` reals; the zero imaginary parts of `X[0]`, and of
///   `X[n/2]` for even `n`, are stored explicitly).
/// - [`Pack`](Self::Pack): `re_0, re_1, im_1, re_2, im_2, ...` (`n` reals;
///   `im_{n/2}` is absent for even `n`).
/// - [`Perm`](Self::Perm): like `Pack`, but for even `n` the real
///   `re_{n/2}` is stored second: `re_0, re_{n/2}, re_1, im_1, ...`;
///   identical to `Pack` for odd `n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RealPackedFormat {
    /// `DFTI_CCS_FORMAT`.
    Ccs,
    /// `DFTI_PACK_FORMAT`.
    Pack,
    /// `DFTI_PERM_FORMAT`.
    Perm,
}

impl RealPackedFormat {
    #[inline]
    fn as_dfti(self) -> sys::DFTI_CONFIG_VALUE::Type {
        match self {
            Self::Ccs => sys::DFTI_CONFIG_VALUE::DFTI_CCS_FORMAT,
            Self::Pack => sys::DFTI_CONFIG_VALUE::DFTI_PACK_FORMAT,
            Self::Perm => sys::DFTI_CONFIG_VALUE::DFTI_PERM_FORMAT,
        }
    }
}

/// 1-D in-place real FFT using a packed (real-array) spectrum format.
///
/// The same buffer holds the real signal on entry to
/// [`forward_in_place`](Self::forward_in_place) and the packed spectrum on
/// exit (and vice versa for [`backward_in_place`](Self::backward_in_place)).
/// The buffer must have at least [`buffer_len`](Self::buffer_len) elements
/// (`n + 2` for CCS with even `n`, otherwise `n`; for CCS the signal
/// occupies the first `n`). Contrast [`RealFftPlan`], which is out of place
/// and uses the CCE complex layout.
pub struct PackedRealFftPlan<T: FftScalar> {
    handle: DFTI_DESCRIPTOR_HANDLE,
    buffer_len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: FftScalar> Send for PackedRealFftPlan<T> {}

impl<T: FftScalar> PackedRealFftPlan<T> {
    /// Plan of length `n` with unit scaling.
    pub fn new(n: usize, format: RealPackedFormat) -> Result<Self> {
        Self::build(n, format, None)
    }

    /// Plan of length `n` with explicit forward / backward scales (see
    /// [`FftPlan::complex_nd_with_scales`]).
    pub fn with_scales(
        n: usize,
        format: RealPackedFormat,
        forward_scale: T,
        backward_scale: T,
    ) -> Result<Self> {
        Self::build(n, format, Some((forward_scale, backward_scale)))
    }

    fn build(n: usize, format: RealPackedFormat, scales: Option<(T, T)>) -> Result<Self> {
        if n == 0 {
            return Err(Error::InvalidArgument("FFT length must be positive"));
        }
        let n_long = c_long::try_from(n).map_err(|_| Error::DimensionOverflow)?;
        let mut handle: DFTI_DESCRIPTOR_HANDLE = ptr::null_mut();
        check_dfti(unsafe {
            sys::DftiCreateDescriptor(
                &mut handle,
                T::PRECISION.as_dfti(),
                sys::DFTI_CONFIG_VALUE::DFTI_REAL,
                1 as c_long,
                n_long,
            )
        })?;
        let configure = |handle: DFTI_DESCRIPTOR_HANDLE| -> Result<()> {
            check_dfti(unsafe {
                sys::DftiSetValue(
                    handle,
                    sys::DFTI_CONFIG_PARAM::DFTI_PLACEMENT,
                    sys::DFTI_CONFIG_VALUE::DFTI_INPLACE,
                )
            })?;
            check_dfti(unsafe {
                sys::DftiSetValue(
                    handle,
                    sys::DFTI_CONFIG_PARAM::DFTI_CONJUGATE_EVEN_STORAGE,
                    sys::DFTI_CONFIG_VALUE::DFTI_COMPLEX_REAL,
                )
            })?;
            check_dfti(unsafe {
                sys::DftiSetValue(
                    handle,
                    sys::DFTI_CONFIG_PARAM::DFTI_PACKED_FORMAT,
                    format.as_dfti(),
                )
            })?;
            apply_scales(handle, scales)?;
            check_dfti(unsafe { sys::DftiCommitDescriptor(handle) })
        };
        if let Err(e) = configure(handle) {
            unsafe {
                let mut h = handle;
                let _ = sys::DftiFreeDescriptor(&mut h);
            }
            return Err(e);
        }
        let buffer_len = match format {
            RealPackedFormat::Ccs => 2 * (n / 2 + 1),
            _ => n,
        };
        Ok(Self {
            handle,
            buffer_len,
            _marker: PhantomData,
        })
    }

    /// Minimum buffer length (in reals) the transform reads / writes.
    #[inline]
    #[must_use]
    pub fn buffer_len(&self) -> usize {
        self.buffer_len
    }

    /// Forward transform: real signal in, packed spectrum out, in place.
    pub fn forward_in_place(&mut self, buf: &mut [T]) -> Result<()> {
        self.check_len(buf.len())?;
        check_dfti(unsafe {
            sys::DftiComputeForward(self.handle, buf.as_mut_ptr().cast::<core::ffi::c_void>())
        })
    }

    /// Backward transform: packed spectrum in, real signal out, in place
    /// (un-normalized unless the plan was built with scales).
    pub fn backward_in_place(&mut self, buf: &mut [T]) -> Result<()> {
        self.check_len(buf.len())?;
        check_dfti(unsafe {
            sys::DftiComputeBackward(self.handle, buf.as_mut_ptr().cast::<core::ffi::c_void>())
        })
    }

    fn check_len(&self, len: usize) -> Result<()> {
        if len < self.buffer_len {
            return Err(Error::InvalidArgument(
                "packed FFT buffer shorter than buffer_len()",
            ));
        }
        Ok(())
    }
}

impl<T: FftScalar> Drop for PackedRealFftPlan<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = sys::DftiFreeDescriptor(&mut self.handle);
            }
        }
    }
}

// =====================================================================
// BatchedFftPlan — many 1-D complex transforms, with distance and stride
// =====================================================================

/// In-place batch of `howmany` 1-D complex transforms of length `n`
/// (`DFTI_NUMBER_OF_TRANSFORMS`, `DFTI_{INPUT,OUTPUT}_DISTANCE`,
/// `DFTI_{INPUT,OUTPUT}_STRIDES`).
///
/// Element `j` of transform `t` lives at index `t * distance + j * stride`
/// of the buffer. Elements not addressed that way are left untouched.
/// `distance` must be at least `(n - 1) * stride + 1` so that transforms do
/// not overlap (interleaved layouts, e.g. `distance = 1, stride = howmany`,
/// are therefore not accepted by this constructor).
pub struct BatchedFftPlan<T: FftScalar> {
    handle: DFTI_DESCRIPTOR_HANDLE,
    buffer_len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: FftScalar> Send for BatchedFftPlan<T> {}

impl<T: FftScalar> BatchedFftPlan<T> {
    /// Batch with unit scaling.
    pub fn new(n: usize, howmany: usize, distance: usize, stride: usize) -> Result<Self> {
        Self::build(n, howmany, distance, stride, None)
    }

    /// Batch with explicit forward / backward scales (per transform).
    pub fn with_scales(
        n: usize,
        howmany: usize,
        distance: usize,
        stride: usize,
        forward_scale: T,
        backward_scale: T,
    ) -> Result<Self> {
        Self::build(n, howmany, distance, stride, Some((forward_scale, backward_scale)))
    }

    fn build(
        n: usize,
        howmany: usize,
        distance: usize,
        stride: usize,
        scales: Option<(T, T)>,
    ) -> Result<Self> {
        if n == 0 || howmany == 0 || stride == 0 {
            return Err(Error::InvalidArgument(
                "FFT length, batch count and stride must be positive",
            ));
        }
        let extent = (n - 1)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(1))
            .ok_or(Error::DimensionOverflow)?;
        if distance < extent {
            return Err(Error::InvalidArgument(
                "batch distance smaller than one transform's extent (transforms would overlap)",
            ));
        }
        let buffer_len = (howmany - 1)
            .checked_mul(distance)
            .and_then(|v| v.checked_add(extent))
            .ok_or(Error::DimensionOverflow)?;
        let to_long = |v: usize| c_long::try_from(v).map_err(|_| Error::DimensionOverflow);
        let (n_l, h_l, d_l, s_l) = (to_long(n)?, to_long(howmany)?, to_long(distance)?, to_long(stride)?);
        let strides: [c_long; 2] = [0, s_l];

        let mut handle: DFTI_DESCRIPTOR_HANDLE = ptr::null_mut();
        check_dfti(unsafe {
            sys::DftiCreateDescriptor(
                &mut handle,
                T::PRECISION.as_dfti(),
                sys::DFTI_CONFIG_VALUE::DFTI_COMPLEX,
                1 as c_long,
                n_l,
            )
        })?;
        let configure = |handle: DFTI_DESCRIPTOR_HANDLE| -> Result<()> {
            use sys::DFTI_CONFIG_PARAM as P;
            check_dfti(unsafe {
                sys::DftiSetValue(handle, P::DFTI_PLACEMENT, sys::DFTI_CONFIG_VALUE::DFTI_INPLACE)
            })?;
            check_dfti(unsafe { sys::DftiSetValue(handle, P::DFTI_NUMBER_OF_TRANSFORMS, h_l) })?;
            check_dfti(unsafe { sys::DftiSetValue(handle, P::DFTI_INPUT_DISTANCE, d_l) })?;
            check_dfti(unsafe { sys::DftiSetValue(handle, P::DFTI_OUTPUT_DISTANCE, d_l) })?;
            check_dfti(unsafe {
                sys::DftiSetValue(handle, P::DFTI_INPUT_STRIDES, strides.as_ptr())
            })?;
            check_dfti(unsafe {
                sys::DftiSetValue(handle, P::DFTI_OUTPUT_STRIDES, strides.as_ptr())
            })?;
            apply_scales(handle, scales)?;
            check_dfti(unsafe { sys::DftiCommitDescriptor(handle) })
        };
        if let Err(e) = configure(handle) {
            unsafe {
                let mut h = handle;
                let _ = sys::DftiFreeDescriptor(&mut h);
            }
            return Err(e);
        }
        Ok(Self {
            handle,
            buffer_len,
            _marker: PhantomData,
        })
    }

    /// Minimum buffer length: `(howmany - 1) * distance + (n - 1) * stride + 1`.
    #[inline]
    #[must_use]
    pub fn buffer_len(&self) -> usize {
        self.buffer_len
    }

    /// Forward transform of every batch member, in place.
    pub fn forward_in_place(&mut self, x: &mut [Complex<T>]) -> Result<()> {
        self.check_len(x.len())?;
        check_dfti(unsafe { sys::DftiComputeForward(self.handle, x.as_mut_ptr().cast()) })
    }

    /// Backward transform of every batch member, in place (un-normalized
    /// unless the plan was built with scales).
    pub fn backward_in_place(&mut self, x: &mut [Complex<T>]) -> Result<()> {
        self.check_len(x.len())?;
        check_dfti(unsafe { sys::DftiComputeBackward(self.handle, x.as_mut_ptr().cast()) })
    }

    fn check_len(&self, len: usize) -> Result<()> {
        if len < self.buffer_len {
            return Err(Error::InvalidArgument(
                "batched FFT buffer shorter than buffer_len()",
            ));
        }
        Ok(())
    }
}

impl<T: FftScalar> Drop for BatchedFftPlan<T> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = sys::DftiFreeDescriptor(&mut self.handle);
            }
        }
    }
}

// =====================================================================
// BatchedRealFftPlan — many 1-D real transforms (real <-> CCE complex)
// =====================================================================

/// Out-of-place batch of `howmany` 1-D real transforms of length `n`.
/// Forward maps real transform `t` at `input[t * real_distance ..][..n]` to
/// its CCE spectrum at `out[t * complex_distance ..][..n/2 + 1]`; backward
/// reverses that. Elements outside those ranges are untouched. Both
/// distances must be at least one transform's extent (`n` real,
/// `n/2 + 1` complex), so members never overlap.
pub struct BatchedRealFftPlan<T: FftScalar> {
    fwd: DFTI_DESCRIPTOR_HANDLE,
    bwd: DFTI_DESCRIPTOR_HANDLE,
    real_len: usize,
    complex_len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: FftScalar> Send for BatchedRealFftPlan<T> {}

impl<T: FftScalar> BatchedRealFftPlan<T> {
    /// Batch with unit scaling.
    pub fn new(n: usize, howmany: usize, real_distance: usize, complex_distance: usize) -> Result<Self> {
        Self::build(n, howmany, real_distance, complex_distance, None)
    }

    /// Batch with explicit forward / backward scales (per transform).
    pub fn with_scales(
        n: usize,
        howmany: usize,
        real_distance: usize,
        complex_distance: usize,
        forward_scale: T,
        backward_scale: T,
    ) -> Result<Self> {
        Self::build(
            n,
            howmany,
            real_distance,
            complex_distance,
            Some((forward_scale, backward_scale)),
        )
    }

    fn build(
        n: usize,
        howmany: usize,
        rd: usize,
        cd: usize,
        scales: Option<(T, T)>,
    ) -> Result<Self> {
        if n == 0 || howmany == 0 {
            return Err(Error::InvalidArgument("FFT length and batch count must be positive"));
        }
        let h = cce_complex_len(n);
        if rd < n || cd < h {
            return Err(Error::InvalidArgument(
                "batch distance smaller than one transform's extent (transforms would overlap)",
            ));
        }
        let span = |dist: usize, extent: usize| -> Result<usize> {
            (howmany - 1)
                .checked_mul(dist)
                .and_then(|v| v.checked_add(extent))
                .ok_or(Error::DimensionOverflow)
        };
        let (real_len, complex_len) = (span(rd, n)?, span(cd, h)?);
        let to_long = |v: usize| c_long::try_from(v).map_err(|_| Error::DimensionOverflow);
        let (n_l, h_l, rd_l, cd_l) = (to_long(n)?, to_long(howmany)?, to_long(rd)?, to_long(cd)?);

        // DFTI's input / output distances describe the transform being
        // computed, so the backward direction needs its own descriptor with
        // the two distances swapped.
        let make = |in_d: c_long, out_d: c_long| -> Result<DFTI_DESCRIPTOR_HANDLE> {
            let mut handle: DFTI_DESCRIPTOR_HANDLE = ptr::null_mut();
            check_dfti(unsafe {
                sys::DftiCreateDescriptor(
                    &mut handle,
                    T::PRECISION.as_dfti(),
                    sys::DFTI_CONFIG_VALUE::DFTI_REAL,
                    1 as c_long,
                    n_l,
                )
            })?;
            let configure = |handle: DFTI_DESCRIPTOR_HANDLE| -> Result<()> {
                use sys::DFTI_CONFIG_PARAM as P;
                check_dfti(unsafe {
                    sys::DftiSetValue(handle, P::DFTI_PLACEMENT, sys::DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE)
                })?;
                check_dfti(unsafe {
                    sys::DftiSetValue(
                        handle,
                        P::DFTI_CONJUGATE_EVEN_STORAGE,
                        sys::DFTI_CONFIG_VALUE::DFTI_COMPLEX_COMPLEX,
                    )
                })?;
                check_dfti(unsafe { sys::DftiSetValue(handle, P::DFTI_NUMBER_OF_TRANSFORMS, h_l) })?;
                check_dfti(unsafe { sys::DftiSetValue(handle, P::DFTI_INPUT_DISTANCE, in_d) })?;
                check_dfti(unsafe { sys::DftiSetValue(handle, P::DFTI_OUTPUT_DISTANCE, out_d) })?;
                apply_scales(handle, scales)?;
                check_dfti(unsafe { sys::DftiCommitDescriptor(handle) })
            };
            if let Err(e) = configure(handle) {
                unsafe {
                    let mut hh = handle;
                    let _ = sys::DftiFreeDescriptor(&mut hh);
                }
                return Err(e);
            }
            Ok(handle)
        };
        let fwd = make(rd_l, cd_l)?;
        let bwd = match make(cd_l, rd_l) {
            Ok(h) => h,
            Err(e) => {
                unsafe {
                    let mut hh = fwd;
                    let _ = sys::DftiFreeDescriptor(&mut hh);
                }
                return Err(e);
            }
        };
        Ok(Self {
            fwd,
            bwd,
            real_len,
            complex_len,
            _marker: PhantomData,
        })
    }

    /// Minimum real buffer length: `(howmany - 1) * real_distance + n`.
    #[inline]
    #[must_use]
    pub fn real_buffer_len(&self) -> usize {
        self.real_len
    }

    /// Minimum complex buffer length: `(howmany - 1) * complex_distance + n/2 + 1`.
    #[inline]
    #[must_use]
    pub fn complex_buffer_len(&self) -> usize {
        self.complex_len
    }

    /// Forward: real signals in, CCE spectra out.
    pub fn forward(&mut self, input: &[T], out: &mut [Complex<T>]) -> Result<()> {
        self.check_len(input.len(), out.len())?;
        check_dfti(unsafe {
            sys::DftiComputeForward(
                self.fwd,
                input.as_ptr() as *mut core::ffi::c_void,
                out.as_mut_ptr().cast::<core::ffi::c_void>(),
            )
        })
    }

    /// Backward: CCE spectra in, real signals out (un-normalized unless
    /// scaled). The spectra must be conjugate-even.
    pub fn backward(&mut self, input: &[Complex<T>], out: &mut [T]) -> Result<()> {
        self.check_len(out.len(), input.len())?;
        check_dfti(unsafe {
            sys::DftiComputeBackward(
                self.bwd,
                input.as_ptr() as *mut core::ffi::c_void,
                out.as_mut_ptr().cast::<core::ffi::c_void>(),
            )
        })
    }

    fn check_len(&self, real: usize, complex: usize) -> Result<()> {
        if real < self.real_len || complex < self.complex_len {
            return Err(Error::InvalidArgument(
                "batched real FFT buffer shorter than real_buffer_len() / complex_buffer_len()",
            ));
        }
        Ok(())
    }
}

impl<T: FftScalar> Drop for BatchedRealFftPlan<T> {
    fn drop(&mut self) {
        for h in [&mut self.fwd, &mut self.bwd] {
            if !h.is_null() {
                unsafe {
                    let _ = sys::DftiFreeDescriptor(h);
                }
            }
        }
    }
}
