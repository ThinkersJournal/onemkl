//! 1-D convolution and correlation tasks (`vsl?Conv*` / `vsl?Corr*`).
//!
//! The simplest entry points are [`convolve_1d`] and [`correlate_1d`],
//! which build, execute, and tear down a task in one call. For
//! repeated execution against changing inputs, build a [`Conv1d`] /
//! [`Corr1d`] task once and call [`Conv1d::execute`] / [`Corr1d::execute`]
//! repeatedly.
//!
//! ```no_run
//! use onemkl::rng::convolution::{convolve_1d, ConvMode};
//!
//! let x = [1.0_f64, 2.0, 3.0];
//! let y = [1.0_f64, 1.0];
//! let z = convolve_1d::<f64>(ConvMode::Auto, &x, &y).unwrap();
//! // Convolution length is x.len() + y.len() - 1 = 4.
//! ```

use core::ffi::c_int;
use core::marker::PhantomData;
use core::ptr;

use onemkl_sys::{self as sys, VSLConvTaskPtr, VSLCorrTaskPtr};

use crate::error::{Error, Result};

/// Algorithm used by a convolution task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ConvMode {
    /// Let MKL pick between direct and FFT.
    #[default]
    Auto,
    /// Force direct evaluation (good for short inputs).
    Direct,
    /// Force FFT-based evaluation (good for long inputs).
    Fft,
}

impl ConvMode {
    #[inline]
    fn as_int(self) -> c_int {
        // Cast each constant individually so the match arms are
        // type-uniform regardless of whether bindgen typed the
        // underlying constant as i32 or u32.
        match self {
            Self::Auto => sys::VSL_CONV_MODE_AUTO as c_int,
            Self::Direct => sys::VSL_CONV_MODE_DIRECT as c_int,
            Self::Fft => sys::VSL_CONV_MODE_FFT as c_int,
        }
    }
}

/// Algorithm used by a correlation task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CorrMode {
    /// Let MKL pick between direct and FFT.
    #[default]
    Auto,
    /// Force direct evaluation.
    Direct,
    /// Force FFT-based evaluation.
    Fft,
}

impl CorrMode {
    #[inline]
    fn as_int(self) -> c_int {
        match self {
            Self::Auto => sys::VSL_CORR_MODE_AUTO as c_int,
            Self::Direct => sys::VSL_CORR_MODE_DIRECT as c_int,
            Self::Fft => sys::VSL_CORR_MODE_FFT as c_int,
        }
    }
}

/// Real scalar types supported by 1-D convolution / correlation tasks.
#[allow(missing_docs)]
pub trait ConvCorrScalar: Copy + 'static {
    /// Raw `conv_new_task_1d` call.
    ///
    /// # Safety
    ///
    /// `task` must be valid for a write of one `VSLConvTaskPtr`. `mode` must be a `VSL_CONV_MODE_*` constant. `xshape`, `yshape` and `zshape` are lengths (>= 1) of `x`, `y` and the result `z`; `zshape` must equal the output length MKL defines for the chosen mode (for full convolution `xshape + yshape - 1`).
    /// On a non-zero return `*task` must not be used. A non-zero return is the VSL status. The MKL library must be linked and usable.
    /// Extents not listed as verified in this block are unverified (from MKL documentation).
    unsafe fn conv_new_task_1d(
        task: *mut VSLConvTaskPtr,
        mode: c_int,
        xshape: c_int,
        yshape: c_int,
        zshape: c_int,
    ) -> c_int;

    /// Raw `corr_new_task_1d` call.
    ///
    /// # Safety
    ///
    /// `task` must be valid for a write of one `VSLCorrTaskPtr`. `mode` must be a `VSL_CORR_MODE_*` constant. `xshape`, `yshape` and `zshape` are lengths (>= 1) of `x`, `y` and the result `z`; `zshape` must equal the output length MKL defines for the chosen mode.
    /// On a non-zero return `*task` must not be used. A non-zero return is the VSL status. The MKL library must be linked and usable.
    /// Extents not listed as verified in this block are unverified (from MKL documentation).
    unsafe fn corr_new_task_1d(
        task: *mut VSLCorrTaskPtr,
        mode: c_int,
        xshape: c_int,
        yshape: c_int,
        zshape: c_int,
    ) -> c_int;

    /// Raw `conv_exec_1d` call.
    ///
    /// # Safety
    ///
    /// `task` must be a live task of the matching kind created by `conv_new_task_1d` and not yet deleted, and not used from another thread during the call.
    /// `x` must be valid for reads of `1 + (xshape - 1) * |xstride|` elements, `y` of `1 + (yshape - 1) * |ystride|`, and `z` for writes of `1 + (zshape - 1) * |zstride|`, with the shapes given when the task was created and all strides non-zero. `z` must not overlap `x` or `y`.
    /// A non-zero return is the VSL status. The MKL library must be linked and usable.
    /// **Verified** by guard-page probe (smallest buffer for which the call did not fault: `x`, `y`, `z` each `1 + (shape - 1) * stride` elements, strides 1 and 2, VSL_CONV_MODE_DIRECT, x = 8, y = 5, z = 12; oneMKL 2026.2, Linux x86_64). Extents not listed as verified in this block are unverified (from MKL documentation).
    #[allow(clippy::too_many_arguments)]
    unsafe fn conv_exec_1d(
        task: VSLConvTaskPtr,
        x: *const Self,
        xstride: c_int,
        y: *const Self,
        ystride: c_int,
        z: *mut Self,
        zstride: c_int,
    ) -> c_int;

    /// Raw `corr_exec_1d` call.
    ///
    /// # Safety
    ///
    /// `task` must be a live task of the matching kind created by `corr_new_task_1d` and not yet deleted, and not used from another thread during the call.
    /// `x` must be valid for reads of `1 + (xshape - 1) * |xstride|` elements, `y` of `1 + (yshape - 1) * |ystride|`, and `z` for writes of `1 + (zshape - 1) * |zstride|`, with the shapes given when the task was created and all strides non-zero. `z` must not overlap `x` or `y`.
    /// A non-zero return is the VSL status. The MKL library must be linked and usable.
    /// Same extent rule as `conv_exec_1d` (verified there for convolution only; for correlation unverified).
    #[allow(clippy::too_many_arguments)]
    unsafe fn corr_exec_1d(
        task: VSLCorrTaskPtr,
        x: *const Self,
        xstride: c_int,
        y: *const Self,
        ystride: c_int,
        z: *mut Self,
        zstride: c_int,
    ) -> c_int;
}

macro_rules! impl_conv_corr {
    ($ty:ty,
        conv_new=$conv_new:ident, corr_new=$corr_new:ident,
        conv_exec=$conv_exec:ident, corr_exec=$corr_exec:ident
    ) => {
        impl ConvCorrScalar for $ty {
            unsafe fn conv_new_task_1d(
                task: *mut VSLConvTaskPtr,
                mode: c_int,
                xshape: c_int,
                yshape: c_int,
                zshape: c_int,
            ) -> c_int {
                unsafe { sys::$conv_new(task, mode, xshape, yshape, zshape) }
            }
            unsafe fn corr_new_task_1d(
                task: *mut VSLCorrTaskPtr,
                mode: c_int,
                xshape: c_int,
                yshape: c_int,
                zshape: c_int,
            ) -> c_int {
                unsafe { sys::$corr_new(task, mode, xshape, yshape, zshape) }
            }
            unsafe fn conv_exec_1d(
                task: VSLConvTaskPtr,
                x: *const Self,
                xstride: c_int,
                y: *const Self,
                ystride: c_int,
                z: *mut Self,
                zstride: c_int,
            ) -> c_int {
                unsafe { sys::$conv_exec(task, x, xstride, y, ystride, z, zstride) }
            }
            unsafe fn corr_exec_1d(
                task: VSLCorrTaskPtr,
                x: *const Self,
                xstride: c_int,
                y: *const Self,
                ystride: c_int,
                z: *mut Self,
                zstride: c_int,
            ) -> c_int {
                unsafe { sys::$corr_exec(task, x, xstride, y, ystride, z, zstride) }
            }
        }
    };
}

impl_conv_corr!(
    f32,
    conv_new = vslsConvNewTask1D,
    corr_new = vslsCorrNewTask1D,
    conv_exec = vslsConvExec1D,
    corr_exec = vslsCorrExec1D
);
impl_conv_corr!(
    f64,
    conv_new = vsldConvNewTask1D,
    corr_new = vsldCorrNewTask1D,
    conv_exec = vsldConvExec1D,
    corr_exec = vsldCorrExec1D
);

/// Owned 1-D convolution task. Can be reused across multiple
/// [`execute`](Self::execute) calls.
pub struct Conv1d<T: ConvCorrScalar> {
    task: VSLConvTaskPtr,
    xshape: usize,
    yshape: usize,
    zshape: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: ConvCorrScalar + Send> Send for Conv1d<T> {}

impl<T: ConvCorrScalar> Conv1d<T> {
    /// Build a new convolution task for inputs of length `xshape` /
    /// `yshape` and an output buffer of length `zshape`. The natural
    /// full-convolution length is `xshape + yshape - 1`.
    pub fn new(mode: ConvMode, xshape: usize, yshape: usize, zshape: usize) -> Result<Self> {
        let xs: c_int = xshape.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ys: c_int = yshape.try_into().map_err(|_| Error::DimensionOverflow)?;
        let zs: c_int = zshape.try_into().map_err(|_| Error::DimensionOverflow)?;
        let mut task: VSLConvTaskPtr = ptr::null_mut();
        let status = unsafe { T::conv_new_task_1d(&mut task, mode.as_int(), xs, ys, zs) };
        check_vsl(status)?;
        Ok(Self {
            task,
            xshape,
            yshape,
            zshape,
            _marker: PhantomData,
        })
    }

    /// Execute the task: `z = x * y` (convolution).
    /// All three buffers may be strided; pass `xstride = ystride =
    /// zstride = 1` for the dense case.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        x: &[T],
        xstride: usize,
        y: &[T],
        ystride: usize,
        z: &mut [T],
        zstride: usize,
    ) -> Result<()> {
        if x.len() < (self.xshape - 1) * xstride + 1 {
            return Err(Error::InvalidArgument(
                "x is too short for xstride * xshape",
            ));
        }
        if y.len() < (self.yshape - 1) * ystride + 1 {
            return Err(Error::InvalidArgument(
                "y is too short for ystride * yshape",
            ));
        }
        if z.len() < (self.zshape - 1) * zstride + 1 {
            return Err(Error::InvalidArgument(
                "z is too short for zstride * zshape",
            ));
        }
        let xs: c_int = xstride.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ys: c_int = ystride.try_into().map_err(|_| Error::DimensionOverflow)?;
        let zs: c_int = zstride.try_into().map_err(|_| Error::DimensionOverflow)?;
        let status = unsafe {
            T::conv_exec_1d(
                self.task,
                x.as_ptr(),
                xs,
                y.as_ptr(),
                ys,
                z.as_mut_ptr(),
                zs,
            )
        };
        check_vsl(status)
    }
}

impl<T: ConvCorrScalar> Drop for Conv1d<T> {
    fn drop(&mut self) {
        if !self.task.is_null() {
            unsafe {
                let _ = sys::vslConvDeleteTask(&mut self.task);
            }
        }
    }
}

/// Owned 1-D correlation task. Reusable across [`execute`](Self::execute)
/// calls.
pub struct Corr1d<T: ConvCorrScalar> {
    task: VSLCorrTaskPtr,
    xshape: usize,
    yshape: usize,
    zshape: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: ConvCorrScalar + Send> Send for Corr1d<T> {}

impl<T: ConvCorrScalar> Corr1d<T> {
    /// Build a new correlation task. See [`Conv1d::new`] for shape
    /// conventions — they apply equally here.
    pub fn new(mode: CorrMode, xshape: usize, yshape: usize, zshape: usize) -> Result<Self> {
        let xs: c_int = xshape.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ys: c_int = yshape.try_into().map_err(|_| Error::DimensionOverflow)?;
        let zs: c_int = zshape.try_into().map_err(|_| Error::DimensionOverflow)?;
        let mut task: VSLCorrTaskPtr = ptr::null_mut();
        let status = unsafe { T::corr_new_task_1d(&mut task, mode.as_int(), xs, ys, zs) };
        check_vsl(status)?;
        Ok(Self {
            task,
            xshape,
            yshape,
            zshape,
            _marker: PhantomData,
        })
    }

    /// Execute the task: `z = x ⋆ y` (cross-correlation).
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        x: &[T],
        xstride: usize,
        y: &[T],
        ystride: usize,
        z: &mut [T],
        zstride: usize,
    ) -> Result<()> {
        if x.len() < (self.xshape - 1) * xstride + 1 {
            return Err(Error::InvalidArgument(
                "x is too short for xstride * xshape",
            ));
        }
        if y.len() < (self.yshape - 1) * ystride + 1 {
            return Err(Error::InvalidArgument(
                "y is too short for ystride * yshape",
            ));
        }
        if z.len() < (self.zshape - 1) * zstride + 1 {
            return Err(Error::InvalidArgument(
                "z is too short for zstride * zshape",
            ));
        }
        let xs: c_int = xstride.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ys: c_int = ystride.try_into().map_err(|_| Error::DimensionOverflow)?;
        let zs: c_int = zstride.try_into().map_err(|_| Error::DimensionOverflow)?;
        let status = unsafe {
            T::corr_exec_1d(
                self.task,
                x.as_ptr(),
                xs,
                y.as_ptr(),
                ys,
                z.as_mut_ptr(),
                zs,
            )
        };
        check_vsl(status)
    }
}

impl<T: ConvCorrScalar> Drop for Corr1d<T> {
    fn drop(&mut self) {
        if !self.task.is_null() {
            unsafe {
                let _ = sys::vslCorrDeleteTask(&mut self.task);
            }
        }
    }
}

/// One-shot 1-D convolution: `z = x * y`. Output length is
/// `x.len() + y.len() - 1`.
pub fn convolve_1d<T: ConvCorrScalar + Default>(
    mode: ConvMode,
    x: &[T],
    y: &[T],
) -> Result<Vec<T>> {
    if x.is_empty() || y.is_empty() {
        return Err(Error::InvalidArgument("x and y must each be non-empty"));
    }
    let zshape = x.len() + y.len() - 1;
    let mut task = Conv1d::<T>::new(mode, x.len(), y.len(), zshape)?;
    let mut z: Vec<T> = (0..zshape).map(|_| T::default()).collect();
    task.execute(x, 1, y, 1, &mut z, 1)?;
    Ok(z)
}

/// One-shot 1-D cross-correlation: `z = x ⋆ y`. Output length is
/// `x.len() + y.len() - 1`.
pub fn correlate_1d<T: ConvCorrScalar + Default>(
    mode: CorrMode,
    x: &[T],
    y: &[T],
) -> Result<Vec<T>> {
    if x.is_empty() || y.is_empty() {
        return Err(Error::InvalidArgument("x and y must each be non-empty"));
    }
    let zshape = x.len() + y.len() - 1;
    let mut task = Corr1d::<T>::new(mode, x.len(), y.len(), zshape)?;
    let mut z: Vec<T> = (0..zshape).map(|_| T::default()).collect();
    task.execute(x, 1, y, 1, &mut z, 1)?;
    Ok(z)
}

#[inline]
fn check_vsl(status: c_int) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::PardisoStatus(status))
    }
}
