//! Compact BLAS — batched dense kernels packed for SIMD.
//!
//! When you need to apply the same small dense kernel (a 4×4 GEMM, a
//! 8×8 LU, etc.) to many independent matrices simultaneously, the
//! Compact BLAS API achieves better throughput than calling the
//! standard BLAS in a loop or even using the pointer-array batched
//! API. It packs `nm` matrices into a single buffer with SIMD-width
//! interleaving so each underlying instruction processes a
//! `pack-width`-sized batch in lockstep.
//!
//! Three pack widths are supported:
//!
//! | [`CompactFormat`] | SIMD width | Best on |
//! | --- | --- | --- |
//! | `Sse` | 4 (f32) / 2 (f64) | older Intel cores |
//! | `Avx` | 8 / 4 | most CPUs since Haswell |
//! | `Avx512` | 16 / 8 | Skylake-X, Sapphire Rapids, … |
//!
//! Typical flow:
//!
//! 1. Compute the packed-buffer size with [`get_size`].
//! 2. Pack a slice of `*const T` pointers into the buffer with
//!    [`gepack`].
//! 3. Run the desired compact kernel ([`gemm`], …) on the packed
//!    buffers.
//! 4. Unpack with [`geunpack`] when results are needed back as
//!    regular matrices.
//!
//! Compact BLAS is real-only for the full lifecycle (`f32` / `f64`).
//! Complex GEMM exists but lacks a matching pack / unpack pair, so
//! the Rust wrapper restricts this module to real types.

use core::ffi::c_int;

use onemkl_sys as sys;

use crate::enums::{Diag, Layout, Side, Transpose, UpLo};
use crate::error::{Error, Result};

/// SIMD pack width to use when laying out compact matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompactFormat {
    /// 128-bit SSE pack (4 f32 / 2 f64 lanes).
    Sse,
    /// 256-bit AVX pack (8 f32 / 4 f64 lanes).
    Avx,
    /// 512-bit AVX-512 pack (16 f32 / 8 f64 lanes).
    Avx512,
}

impl CompactFormat {
    #[inline]
    fn as_mkl(self) -> sys::MKL_COMPACT_PACK::Type {
        match self {
            Self::Sse => sys::MKL_COMPACT_PACK::MKL_COMPACT_SSE,
            Self::Avx => sys::MKL_COMPACT_PACK::MKL_COMPACT_AVX,
            Self::Avx512 => sys::MKL_COMPACT_PACK::MKL_COMPACT_AVX512,
        }
    }
}

#[inline]
fn map_layout(layout: Layout) -> sys::MKL_LAYOUT::Type {
    match layout {
        Layout::RowMajor => sys::MKL_LAYOUT::MKL_ROW_MAJOR,
        Layout::ColMajor => sys::MKL_LAYOUT::MKL_COL_MAJOR,
    }
}

#[inline]
fn map_transpose(t: Transpose) -> sys::MKL_TRANSPOSE::Type {
    match t {
        Transpose::NoTrans => sys::MKL_TRANSPOSE::MKL_NOTRANS,
        Transpose::Trans => sys::MKL_TRANSPOSE::MKL_TRANS,
        Transpose::ConjTrans => sys::MKL_TRANSPOSE::MKL_CONJTRANS,
    }
}

/// Real scalar types supported by the Compact BLAS lifecycle.
#[allow(missing_docs)]
pub trait CompactScalar: Copy + 'static {
    unsafe fn get_size_compact(
        ld: c_int,
        sd: c_int,
        format: sys::MKL_COMPACT_PACK::Type,
        nm: c_int,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    unsafe fn gepack_compact(
        layout: sys::MKL_LAYOUT::Type,
        rows: c_int,
        cols: c_int,
        a: *const *const Self,
        lda: c_int,
        ap: *mut Self,
        ldap: c_int,
        format: sys::MKL_COMPACT_PACK::Type,
        nm: c_int,
    );

    #[allow(clippy::too_many_arguments)]
    unsafe fn geunpack_compact(
        layout: sys::MKL_LAYOUT::Type,
        rows: c_int,
        cols: c_int,
        a: *const *mut Self,
        lda: c_int,
        ap: *const Self,
        ldap: c_int,
        format: sys::MKL_COMPACT_PACK::Type,
        nm: c_int,
    );

    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_compact(
        layout: sys::MKL_LAYOUT::Type,
        transa: sys::MKL_TRANSPOSE::Type,
        transb: sys::MKL_TRANSPOSE::Type,
        m: c_int, n: c_int, k: c_int,
        alpha: Self,
        a: *const Self, ldap: c_int,
        b: *const Self, ldbp: c_int,
        beta: Self,
        c: *mut Self, ldcp: c_int,
        format: sys::MKL_COMPACT_PACK::Type,
        nm: c_int,
    );
}

impl CompactScalar for f32 {
    unsafe fn get_size_compact(
        ld: c_int, sd: c_int,
        format: sys::MKL_COMPACT_PACK::Type, nm: c_int,
    ) -> c_int {
        unsafe { sys::mkl_sget_size_compact(ld, sd, format, nm) }
    }
    unsafe fn gepack_compact(
        layout: sys::MKL_LAYOUT::Type, rows: c_int, cols: c_int,
        a: *const *const Self, lda: c_int,
        ap: *mut Self, ldap: c_int,
        format: sys::MKL_COMPACT_PACK::Type, nm: c_int,
    ) {
        unsafe {
            sys::mkl_sgepack_compact(layout, rows, cols, a, lda, ap, ldap, format, nm)
        }
    }
    unsafe fn geunpack_compact(
        layout: sys::MKL_LAYOUT::Type, rows: c_int, cols: c_int,
        a: *const *mut Self, lda: c_int,
        ap: *const Self, ldap: c_int,
        format: sys::MKL_COMPACT_PACK::Type, nm: c_int,
    ) {
        unsafe {
            sys::mkl_sgeunpack_compact(layout, rows, cols, a, lda, ap, ldap, format, nm)
        }
    }
    unsafe fn gemm_compact(
        layout: sys::MKL_LAYOUT::Type,
        transa: sys::MKL_TRANSPOSE::Type, transb: sys::MKL_TRANSPOSE::Type,
        m: c_int, n: c_int, k: c_int,
        alpha: Self,
        a: *const Self, ldap: c_int,
        b: *const Self, ldbp: c_int,
        beta: Self, c: *mut Self, ldcp: c_int,
        format: sys::MKL_COMPACT_PACK::Type, nm: c_int,
    ) {
        unsafe {
            sys::mkl_sgemm_compact(
                layout, transa, transb, m, n, k, alpha,
                a, ldap, b, ldbp, beta, c, ldcp, format, nm,
            )
        }
    }
}

impl CompactScalar for f64 {
    unsafe fn get_size_compact(
        ld: c_int, sd: c_int,
        format: sys::MKL_COMPACT_PACK::Type, nm: c_int,
    ) -> c_int {
        unsafe { sys::mkl_dget_size_compact(ld, sd, format, nm) }
    }
    unsafe fn gepack_compact(
        layout: sys::MKL_LAYOUT::Type, rows: c_int, cols: c_int,
        a: *const *const Self, lda: c_int,
        ap: *mut Self, ldap: c_int,
        format: sys::MKL_COMPACT_PACK::Type, nm: c_int,
    ) {
        unsafe {
            sys::mkl_dgepack_compact(layout, rows, cols, a, lda, ap, ldap, format, nm)
        }
    }
    unsafe fn geunpack_compact(
        layout: sys::MKL_LAYOUT::Type, rows: c_int, cols: c_int,
        a: *const *mut Self, lda: c_int,
        ap: *const Self, ldap: c_int,
        format: sys::MKL_COMPACT_PACK::Type, nm: c_int,
    ) {
        unsafe {
            sys::mkl_dgeunpack_compact(layout, rows, cols, a, lda, ap, ldap, format, nm)
        }
    }
    unsafe fn gemm_compact(
        layout: sys::MKL_LAYOUT::Type,
        transa: sys::MKL_TRANSPOSE::Type, transb: sys::MKL_TRANSPOSE::Type,
        m: c_int, n: c_int, k: c_int,
        alpha: Self,
        a: *const Self, ldap: c_int,
        b: *const Self, ldbp: c_int,
        beta: Self, c: *mut Self, ldcp: c_int,
        format: sys::MKL_COMPACT_PACK::Type, nm: c_int,
    ) {
        unsafe {
            sys::mkl_dgemm_compact(
                layout, transa, transb, m, n, k, alpha,
                a, ldap, b, ldbp, beta, c, ldcp, format, nm,
            )
        }
    }
}

/// Number of scalars needed to hold `num_matrices` compact-packed
/// matrices of shape `ld × sd` (leading dimension × short dimension)
/// in the given format. Wraps `mkl_*get_size_compact`.
pub fn get_size<T: CompactScalar>(
    ld: usize,
    sd: usize,
    format: CompactFormat,
    num_matrices: usize,
) -> Result<usize> {
    let ld_i: c_int = ld.try_into().map_err(|_| Error::DimensionOverflow)?;
    let sd_i: c_int = sd.try_into().map_err(|_| Error::DimensionOverflow)?;
    let nm_i: c_int = num_matrices
        .try_into()
        .map_err(|_| Error::DimensionOverflow)?;
    let bytes = unsafe { T::get_size_compact(ld_i, sd_i, format.as_mkl(), nm_i) };
    if bytes < 0 {
        return Err(Error::LapackComputationFailure { info: bytes });
    }
    let elem_size = core::mem::size_of::<T>();
    if elem_size == 0 {
        return Err(Error::InvalidArgument("zero-sized scalar"));
    }
    Ok((bytes as usize).div_ceil(elem_size))
}

/// Pack `num_matrices` regular matrices (one per pointer in `a`)
/// into the compact buffer `ap`. Wraps `mkl_*gepack_compact`.
///
/// `ldap` is the leading dimension of the compact buffer (use the
/// value returned by [`get_size`] / `num_matrices` arithmetic, or
/// the value MKL's documentation specifies for the chosen format).
///
/// # Safety
///
/// Each pointer in `a` must point to a buffer of at least the
/// matrix's storage size and remain valid for the duration of this
/// call.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gepack<T: CompactScalar>(
    layout: Layout,
    rows: usize,
    cols: usize,
    a: &[*const T],
    lda: usize,
    ap: &mut [T],
    ldap: usize,
    format: CompactFormat,
) -> Result<()> {
    let rows_i: c_int = rows.try_into().map_err(|_| Error::DimensionOverflow)?;
    let cols_i: c_int = cols.try_into().map_err(|_| Error::DimensionOverflow)?;
    let lda_i: c_int = lda.try_into().map_err(|_| Error::DimensionOverflow)?;
    let ldap_i: c_int = ldap.try_into().map_err(|_| Error::DimensionOverflow)?;
    let nm_i: c_int = a.len().try_into().map_err(|_| Error::DimensionOverflow)?;
    unsafe {
        T::gepack_compact(
            map_layout(layout),
            rows_i, cols_i,
            a.as_ptr(),
            lda_i,
            ap.as_mut_ptr(),
            ldap_i,
            format.as_mkl(),
            nm_i,
        );
    }
    Ok(())
}

/// Unpack a compact buffer into individual matrices addressed by
/// `a` (one `*mut T` per matrix). Wraps `mkl_*geunpack_compact`.
///
/// # Safety
///
/// Each pointer in `a` must point to a buffer of at least the
/// matrix's storage size and remain valid for the duration of this
/// call.
#[allow(clippy::too_many_arguments)]
pub unsafe fn geunpack<T: CompactScalar>(
    layout: Layout,
    rows: usize,
    cols: usize,
    a: &[*mut T],
    lda: usize,
    ap: &[T],
    ldap: usize,
    format: CompactFormat,
) -> Result<()> {
    let rows_i: c_int = rows.try_into().map_err(|_| Error::DimensionOverflow)?;
    let cols_i: c_int = cols.try_into().map_err(|_| Error::DimensionOverflow)?;
    let lda_i: c_int = lda.try_into().map_err(|_| Error::DimensionOverflow)?;
    let ldap_i: c_int = ldap.try_into().map_err(|_| Error::DimensionOverflow)?;
    let nm_i: c_int = a.len().try_into().map_err(|_| Error::DimensionOverflow)?;
    unsafe {
        T::geunpack_compact(
            map_layout(layout),
            rows_i, cols_i,
            a.as_ptr(),
            lda_i,
            ap.as_ptr(),
            ldap_i,
            format.as_mkl(),
            nm_i,
        );
    }
    Ok(())
}

/// Compact GEMM: perform `num_matrices` GEMMs `C ← α · op(A) · op(B)
/// + β · C` in lockstep on packed buffers. Wraps `mkl_*gemm_compact`.
#[allow(clippy::too_many_arguments)]
pub fn gemm<T: CompactScalar>(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    ap: &[T],
    ldap: usize,
    bp: &[T],
    ldbp: usize,
    beta: T,
    cp: &mut [T],
    ldcp: usize,
    format: CompactFormat,
    num_matrices: usize,
) -> Result<()> {
    let m_i: c_int = m.try_into().map_err(|_| Error::DimensionOverflow)?;
    let n_i: c_int = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let k_i: c_int = k.try_into().map_err(|_| Error::DimensionOverflow)?;
    let ldap_i: c_int = ldap.try_into().map_err(|_| Error::DimensionOverflow)?;
    let ldbp_i: c_int = ldbp.try_into().map_err(|_| Error::DimensionOverflow)?;
    let ldcp_i: c_int = ldcp.try_into().map_err(|_| Error::DimensionOverflow)?;
    let nm_i: c_int = num_matrices
        .try_into()
        .map_err(|_| Error::DimensionOverflow)?;
    unsafe {
        T::gemm_compact(
            map_layout(layout),
            map_transpose(transa),
            map_transpose(transb),
            m_i, n_i, k_i, alpha,
            ap.as_ptr(), ldap_i,
            bp.as_ptr(), ldbp_i,
            beta,
            cp.as_mut_ptr(), ldcp_i,
            format.as_mkl(),
            nm_i,
        );
    }
    Ok(())
}


#[inline]
fn map_side(s: Side) -> sys::MKL_SIDE::Type {
    match s {
        Side::Left => sys::MKL_SIDE::MKL_LEFT,
        Side::Right => sys::MKL_SIDE::MKL_RIGHT,
    }
}

#[inline]
fn map_uplo(u: UpLo) -> sys::MKL_UPLO::Type {
    match u {
        UpLo::Upper => sys::MKL_UPLO::MKL_UPPER,
        UpLo::Lower => sys::MKL_UPLO::MKL_LOWER,
    }
}

#[inline]
fn map_diag(d: Diag) -> sys::MKL_DIAG::Type {
    match d {
        Diag::NonUnit => sys::MKL_DIAG::MKL_NONUNIT,
        Diag::Unit => sys::MKL_DIAG::MKL_UNIT,
    }
}

/// Factorisation / solve kernels beyond GEMM (`f32` / `f64`).
#[allow(missing_docs)]
pub trait CompactFactorScalar: CompactScalar {
    #[allow(clippy::too_many_arguments)]
    unsafe fn trsm_compact(
        layout: sys::MKL_LAYOUT::Type,
        side: sys::MKL_SIDE::Type,
        uplo: sys::MKL_UPLO::Type,
        transa: sys::MKL_TRANSPOSE::Type,
        diag: sys::MKL_DIAG::Type,
        m: c_int, n: c_int,
        alpha: Self,
        a: *const Self, ldap: c_int,
        b: *mut Self, ldbp: c_int,
        format: sys::MKL_COMPACT_PACK::Type, nm: c_int,
    );
    #[allow(clippy::too_many_arguments)]
    unsafe fn potrf_compact(
        layout: sys::MKL_LAYOUT::Type,
        uplo: sys::MKL_UPLO::Type,
        n: c_int,
        ap: *mut Self, ldap: c_int,
        info: *mut c_int,
        format: sys::MKL_COMPACT_PACK::Type, nm: c_int,
    );
    #[allow(clippy::too_many_arguments)]
    unsafe fn getrfnp_compact(
        layout: sys::MKL_LAYOUT::Type,
        m: c_int, n: c_int,
        ap: *mut Self, ldap: c_int,
        info: *mut c_int,
        format: sys::MKL_COMPACT_PACK::Type, nm: c_int,
    );
}

macro_rules! impl_compact_factor {
    ($t:ty, $trsm:ident, $potrf:ident, $getrfnp:ident) => {
        impl CompactFactorScalar for $t {
            unsafe fn trsm_compact(
                layout: sys::MKL_LAYOUT::Type,
                side: sys::MKL_SIDE::Type,
                uplo: sys::MKL_UPLO::Type,
                transa: sys::MKL_TRANSPOSE::Type,
                diag: sys::MKL_DIAG::Type,
                m: c_int, n: c_int,
                alpha: Self,
                a: *const Self, ldap: c_int,
                b: *mut Self, ldbp: c_int,
                format: sys::MKL_COMPACT_PACK::Type, nm: c_int,
            ) {
                unsafe {
                    sys::$trsm(
                        layout, side, uplo, transa, diag, m, n, alpha,
                        a, ldap, b, ldbp, format, nm,
                    )
                }
            }
            unsafe fn potrf_compact(
                layout: sys::MKL_LAYOUT::Type,
                uplo: sys::MKL_UPLO::Type,
                n: c_int,
                ap: *mut Self, ldap: c_int,
                info: *mut c_int,
                format: sys::MKL_COMPACT_PACK::Type, nm: c_int,
            ) {
                unsafe { sys::$potrf(layout, uplo, n, ap, ldap, info, format, nm) }
            }
            unsafe fn getrfnp_compact(
                layout: sys::MKL_LAYOUT::Type,
                m: c_int, n: c_int,
                ap: *mut Self, ldap: c_int,
                info: *mut c_int,
                format: sys::MKL_COMPACT_PACK::Type, nm: c_int,
            ) {
                unsafe { sys::$getrfnp(layout, m, n, ap, ldap, info, format, nm) }
            }
        }
    };
}
impl_compact_factor!(f32, mkl_strsm_compact, mkl_spotrf_compact, mkl_sgetrfnp_compact);
impl_compact_factor!(f64, mkl_dtrsm_compact, mkl_dpotrf_compact, mkl_dgetrfnp_compact);

/// Packed buffer length needed for a `rows x cols` matrix in `layout`
/// with packed leading dimension `ldap`, or an error if `ldap` is smaller
/// than the matrix's leading dimension.
fn packed_len<T: CompactScalar>(
    layout: Layout,
    rows: usize,
    cols: usize,
    ldap: usize,
    format: CompactFormat,
    nm: usize,
) -> Result<usize> {
    let (ld, sd) = match layout {
        Layout::ColMajor => (rows, cols),
        Layout::RowMajor => (cols, rows),
    };
    if ldap < ld {
        return Err(Error::InvalidArgument("ldap is smaller than the matrix's leading dimension"));
    }
    get_size::<T>(ldap, sd, format, nm)
}

fn to_c(v: usize) -> Result<c_int> {
    v.try_into().map_err(|_| Error::DimensionOverflow)
}

/// Compact TRSM: solve `op(A) X = alpha B` (or `X op(A) = alpha B`) for
/// `num_matrices` triangular systems in lockstep; `B` is overwritten by
/// `X`. `A` is `m x m` for `Side::Left`, `n x n` for `Side::Right`; `B` is
/// `m x n`. Both are in the compact layout of `format`. Buffer lengths are
/// checked against [`get_size`]. Wraps `mkl_*trsm_compact`.
#[allow(clippy::too_many_arguments)]
pub fn trsm<T: CompactFactorScalar>(
    layout: Layout,
    side: Side,
    uplo: UpLo,
    transa: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: T,
    ap: &[T],
    ldap: usize,
    bp: &mut [T],
    ldbp: usize,
    format: CompactFormat,
    num_matrices: usize,
) -> Result<()> {
    let k = match side {
        Side::Left => m,
        Side::Right => n,
    };
    if ap.len() < packed_len::<T>(layout, k, k, ldap, format, num_matrices)? {
        return Err(Error::InvalidArgument("ap is smaller than the compact size of A"));
    }
    if bp.len() < packed_len::<T>(layout, m, n, ldbp, format, num_matrices)? {
        return Err(Error::InvalidArgument("bp is smaller than the compact size of B"));
    }
    let (m_i, n_i, ldap_i, ldbp_i, nm_i) =
        (to_c(m)?, to_c(n)?, to_c(ldap)?, to_c(ldbp)?, to_c(num_matrices)?);
    unsafe {
        T::trsm_compact(
            map_layout(layout), map_side(side), map_uplo(uplo),
            map_transpose(transa), map_diag(diag),
            m_i, n_i, alpha,
            ap.as_ptr(), ldap_i,
            bp.as_mut_ptr(), ldbp_i,
            format.as_mkl(), nm_i,
        );
    }
    Ok(())
}

/// Compact Cholesky: factor `num_matrices` symmetric positive-definite
/// `n x n` matrices in place (`uplo` selects the stored triangle). Returns
/// MKL's `info` (0 on success; MKL's compact interface reports one value
/// for the whole batch). Wraps `mkl_*potrf_compact`.
pub fn potrf<T: CompactFactorScalar>(
    layout: Layout,
    uplo: UpLo,
    n: usize,
    ap: &mut [T],
    ldap: usize,
    format: CompactFormat,
    num_matrices: usize,
) -> Result<i32> {
    if ap.len() < packed_len::<T>(layout, n, n, ldap, format, num_matrices)? {
        return Err(Error::InvalidArgument("ap is smaller than the compact size of A"));
    }
    let (n_i, ldap_i, nm_i) = (to_c(n)?, to_c(ldap)?, to_c(num_matrices)?);
    let mut info: c_int = 0;
    unsafe {
        T::potrf_compact(
            map_layout(layout), map_uplo(uplo), n_i,
            ap.as_mut_ptr(), ldap_i, &mut info,
            format.as_mkl(), nm_i,
        );
    }
    Ok(info)
}

/// Compact LU without pivoting: factor `num_matrices` `m x n` matrices in
/// place (`L` unit-lower and `U` share the buffer). Returns MKL's `info`.
/// No pivoting means the result is only meaningful for matrices whose
/// leading minors are non-singular. Wraps `mkl_*getrfnp_compact`.
pub fn getrfnp<T: CompactFactorScalar>(
    layout: Layout,
    m: usize,
    n: usize,
    ap: &mut [T],
    ldap: usize,
    format: CompactFormat,
    num_matrices: usize,
) -> Result<i32> {
    if ap.len() < packed_len::<T>(layout, m, n, ldap, format, num_matrices)? {
        return Err(Error::InvalidArgument("ap is smaller than the compact size of A"));
    }
    let (m_i, n_i, ldap_i, nm_i) = (to_c(m)?, to_c(n)?, to_c(ldap)?, to_c(num_matrices)?);
    let mut info: c_int = 0;
    unsafe {
        T::getrfnp_compact(
            map_layout(layout), m_i, n_i,
            ap.as_mut_ptr(), ldap_i, &mut info,
            format.as_mkl(), nm_i,
        );
    }
    Ok(info)
}
