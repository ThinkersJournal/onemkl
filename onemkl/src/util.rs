// SPDX-License-Identifier: MIT OR Apache-2.0
//! Internal helpers shared across the public modules.

use onemkl_sys::MKL_INT;

use crate::error::{Error, Result};

/// Convert an unsigned dimension into [`MKL_INT`], failing on overflow.
#[inline]
pub(crate) fn dim_to_mkl_int(value: usize) -> Result<MKL_INT> {
    MKL_INT::try_from(value).map_err(|_| Error::DimensionOverflow)
}

/// Convert a signed stride into [`MKL_INT`], failing on overflow.
///
/// BLAS allows negative strides (the vector is traversed from the end), so
/// we keep the value signed.
#[inline]
pub(crate) fn stride_to_mkl_int(stride: i64) -> Result<MKL_INT> {
    MKL_INT::try_from(stride).map_err(|_| Error::DimensionOverflow)
}

/// Element count derivable from a slice of length `slice_len` accessed
/// with stride `inc`. Returns `0` if the slice is empty; errors if the
/// stride is `0`.
#[inline]
pub(crate) fn derive_n(slice_len: usize, inc: i64) -> Result<usize> {
    if inc == 0 {
        return Err(Error::InvalidArgument("vector stride must be non-zero"));
    }
    if slice_len == 0 {
        return Ok(0);
    }
    let abs_inc = inc.unsigned_abs() as usize;
    Ok((slice_len - 1) / abs_inc + 1)
}

/// Validate that two BLAS vectors with separate strides address the same
/// number of elements. Returns the agreed-upon `n`.
#[inline]
pub(crate) fn check_two_vectors<T, U>(x: &[T], incx: i64, y: &[U], incy: i64) -> Result<usize> {
    let nx = derive_n(x.len(), incx)?;
    let ny = derive_n(y.len(), incy)?;
    if nx != ny {
        return Err(Error::InvalidArgument(
            "x and y vectors must contain the same number of elements (length / |stride|)",
        ));
    }
    Ok(nx)
}

/// Smallest contiguous buffer length that holds `n` elements at absolute
/// stride `|inc|`.
#[inline]
pub(crate) fn vector_min_len(n: usize, inc: i64) -> usize {
    if n == 0 {
        return 0;
    }
    1 + (n - 1) * inc.unsigned_abs() as usize
}

/// Narrow a routine status (`MKL_INT`) to the `i32` carried by [`crate::Error`]
/// variants. Under `lp64` this is the identity; under `ilp64` a status that
/// does not fit saturates (statuses are small).
#[inline]
#[allow(
    clippy::useless_conversion,
    reason = "MKL_INT is i32 under lp64 and i64 under ilp64; the conversion narrows only under ilp64"
)]
pub(crate) fn status_i32(status: MKL_INT) -> i32 {
    i32::try_from(status).unwrap_or(if status < 0 { i32::MIN } else { i32::MAX })
}

/// Validate compressed (CSR/CSC-style) index arrays before they reach an
/// index-chasing FFI routine (`MKL_INT` is `i32` or `i64`):
///
/// * `ptr.len() == outer + 1` and `ptr[0] == base`,
/// * `ptr` is non-decreasing and `ptr[outer] - base <= idx.len()`,
/// * every `idx` entry within `ptr[outer] - base` lies in
///   `base .. base + inner_bound`.
///
/// Returns the number of stored entries (`ptr[outer] - base`).
#[cfg(any(
    feature = "sparse",
    feature = "dss",
    feature = "pardiso",
    feature = "feast",
    feature = "preconditioners"
))]
pub(crate) fn validate_compressed(
    ptr: &[onemkl_sys::MKL_INT],
    idx: &[onemkl_sys::MKL_INT],
    base: onemkl_sys::MKL_INT,
    outer: usize,
    inner_bound: usize,
) -> crate::error::Result<usize> {
    use crate::error::Error;
    if ptr.len() != outer + 1 {
        return Err(Error::InvalidArgument(
            "pointer array must have length outer + 1",
        ));
    }
    if ptr[0] != base {
        return Err(Error::InvalidArgument(
            "pointer array must start at the index base",
        ));
    }
    let mut prev = ptr[0];
    for &p in &ptr[1..] {
        if p < prev {
            return Err(Error::InvalidArgument(
                "pointer array must be non-decreasing",
            ));
        }
        prev = p;
    }
    let nnz = (prev - base) as usize;
    if nnz > idx.len() {
        return Err(Error::InvalidArgument(
            "pointer array ends past the end of the index array",
        ));
    }
    let lo = base;
    let hi = (base as i128) + (inner_bound as i128);
    for &c in &idx[..nnz] {
        if c < lo || (c as i128) >= hi {
            return Err(Error::InvalidArgument("index entry out of range"));
        }
    }
    Ok(nnz)
}

/// Check that a `rows × cols` dense operand with leading dimension `ld`
/// (row-major when `row_major`) fits a `len`-element buffer.
#[cfg(feature = "sparse")]
pub(crate) fn check_dense_extent(
    row_major: bool,
    rows: usize,
    cols: usize,
    ld: usize,
    len: usize,
    what: &'static str,
) -> crate::error::Result<()> {
    use crate::error::Error;
    let (contig, strided) = if row_major {
        (cols, rows)
    } else {
        (rows, cols)
    };
    if contig == 0 || strided == 0 {
        return Ok(());
    }
    let need = (strided - 1)
        .checked_mul(ld)
        .and_then(|v| v.checked_add(contig))
        .ok_or(Error::DimensionOverflow)?;
    if ld < contig || len < need {
        return Err(Error::InvalidArgument(what));
    }
    Ok(())
}
