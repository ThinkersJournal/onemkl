// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pack / Compute GEMM API.
//!
//! When the same operand is used in many GEMMs (e.g. a fixed weight
//! matrix multiplied by many input batches), MKL can pack it once
//! into an internal layout and reuse the packed form in repeated
//! `gemm_compute` calls. This avoids re-doing the same data
//! reordering on every call and yields a meaningful speedup for
//! medium-sized matrices.
//!
//! Two-step usage:
//!
//! ```ignore
//! use onemkl::blas::packed::{gemm_compute_packed_a, PackedMatrix};
//! use onemkl::{Layout, Transpose};
//!
//! // Pack A once.
//! let packed_a = PackedMatrix::<f64>::pack_a(
//!     Layout::RowMajor, Transpose::NoTrans,
//!     m, n, k, alpha, &a_storage, lda,
//! ).unwrap();
//!
//! // Reuse in many GEMMs against different B / C.
//! for batch in batches {
//!     gemm_compute_packed_a(
//!         Layout::RowMajor, Transpose::NoTrans,
//!         m, n, k, &packed_a, &batch.b, ldb, 0.0, &mut batch.c, ldc,
//!     ).unwrap();
//! }
//! ```
//!
//! Pack/compute is available for f32 and f64 only — MKL doesn't ship
//! complex variants.

use core::marker::PhantomData;

use onemkl_sys::{self as sys, MKL_INT};

use crate::enums::{Layout, Transpose};
use crate::error::{Error, Result};

/// Which GEMM operand is being packed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackIdentifier {
    /// Pack the `A` operand (left-hand side).
    A,
    /// Pack the `B` operand (right-hand side).
    B,
}

impl PackIdentifier {
    #[inline]
    fn as_cblas(self) -> sys::CBLAS_IDENTIFIER::Type {
        match self {
            Self::A => sys::CBLAS_IDENTIFIER::CblasAMatrix,
            Self::B => sys::CBLAS_IDENTIFIER::CblasBMatrix,
        }
    }
}

/// Real scalar types supported by the pack / compute GEMM API.
#[allow(missing_docs)]
pub trait PackedGemmScalar: Copy + 'static {
    /// Raw `gemm_pack_get_size` call.
    ///
    /// # Safety
    ///
    /// No pointers. `identifier` must be `CblasAMatrix` or `CblasBMatrix` and `m`, `n`, `k` the GEMM dimensions (non-negative). The returned size is a **byte** count (the wrapper allocates it as bytes; unverified against MKL docs, but for `m = n = k = 100` `f64` it returned 5017728, far above the 80000 bytes of the matrix, so it includes blocking padding); callers must size `dest` of `gemm_pack` with at least this many bytes. The prototype `size_t cblas_?gemm_pack_get_size(CBLAS_IDENTIFIER, MKL_INT, MKL_INT, MKL_INT)` is in `mkl_cblas.h`.
    /// The MKL library must be linked and usable.
    unsafe fn gemm_pack_get_size(
        identifier: sys::CBLAS_IDENTIFIER::Type,
        m: MKL_INT,
        n: MKL_INT,
        k: MKL_INT,
    ) -> usize;

    /// Raw `gemm_pack` call.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads of the matrix `identifier` selects (`A`: `m x k`, `B`: `k x n`, after `trans`) with leading dimension `ld` per `layout`: `ld * cols` elements (column-major, `ld >= rows`) or `ld * rows` (row-major, `ld >= cols`).
    /// `dest` must be valid for writes of at least `gemm_pack_get_size(identifier, m, n, k)` bytes and suitably aligned as MKL requires (at least `Self` alignment; MKL recommends 64 bytes); it must not overlap `src`. `alpha` is folded into the packed data.
    /// The MKL library must be linked and usable.
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the oneMKL C signature cblas_?gemm_pack argument for argument"
    )]
    unsafe fn gemm_pack(
        layout: sys::CBLAS_LAYOUT::Type,
        identifier: sys::CBLAS_IDENTIFIER::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        m: MKL_INT,
        n: MKL_INT,
        k: MKL_INT,
        alpha: Self,
        src: *const Self,
        ld: MKL_INT,
        dest: *mut Self,
    );

    /// Raw `gemm_compute` call.
    ///
    /// # Safety
    ///
    /// `a` and `b` are either ordinary matrices or buffers produced by `gemm_pack` with the same `layout`, dimensions and operand `identifier`, matching `transa`/`transb` (`CblasPacked` for a packed operand); a packed buffer must not have been freed, resized or modified since packing, and the `alpha` used when packing is the one applied.
    /// For an unpacked operand the usual extent rule holds: `ld * cols` (column-major) or `ld * rows` (row-major) elements, `ld` at least the row/column length. `c` is `m x n` with `ldc`, read when `beta != 0` and written; it must be valid for `ldc * n` (column-major) or `ldc * m` (row-major) elements and must not overlap `a` or `b`.
    /// The MKL library must be linked and usable.
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the oneMKL C signature cblas_?gemm_compute argument for argument"
    )]
    unsafe fn gemm_compute(
        layout: sys::CBLAS_LAYOUT::Type,
        transa: MKL_INT,
        transb: MKL_INT,
        m: MKL_INT,
        n: MKL_INT,
        k: MKL_INT,
        a: *const Self,
        lda: MKL_INT,
        b: *const Self,
        ldb: MKL_INT,
        beta: Self,
        c: *mut Self,
        ldc: MKL_INT,
    );
}

impl PackedGemmScalar for f32 {
    unsafe fn gemm_pack_get_size(
        identifier: sys::CBLAS_IDENTIFIER::Type,
        m: MKL_INT,
        n: MKL_INT,
        k: MKL_INT,
    ) -> usize {
        unsafe { sys::cblas_sgemm_pack_get_size(identifier, m, n, k) }
    }
    unsafe fn gemm_pack(
        layout: sys::CBLAS_LAYOUT::Type,
        identifier: sys::CBLAS_IDENTIFIER::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        m: MKL_INT,
        n: MKL_INT,
        k: MKL_INT,
        alpha: Self,
        src: *const Self,
        ld: MKL_INT,
        dest: *mut Self,
    ) {
        unsafe { sys::cblas_sgemm_pack(layout, identifier, trans, m, n, k, alpha, src, ld, dest) }
    }
    unsafe fn gemm_compute(
        layout: sys::CBLAS_LAYOUT::Type,
        transa: MKL_INT,
        transb: MKL_INT,
        m: MKL_INT,
        n: MKL_INT,
        k: MKL_INT,
        a: *const Self,
        lda: MKL_INT,
        b: *const Self,
        ldb: MKL_INT,
        beta: Self,
        c: *mut Self,
        ldc: MKL_INT,
    ) {
        unsafe {
            sys::cblas_sgemm_compute(
                layout, transa, transb, m, n, k, a, lda, b, ldb, beta, c, ldc,
            )
        }
    }
}

impl PackedGemmScalar for f64 {
    unsafe fn gemm_pack_get_size(
        identifier: sys::CBLAS_IDENTIFIER::Type,
        m: MKL_INT,
        n: MKL_INT,
        k: MKL_INT,
    ) -> usize {
        unsafe { sys::cblas_dgemm_pack_get_size(identifier, m, n, k) }
    }
    unsafe fn gemm_pack(
        layout: sys::CBLAS_LAYOUT::Type,
        identifier: sys::CBLAS_IDENTIFIER::Type,
        trans: sys::CBLAS_TRANSPOSE::Type,
        m: MKL_INT,
        n: MKL_INT,
        k: MKL_INT,
        alpha: Self,
        src: *const Self,
        ld: MKL_INT,
        dest: *mut Self,
    ) {
        unsafe { sys::cblas_dgemm_pack(layout, identifier, trans, m, n, k, alpha, src, ld, dest) }
    }
    unsafe fn gemm_compute(
        layout: sys::CBLAS_LAYOUT::Type,
        transa: MKL_INT,
        transb: MKL_INT,
        m: MKL_INT,
        n: MKL_INT,
        k: MKL_INT,
        a: *const Self,
        lda: MKL_INT,
        b: *const Self,
        ldb: MKL_INT,
        beta: Self,
        c: *mut Self,
        ldc: MKL_INT,
    ) {
        unsafe {
            sys::cblas_dgemm_compute(
                layout, transa, transb, m, n, k, a, lda, b, ldb, beta, c, ldc,
            )
        }
    }
}

/// A GEMM operand pre-packed into MKL's internal layout. The packed
/// form encodes the `alpha` scaling that was applied at pack time;
/// pass `1.0` if you want to apply scaling separately at compute
/// time.
pub struct PackedMatrix<T: PackedGemmScalar> {
    /// Backing storage. The packed format is opaque; we expose only
    /// the typed pointer to MKL's compute kernel.
    buffer: Vec<T>,
    identifier: PackIdentifier,
    m: usize,
    n: usize,
    k: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: PackedGemmScalar + Send> Send for PackedMatrix<T> {}

impl<T: PackedGemmScalar> PackedMatrix<T> {
    /// Pack the `A` operand of a future GEMM. `src` / `ld` describe
    /// the unpacked source matrix in the usual CBLAS convention; the
    /// resulting [`PackedMatrix`] can then be used in repeated
    /// [`gemm_compute_packed_a`] calls.
    #[allow(
        clippy::too_many_arguments,
        reason = "stable public surface: published since v0.1.0 with 8 arguments; changing the signature breaks consumers"
    )]
    pub fn pack_a(
        layout: Layout,
        trans: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        src: &[T],
        ld: usize,
    ) -> Result<Self> {
        Self::pack(PackArgs {
            layout,
            identifier: PackIdentifier::A,
            trans,
            m,
            n,
            k,
            alpha,
            src,
            ld,
        })
    }

    /// Pack the `B` operand of a future GEMM.
    #[allow(
        clippy::too_many_arguments,
        reason = "stable public surface: published since v0.1.0 with 8 arguments; changing the signature breaks consumers"
    )]
    pub fn pack_b(
        layout: Layout,
        trans: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        src: &[T],
        ld: usize,
    ) -> Result<Self> {
        Self::pack(PackArgs {
            layout,
            identifier: PackIdentifier::B,
            trans,
            m,
            n,
            k,
            alpha,
            src,
            ld,
        })
    }

    fn pack(args: PackArgs<'_, T>) -> Result<Self> {
        let PackArgs {
            layout,
            identifier,
            trans,
            m,
            n,
            k,
            alpha,
            src,
            ld,
        } = args;
        let m_i: MKL_INT = m.try_into().map_err(|_| Error::DimensionOverflow)?;
        let n_i: MKL_INT = n.try_into().map_err(|_| Error::DimensionOverflow)?;
        let k_i: MKL_INT = k.try_into().map_err(|_| Error::DimensionOverflow)?;
        let ld_i: MKL_INT = ld.try_into().map_err(|_| Error::DimensionOverflow)?;

        let bytes_needed = unsafe { T::gemm_pack_get_size(identifier.as_cblas(), m_i, n_i, k_i) };
        let elem_size = core::mem::size_of::<T>();
        if elem_size == 0 {
            return Err(Error::InvalidArgument("zero-sized scalar"));
        }
        let elements_needed = bytes_needed.div_ceil(elem_size);
        // Validate src is large enough for the unpacked source. The
        // source has shape (m × k) for A or (k × n) for B, possibly
        // transposed.
        let (src_rows, src_cols) = match identifier {
            PackIdentifier::A => match trans {
                Transpose::NoTrans => (m, k),
                _ => (k, m),
            },
            PackIdentifier::B => match trans {
                Transpose::NoTrans => (k, n),
                _ => (n, k),
            },
        };
        let contiguous = match layout {
            Layout::RowMajor => src_cols,
            Layout::ColMajor => src_rows,
        };
        if contiguous > 0 && ld < contiguous {
            return Err(Error::InvalidArgument(
                "ld is smaller than the contiguous dimension of the pack source",
            ));
        }
        let needed_src = match layout {
            Layout::RowMajor => src_rows.saturating_mul(ld),
            Layout::ColMajor => src_cols.saturating_mul(ld),
        };
        if src.len() < needed_src {
            return Err(Error::InvalidArgument(
                "src buffer too small for pack source shape",
            ));
        }

        let mut buffer: Vec<T> = Vec::with_capacity(elements_needed);
        // Safety: MKL writes `bytes_needed` bytes into the buffer;
        // we set len after the pack call. The buffer's allocation
        // is sized to fit those bytes (rounded up to an element
        // count).
        unsafe {
            T::gemm_pack(
                layout.as_cblas(),
                identifier.as_cblas(),
                trans.as_cblas(),
                m_i,
                n_i,
                k_i,
                alpha,
                src.as_ptr(),
                ld_i,
                buffer.as_mut_ptr(),
            );
            buffer.set_len(elements_needed);
        }
        Ok(Self {
            buffer,
            identifier,
            m,
            n,
            k,
            _marker: PhantomData,
        })
    }

    /// Which operand this packed matrix represents.
    #[inline]
    #[must_use]
    pub fn identifier(&self) -> PackIdentifier {
        self.identifier
    }
}

/// `rows × cols` dense operand with leading dimension `ld` must fit `len`.
fn check_dense(
    layout: Layout,
    rows: usize,
    cols: usize,
    ld: usize,
    len: usize,
    name: &'static str,
) -> Result<()> {
    let (contiguous, strided) = match layout {
        Layout::RowMajor => (cols, rows),
        Layout::ColMajor => (rows, cols),
    };
    if contiguous == 0 || strided == 0 {
        return Ok(());
    }
    let needed = (strided - 1)
        .checked_mul(ld)
        .and_then(|v| v.checked_add(contiguous))
        .ok_or(Error::DimensionOverflow)?;
    if ld < contiguous || len < needed {
        return Err(Error::InvalidArgument(match name {
            "a" => "a is too small for its leading dimension and shape",
            "b" => "b is too small for its leading dimension and shape",
            _ => "c is too small for its leading dimension and shape",
        }));
    }
    Ok(())
}

const CBLAS_PACKED: MKL_INT = 151; // sys::CBLAS_STORAGE::CblasPacked

#[inline]
fn trans_as_compute_int(t: Transpose) -> MKL_INT {
    t.as_cblas() as MKL_INT
}

/// Run `C ← op(A_packed) · op(B) + beta · C` where `A_packed` was
/// produced by [`PackedMatrix::pack_a`].
///
/// `transb` applies to the unpacked `B` operand. `alpha` is baked
/// into `A_packed` (it was applied at pack time), so this function
/// has no `alpha` parameter.
#[allow(
    clippy::too_many_arguments,
    reason = "stable public surface: published since v0.1.0 with 11 arguments; changing the signature breaks consumers"
)]
pub fn gemm_compute_packed_a<T: PackedGemmScalar>(
    layout: Layout,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    a_packed: &PackedMatrix<T>,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) -> Result<()> {
    if a_packed.identifier != PackIdentifier::A {
        return Err(Error::InvalidArgument(
            "a_packed must be packed as the A operand",
        ));
    }
    if a_packed.m != m || a_packed.n != n || a_packed.k != k {
        return Err(Error::InvalidArgument(
            "a_packed shape does not match the requested compute shape",
        ));
    }
    let (b_rows, b_cols) = match transb {
        Transpose::NoTrans => (k, n),
        _ => (n, k),
    };
    check_dense(layout, b_rows, b_cols, ldb, b.len(), "b")?;
    check_dense(layout, m, n, ldc, c.len(), "c")?;
    let m_i: MKL_INT = m.try_into().map_err(|_| Error::DimensionOverflow)?;
    let n_i: MKL_INT = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let k_i: MKL_INT = k.try_into().map_err(|_| Error::DimensionOverflow)?;
    let ldb_i: MKL_INT = ldb.try_into().map_err(|_| Error::DimensionOverflow)?;
    let ldc_i: MKL_INT = ldc.try_into().map_err(|_| Error::DimensionOverflow)?;
    // For packed compute, MKL ignores lda but still reads it as a
    // parameter; pass m to keep it well-defined.
    let lda_i: MKL_INT = m_i;
    unsafe {
        T::gemm_compute(
            layout.as_cblas(),
            CBLAS_PACKED,
            trans_as_compute_int(transb),
            m_i,
            n_i,
            k_i,
            a_packed.buffer.as_ptr(),
            lda_i,
            b.as_ptr(),
            ldb_i,
            beta,
            c.as_mut_ptr(),
            ldc_i,
        );
    }
    Ok(())
}

/// Run `C ← op(A) · op(B_packed) + beta · C` where `B_packed` was
/// produced by [`PackedMatrix::pack_b`].
#[allow(
    clippy::too_many_arguments,
    reason = "stable public surface: published since v0.1.0 with 11 arguments; changing the signature breaks consumers"
)]
pub fn gemm_compute_packed_b<T: PackedGemmScalar>(
    layout: Layout,
    transa: Transpose,
    m: usize,
    n: usize,
    k: usize,
    a: &[T],
    lda: usize,
    b_packed: &PackedMatrix<T>,
    beta: T,
    c: &mut [T],
    ldc: usize,
) -> Result<()> {
    if b_packed.identifier != PackIdentifier::B {
        return Err(Error::InvalidArgument(
            "b_packed must be packed as the B operand",
        ));
    }
    if b_packed.m != m || b_packed.n != n || b_packed.k != k {
        return Err(Error::InvalidArgument(
            "b_packed shape does not match the requested compute shape",
        ));
    }
    let (a_rows, a_cols) = match transa {
        Transpose::NoTrans => (m, k),
        _ => (k, m),
    };
    check_dense(layout, a_rows, a_cols, lda, a.len(), "a")?;
    check_dense(layout, m, n, ldc, c.len(), "c")?;
    let m_i: MKL_INT = m.try_into().map_err(|_| Error::DimensionOverflow)?;
    let n_i: MKL_INT = n.try_into().map_err(|_| Error::DimensionOverflow)?;
    let k_i: MKL_INT = k.try_into().map_err(|_| Error::DimensionOverflow)?;
    let lda_i: MKL_INT = lda.try_into().map_err(|_| Error::DimensionOverflow)?;
    let ldc_i: MKL_INT = ldc.try_into().map_err(|_| Error::DimensionOverflow)?;
    let ldb_i: MKL_INT = n_i;
    unsafe {
        T::gemm_compute(
            layout.as_cblas(),
            trans_as_compute_int(transa),
            CBLAS_PACKED,
            m_i,
            n_i,
            k_i,
            a.as_ptr(),
            lda_i,
            b_packed.buffer.as_ptr(),
            ldb_i,
            beta,
            c.as_mut_ptr(),
            ldc_i,
        );
    }
    Ok(())
}

/// The arguments of the private `pack` helper, gathered in one struct.
struct PackArgs<'a, T> {
    layout: Layout,
    identifier: PackIdentifier,
    trans: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    src: &'a [T],
    ld: usize,
}
