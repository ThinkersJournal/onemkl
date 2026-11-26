//! Scalar dispatch trait for Inspector-Executor Sparse BLAS.

use core::ffi::c_int;

use num_complex::{Complex32, Complex64};
use onemkl_sys::{
    self as sys, matrix_descr, sparse_index_base_t, sparse_layout_t, sparse_matrix_t,
    sparse_operation_t, sparse_status_t,
};

use crate::scalar::Scalar;

/// Scalar types supported by the IE sparse BLAS.
///
/// Implemented for [`f32`], [`f64`], [`Complex32`], and [`Complex64`].
/// Methods are `unsafe` thunks that forward to the corresponding
/// `mkl_sparse_*` C symbol.
#[allow(missing_docs)]
pub trait SparseScalar: Scalar {
    /// Raw `sparse_create_csr` call.
    ///
    /// # Safety
    ///
    /// `a` must be valid for a write of one `sparse_matrix_t`.
    /// **Pointer retention:** MKL does NOT copy the index and value arrays; the handle stores these pointers. Every array passed here must therefore stay allocated, at a stable address, with unchanged index contents, and must not be mutably aliased by Rust references while MKL may read them, until `mkl_sparse_destroy` has been called on the created handle.
    /// `indexing` must match how the index arrays are actually based (zero or one).
    /// `rows_start` and `rows_end` must each hold `rows` entries (`rows_end[i]` is the exclusive end of row `i`); `col_indx` and `values` must each hold `rows_end[rows-1] - indexing` entries, with every column index in `[0, cols)` after removing the base.
    /// The MKL library must be linked and usable. The returned status is MKL's; on a non-success status `*a` must not be used.
    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_create_csr(
        a: *mut sparse_matrix_t,
        indexing: sparse_index_base_t::Type,
        rows: c_int,
        cols: c_int,
        rows_start: *mut c_int,
        rows_end: *mut c_int,
        col_indx: *mut c_int,
        values: *mut Self,
    ) -> sparse_status_t::Type;

    /// Raw `sparse_create_coo` call.
    ///
    /// # Safety
    ///
    /// `a` must be valid for a write of one `sparse_matrix_t`.
    /// **Pointer retention:** MKL does NOT copy the index and value arrays; the handle stores these pointers. Every array passed here must therefore stay allocated, at a stable address, with unchanged index contents, and must not be mutably aliased by Rust references while MKL may read them, until `mkl_sparse_destroy` has been called on the created handle.
    /// `indexing` must match how the index arrays are actually based (zero or one).
    /// `row_indx`, `col_indx` and `values` must each hold `nnz` entries, with every row index in `[0, rows)` and column index in `[0, cols)` after removing the base.
    /// The MKL library must be linked and usable. The returned status is MKL's; on a non-success status `*a` must not be used.
    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_create_coo(
        a: *mut sparse_matrix_t,
        indexing: sparse_index_base_t::Type,
        rows: c_int,
        cols: c_int,
        nnz: c_int,
        row_indx: *mut c_int,
        col_indx: *mut c_int,
        values: *mut Self,
    ) -> sparse_status_t::Type;

    /// Raw `sparse_create_csc` call.
    ///
    /// # Safety
    ///
    /// `a` must be valid for a write of one `sparse_matrix_t`.
    /// **Pointer retention:** MKL does NOT copy the index and value arrays; the handle stores these pointers. Every array passed here must therefore stay allocated, at a stable address, with unchanged index contents, and must not be mutably aliased by Rust references while MKL may read them, until `mkl_sparse_destroy` has been called on the created handle.
    /// `indexing` must match how the index arrays are actually based (zero or one).
    /// `cols_start` and `cols_end` must each hold `cols` entries; `row_indx` and `values` must each hold `cols_end[cols-1] - indexing` entries, with every row index in `[0, rows)` after removing the base.
    /// The MKL library must be linked and usable. The returned status is MKL's; on a non-success status `*a` must not be used.
    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_create_csc(
        a: *mut sparse_matrix_t,
        indexing: sparse_index_base_t::Type,
        rows: c_int,
        cols: c_int,
        cols_start: *mut c_int,
        cols_end: *mut c_int,
        row_indx: *mut c_int,
        values: *mut Self,
    ) -> sparse_status_t::Type;

    /// Raw `sparse_create_bsr` call.
    ///
    /// # Safety
    ///
    /// `a` must be valid for a write of one `sparse_matrix_t`.
    /// **Pointer retention:** MKL does NOT copy the index and value arrays; the handle stores these pointers. Every array passed here must therefore stay allocated, at a stable address, with unchanged index contents, and must not be mutably aliased by Rust references while MKL may read them, until `mkl_sparse_destroy` has been called on the created handle.
    /// `indexing` must match how the index arrays are actually based (zero or one).
    /// `rows` and `cols` count blocks, not scalar rows/columns; `block_size` is the square block edge and `block_layout` the in-block storage order. `rows_start` and `rows_end` must each hold `rows` entries; `col_indx` must hold one entry per stored block and `values` `block_size * block_size` entries per stored block.
    /// The MKL library must be linked and usable. The returned status is MKL's; on a non-success status `*a` must not be used.
    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_create_bsr(
        a: *mut sparse_matrix_t,
        indexing: sparse_index_base_t::Type,
        block_layout: onemkl_sys::sparse_layout_t::Type,
        rows: c_int,
        cols: c_int,
        block_size: c_int,
        rows_start: *mut c_int,
        rows_end: *mut c_int,
        col_indx: *mut c_int,
        values: *mut Self,
    ) -> sparse_status_t::Type;

    /// Raw `sparse_mv` call.
    ///
    /// # Safety
    ///
    /// `a` must be a live handle created by one of the `sparse_create_*` calls (or produced by MKL) and not yet destroyed; `descr` must describe its structure (type, fill mode, diagonal).
    /// `x` must be valid for reads of `cols(op(A))` elements and `y` for reads (when `beta != 0`) and writes of `rows(op(A))` elements, where `op(A)` is `A` for the non-transpose operation and `A^T`/`A^H` otherwise. `x` and `y` must not overlap.
    /// The MKL library must be linked and usable. The returned status is MKL's; on a non-success status `*a` must not be used.
    unsafe fn sparse_mv(
        op: sparse_operation_t::Type,
        alpha: Self,
        a: sparse_matrix_t,
        descr: matrix_descr,
        x: *const Self,
        beta: Self,
        y: *mut Self,
    ) -> sparse_status_t::Type;

    /// Raw `sparse_mm` call.
    ///
    /// # Safety
    ///
    /// `a` must be a live handle created by one of the `sparse_create_*` calls and not yet destroyed; `descr` must describe its structure.
    /// `x` is a dense matrix of `cols(op(A))` rows by `columns` columns and `y` one of `rows(op(A))` rows by `columns` columns, both stored per `layout` (row-major: leading dimension is the row stride and must be `>= columns`; column-major: it is the column stride and must be `>=` the row count). `x` must be valid for reads of the whole strided extent, `y` for reads (when `beta != 0`) and writes. `x` and `y` must not overlap.
    /// The MKL library must be linked and usable. The returned status is MKL's; on a non-success status `*a` must not be used.
    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_mm(
        op: sparse_operation_t::Type,
        alpha: Self,
        a: sparse_matrix_t,
        descr: matrix_descr,
        layout: sparse_layout_t::Type,
        x: *const Self,
        columns: c_int,
        ldx: c_int,
        beta: Self,
        y: *mut Self,
        ldy: c_int,
    ) -> sparse_status_t::Type;

    /// Raw `sparse_trsv` call.
    ///
    /// # Safety
    ///
    /// `a` must be a live, square handle created by one of the `sparse_create_*` calls and not yet destroyed; `descr` must describe it as triangular (with fill mode and diagonal kind) and the diagonal must be non-singular as MKL requires.
    /// `x` must be valid for reads and `y` for writes of `n` elements, where `n` is the order of `A`; `x` and `y` must not partially overlap (do not assume in-place solve is supported).
    /// The MKL library must be linked and usable. The returned status is MKL's; on a non-success status `*a` must not be used.
    unsafe fn sparse_trsv(
        op: sparse_operation_t::Type,
        alpha: Self,
        a: sparse_matrix_t,
        descr: matrix_descr,
        x: *const Self,
        y: *mut Self,
    ) -> sparse_status_t::Type;

    /// Raw `sparse_add` call.
    ///
    /// # Safety
    ///
    /// `a` and `b` must be live handles of identical dimensions, created by `sparse_create_*` or by MKL, and not yet destroyed. `c` must be valid for a write of one `sparse_matrix_t`.
    /// On success `*c` is a new handle owned by the caller, who must release it with `mkl_sparse_destroy`; it may reference data owned by `a` and `b`, so neither source nor their backing arrays may be freed first.
    /// The MKL library must be linked and usable. The returned status is MKL's; on a non-success status `*a` must not be used.
    unsafe fn sparse_add(
        op: sparse_operation_t::Type,
        a: sparse_matrix_t,
        alpha: Self,
        b: sparse_matrix_t,
        c: *mut sparse_matrix_t,
    ) -> sparse_status_t::Type;

    /// Raw `sparse_dotmv` call.
    ///
    /// # Safety
    ///
    /// `a` must be a live, square handle created by `sparse_create_*` and not yet destroyed; `descr` must describe it.
    /// `x` must be valid for reads of `cols(op(A))` elements and `y` for reads (when `beta != 0`) and writes of `rows(op(A))` elements; `d` must be valid for a write of exactly one `Self` (the dot product `x . y`). `x`, `y` and `d` must not overlap.
    /// The MKL library must be linked and usable. The returned status is MKL's; on a non-success status `*a` must not be used.
    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_dotmv(
        op: sparse_operation_t::Type,
        alpha: Self,
        a: sparse_matrix_t,
        descr: matrix_descr,
        x: *const Self,
        beta: Self,
        y: *mut Self,
        d: *mut Self,
    ) -> sparse_status_t::Type;

    /// Raw `sparse_symgs` call.
    ///
    /// # Safety
    ///
    /// `a` must be a live, square, symmetric-described handle (`descr` fill mode and diagonal set) created by `sparse_create_*` and not yet destroyed.
    /// `b` must be valid for reads and `x` for reads and writes of `n` elements, where `n` is the order of `A`; `x` carries the initial guess in and the updated iterate out. `b` and `x` must not overlap.
    /// The MKL library must be linked and usable. The returned status is MKL's; on a non-success status `*a` must not be used.
    unsafe fn sparse_symgs(
        op: sparse_operation_t::Type,
        a: sparse_matrix_t,
        descr: matrix_descr,
        alpha: Self,
        b: *const Self,
        x: *mut Self,
    ) -> sparse_status_t::Type;

    /// Raw `sparse_symgs_mv` call.
    ///
    /// # Safety
    ///
    /// `a` must be a live, square, symmetric-described handle created by `sparse_create_*` and not yet destroyed.
    /// `b` must be valid for reads, and `x` and `y` for writes (`x` also read as the initial guess), of `n` elements, where `n` is the order of `A`; `y` receives `A * x`. `b`, `x` and `y` must not overlap.
    /// The MKL library must be linked and usable. The returned status is MKL's; on a non-success status `*a` must not be used.
    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_symgs_mv(
        op: sparse_operation_t::Type,
        a: sparse_matrix_t,
        descr: matrix_descr,
        alpha: Self,
        b: *const Self,
        x: *mut Self,
        y: *mut Self,
    ) -> sparse_status_t::Type;

    /// Raw `sparse_spmmd` call.
    ///
    /// # Safety
    ///
    /// `a` and `b` must be live handles whose dimensions make `op(A) * B` defined, created by `sparse_create_*` and not yet destroyed.
    /// `c` must be valid for writes of the dense `rows(op(A)) x cols(B)` result stored per `layout`: row-major needs `ldc >= cols(B)` and `rows(op(A)) * ldc` elements; column-major needs `ldc >= rows(op(A))` and `cols(B) * ldc` elements. `c` must not overlap any storage owned by `a` or `b`.
    /// The MKL library must be linked and usable. The returned status is MKL's; on a non-success status `*a` must not be used.
    #[allow(clippy::too_many_arguments)]
    unsafe fn sparse_spmmd(
        op: sparse_operation_t::Type,
        a: sparse_matrix_t,
        b: sparse_matrix_t,
        layout: onemkl_sys::sparse_layout_t::Type,
        c: *mut Self,
        ldc: c_int,
    ) -> sparse_status_t::Type;
}

macro_rules! impl_sparse_real {
    ($ty:ty,
        create_csr=$create:ident, create_coo=$create_coo:ident,
        create_csc=$create_csc:ident, create_bsr=$create_bsr:ident,
        mv=$mv:ident, mm=$mm:ident, trsv=$trsv:ident,
        add=$add:ident, spmmd=$spmmd:ident,
        dotmv=$dotmv:ident, symgs=$symgs:ident, symgs_mv=$symgs_mv:ident,
    ) => {
        impl SparseScalar for $ty {
            unsafe fn sparse_create_csr(
                a: *mut sparse_matrix_t,
                indexing: sparse_index_base_t::Type,
                rows: c_int,
                cols: c_int,
                rows_start: *mut c_int,
                rows_end: *mut c_int,
                col_indx: *mut c_int,
                values: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$create(
                        a, indexing, rows, cols, rows_start, rows_end, col_indx, values,
                    )
                }
            }
            unsafe fn sparse_create_coo(
                a: *mut sparse_matrix_t,
                indexing: sparse_index_base_t::Type,
                rows: c_int,
                cols: c_int,
                nnz: c_int,
                row_indx: *mut c_int,
                col_indx: *mut c_int,
                values: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$create_coo(a, indexing, rows, cols, nnz, row_indx, col_indx, values)
                }
            }
            unsafe fn sparse_create_csc(
                a: *mut sparse_matrix_t,
                indexing: sparse_index_base_t::Type,
                rows: c_int,
                cols: c_int,
                cols_start: *mut c_int,
                cols_end: *mut c_int,
                row_indx: *mut c_int,
                values: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$create_csc(
                        a, indexing, rows, cols, cols_start, cols_end, row_indx, values,
                    )
                }
            }
            unsafe fn sparse_create_bsr(
                a: *mut sparse_matrix_t,
                indexing: sparse_index_base_t::Type,
                block_layout: onemkl_sys::sparse_layout_t::Type,
                rows: c_int,
                cols: c_int,
                block_size: c_int,
                rows_start: *mut c_int,
                rows_end: *mut c_int,
                col_indx: *mut c_int,
                values: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$create_bsr(
                        a,
                        indexing,
                        block_layout,
                        rows,
                        cols,
                        block_size,
                        rows_start,
                        rows_end,
                        col_indx,
                        values,
                    )
                }
            }
            unsafe fn sparse_mv(
                op: sparse_operation_t::Type,
                alpha: Self,
                a: sparse_matrix_t,
                descr: matrix_descr,
                x: *const Self,
                beta: Self,
                y: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe { sys::$mv(op, alpha, a, descr, x, beta, y) }
            }
            unsafe fn sparse_mm(
                op: sparse_operation_t::Type,
                alpha: Self,
                a: sparse_matrix_t,
                descr: matrix_descr,
                layout: sparse_layout_t::Type,
                x: *const Self,
                columns: c_int,
                ldx: c_int,
                beta: Self,
                y: *mut Self,
                ldy: c_int,
            ) -> sparse_status_t::Type {
                unsafe { sys::$mm(op, alpha, a, descr, layout, x, columns, ldx, beta, y, ldy) }
            }
            unsafe fn sparse_trsv(
                op: sparse_operation_t::Type,
                alpha: Self,
                a: sparse_matrix_t,
                descr: matrix_descr,
                x: *const Self,
                y: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe { sys::$trsv(op, alpha, a, descr, x, y) }
            }
            unsafe fn sparse_add(
                op: sparse_operation_t::Type,
                a: sparse_matrix_t,
                alpha: Self,
                b: sparse_matrix_t,
                c: *mut sparse_matrix_t,
            ) -> sparse_status_t::Type {
                unsafe { sys::$add(op, a, alpha, b, c) }
            }
            unsafe fn sparse_spmmd(
                op: sparse_operation_t::Type,
                a: sparse_matrix_t,
                b: sparse_matrix_t,
                layout: onemkl_sys::sparse_layout_t::Type,
                c: *mut Self,
                ldc: c_int,
            ) -> sparse_status_t::Type {
                unsafe { sys::$spmmd(op, a, b, layout, c, ldc) }
            }
            unsafe fn sparse_dotmv(
                op: sparse_operation_t::Type,
                alpha: Self,
                a: sparse_matrix_t,
                descr: matrix_descr,
                x: *const Self,
                beta: Self,
                y: *mut Self,
                d: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe { sys::$dotmv(op, alpha, a, descr, x, beta, y, d) }
            }
            unsafe fn sparse_symgs(
                op: sparse_operation_t::Type,
                a: sparse_matrix_t,
                descr: matrix_descr,
                alpha: Self,
                b: *const Self,
                x: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe { sys::$symgs(op, a, descr, alpha, b, x) }
            }
            unsafe fn sparse_symgs_mv(
                op: sparse_operation_t::Type,
                a: sparse_matrix_t,
                descr: matrix_descr,
                alpha: Self,
                b: *const Self,
                x: *mut Self,
                y: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe { sys::$symgs_mv(op, a, descr, alpha, b, x, y) }
            }
        }
    };
}

impl_sparse_real!(
    f32,
    create_csr = mkl_sparse_s_create_csr,
    create_coo = mkl_sparse_s_create_coo,
    create_csc = mkl_sparse_s_create_csc,
    create_bsr = mkl_sparse_s_create_bsr,
    mv = mkl_sparse_s_mv,
    mm = mkl_sparse_s_mm,
    trsv = mkl_sparse_s_trsv,
    add = mkl_sparse_s_add,
    spmmd = mkl_sparse_s_spmmd,
    dotmv = mkl_sparse_s_dotmv,
    symgs = mkl_sparse_s_symgs,
    symgs_mv = mkl_sparse_s_symgs_mv,
);
impl_sparse_real!(
    f64,
    create_csr = mkl_sparse_d_create_csr,
    create_coo = mkl_sparse_d_create_coo,
    create_csc = mkl_sparse_d_create_csc,
    create_bsr = mkl_sparse_d_create_bsr,
    mv = mkl_sparse_d_mv,
    mm = mkl_sparse_d_mm,
    trsv = mkl_sparse_d_trsv,
    add = mkl_sparse_d_add,
    spmmd = mkl_sparse_d_spmmd,
    dotmv = mkl_sparse_d_dotmv,
    symgs = mkl_sparse_d_symgs,
    symgs_mv = mkl_sparse_d_symgs_mv,
);

macro_rules! impl_sparse_complex {
    ($ty:ty,
        create_csr=$create:ident, create_coo=$create_coo:ident,
        create_csc=$create_csc:ident, create_bsr=$create_bsr:ident,
        mv=$mv:ident, mm=$mm:ident, trsv=$trsv:ident,
        add=$add:ident, spmmd=$spmmd:ident,
        dotmv=$dotmv:ident, symgs=$symgs:ident, symgs_mv=$symgs_mv:ident,
    ) => {
        impl SparseScalar for $ty {
            unsafe fn sparse_create_csr(
                a: *mut sparse_matrix_t,
                indexing: sparse_index_base_t::Type,
                rows: c_int,
                cols: c_int,
                rows_start: *mut c_int,
                rows_end: *mut c_int,
                col_indx: *mut c_int,
                values: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$create(
                        a,
                        indexing,
                        rows,
                        cols,
                        rows_start,
                        rows_end,
                        col_indx,
                        values.cast(),
                    )
                }
            }
            unsafe fn sparse_create_coo(
                a: *mut sparse_matrix_t,
                indexing: sparse_index_base_t::Type,
                rows: c_int,
                cols: c_int,
                nnz: c_int,
                row_indx: *mut c_int,
                col_indx: *mut c_int,
                values: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$create_coo(
                        a,
                        indexing,
                        rows,
                        cols,
                        nnz,
                        row_indx,
                        col_indx,
                        values.cast(),
                    )
                }
            }
            unsafe fn sparse_create_csc(
                a: *mut sparse_matrix_t,
                indexing: sparse_index_base_t::Type,
                rows: c_int,
                cols: c_int,
                cols_start: *mut c_int,
                cols_end: *mut c_int,
                row_indx: *mut c_int,
                values: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$create_csc(
                        a,
                        indexing,
                        rows,
                        cols,
                        cols_start,
                        cols_end,
                        row_indx,
                        values.cast(),
                    )
                }
            }
            unsafe fn sparse_create_bsr(
                a: *mut sparse_matrix_t,
                indexing: sparse_index_base_t::Type,
                block_layout: onemkl_sys::sparse_layout_t::Type,
                rows: c_int,
                cols: c_int,
                block_size: c_int,
                rows_start: *mut c_int,
                rows_end: *mut c_int,
                col_indx: *mut c_int,
                values: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$create_bsr(
                        a,
                        indexing,
                        block_layout,
                        rows,
                        cols,
                        block_size,
                        rows_start,
                        rows_end,
                        col_indx,
                        values.cast(),
                    )
                }
            }
            unsafe fn sparse_mv(
                op: sparse_operation_t::Type,
                alpha: Self,
                a: sparse_matrix_t,
                descr: matrix_descr,
                x: *const Self,
                beta: Self,
                y: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$mv(
                        op,
                        core::mem::transmute_copy(&alpha),
                        a,
                        descr,
                        x.cast(),
                        core::mem::transmute_copy(&beta),
                        y.cast(),
                    )
                }
            }
            unsafe fn sparse_mm(
                op: sparse_operation_t::Type,
                alpha: Self,
                a: sparse_matrix_t,
                descr: matrix_descr,
                layout: sparse_layout_t::Type,
                x: *const Self,
                columns: c_int,
                ldx: c_int,
                beta: Self,
                y: *mut Self,
                ldy: c_int,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$mm(
                        op,
                        core::mem::transmute_copy(&alpha),
                        a,
                        descr,
                        layout,
                        x.cast(),
                        columns,
                        ldx,
                        core::mem::transmute_copy(&beta),
                        y.cast(),
                        ldy,
                    )
                }
            }
            unsafe fn sparse_trsv(
                op: sparse_operation_t::Type,
                alpha: Self,
                a: sparse_matrix_t,
                descr: matrix_descr,
                x: *const Self,
                y: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$trsv(
                        op,
                        core::mem::transmute_copy(&alpha),
                        a,
                        descr,
                        x.cast(),
                        y.cast(),
                    )
                }
            }
            unsafe fn sparse_add(
                op: sparse_operation_t::Type,
                a: sparse_matrix_t,
                alpha: Self,
                b: sparse_matrix_t,
                c: *mut sparse_matrix_t,
            ) -> sparse_status_t::Type {
                unsafe { sys::$add(op, a, core::mem::transmute_copy(&alpha), b, c) }
            }
            unsafe fn sparse_spmmd(
                op: sparse_operation_t::Type,
                a: sparse_matrix_t,
                b: sparse_matrix_t,
                layout: onemkl_sys::sparse_layout_t::Type,
                c: *mut Self,
                ldc: c_int,
            ) -> sparse_status_t::Type {
                unsafe { sys::$spmmd(op, a, b, layout, c.cast(), ldc) }
            }
            unsafe fn sparse_dotmv(
                op: sparse_operation_t::Type,
                alpha: Self,
                a: sparse_matrix_t,
                descr: matrix_descr,
                x: *const Self,
                beta: Self,
                y: *mut Self,
                d: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$dotmv(
                        op,
                        core::mem::transmute_copy(&alpha),
                        a,
                        descr,
                        x.cast(),
                        core::mem::transmute_copy(&beta),
                        y.cast(),
                        d.cast(),
                    )
                }
            }
            unsafe fn sparse_symgs(
                op: sparse_operation_t::Type,
                a: sparse_matrix_t,
                descr: matrix_descr,
                alpha: Self,
                b: *const Self,
                x: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$symgs(
                        op,
                        a,
                        descr,
                        core::mem::transmute_copy(&alpha),
                        b.cast(),
                        x.cast(),
                    )
                }
            }
            unsafe fn sparse_symgs_mv(
                op: sparse_operation_t::Type,
                a: sparse_matrix_t,
                descr: matrix_descr,
                alpha: Self,
                b: *const Self,
                x: *mut Self,
                y: *mut Self,
            ) -> sparse_status_t::Type {
                unsafe {
                    sys::$symgs_mv(
                        op,
                        a,
                        descr,
                        core::mem::transmute_copy(&alpha),
                        b.cast(),
                        x.cast(),
                        y.cast(),
                    )
                }
            }
        }
    };
}

impl_sparse_complex!(
    Complex32,
    create_csr = mkl_sparse_c_create_csr,
    create_coo = mkl_sparse_c_create_coo,
    create_csc = mkl_sparse_c_create_csc,
    create_bsr = mkl_sparse_c_create_bsr,
    mv = mkl_sparse_c_mv,
    mm = mkl_sparse_c_mm,
    trsv = mkl_sparse_c_trsv,
    add = mkl_sparse_c_add,
    spmmd = mkl_sparse_c_spmmd,
    dotmv = mkl_sparse_c_dotmv,
    symgs = mkl_sparse_c_symgs,
    symgs_mv = mkl_sparse_c_symgs_mv,
);
impl_sparse_complex!(
    Complex64,
    create_csr = mkl_sparse_z_create_csr,
    create_coo = mkl_sparse_z_create_coo,
    create_csc = mkl_sparse_z_create_csc,
    create_bsr = mkl_sparse_z_create_bsr,
    mv = mkl_sparse_z_mv,
    mm = mkl_sparse_z_mm,
    trsv = mkl_sparse_z_trsv,
    add = mkl_sparse_z_add,
    spmmd = mkl_sparse_z_spmmd,
    dotmv = mkl_sparse_z_dotmv,
    symgs = mkl_sparse_z_symgs,
    symgs_mv = mkl_sparse_z_symgs_mv,
);
