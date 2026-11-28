// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "blas")]

//! Safe-API memory-safety regressions for the BLAS wrappers: every call
//! below used to reach MKL with a buffer shorter than the kernel reads or
//! writes. Each must now return `Err` *before* the FFI call.

use onemkl::blas::compact::{CompactFormat, gemm as compact_gemm};
use onemkl::blas::extensions::{
    axpy_batch_strided, dgmm_batch_strided, gemm_batch_strided, gemm3m_batch_strided,
    gemv_batch_strided, omatcopy2, trsm_batch_strided,
};
use onemkl::blas::level2::{gbmv, sbmv, tbmv, tbsv};
use onemkl::blas::packed::{PackedMatrix, gemm_compute_packed_a, gemm_compute_packed_b};
use onemkl::{Diag, Layout, MatrixRef, Side, Transpose, UpLo};

const COL: Layout = Layout::ColMajor;

#[test]
fn banded_views_must_cover_the_band() {
    let empty = MatrixRef::<f64>::new(&[], 0, 0, COL).unwrap();
    let one = [1.0f64];
    let a1 = MatrixRef::new(&one, 1, 1, COL).unwrap();
    let x = vec![0.0f64; 1000];
    let mut y = vec![0.0f64; 1000];
    assert!(
        gbmv::<f64>(
            Transpose::NoTrans,
            &empty,
            1000,
            1000,
            0,
            0,
            1.0,
            &x,
            1,
            0.0,
            &mut y,
            1
        )
        .is_err()
    );
    assert!(
        gbmv::<f64>(
            Transpose::NoTrans,
            &a1,
            1000,
            1000,
            0,
            0,
            1.0,
            &x,
            1,
            0.0,
            &mut y,
            1
        )
        .is_err()
    );
    let mut xm = vec![1.0f64; 100];
    assert!(
        tbmv::<f64>(
            UpLo::Upper,
            Transpose::NoTrans,
            Diag::NonUnit,
            100,
            0,
            &a1,
            &mut xm,
            1
        )
        .is_err()
    );
    assert!(
        tbsv::<f64>(
            UpLo::Upper,
            Transpose::NoTrans,
            Diag::NonUnit,
            100,
            0,
            &a1,
            &mut xm,
            1
        )
        .is_err()
    );
    assert!(sbmv::<f64>(UpLo::Upper, 1000, 0, 1.0, &a1, &x, 1, 0.0, &mut y, 1).is_err());
}

#[test]
fn omatcopy2_buffers_are_checked() {
    let a = [1.0f64];
    let mut b = [0.0f64];
    assert!(
        omatcopy2::<f64>(
            COL,
            Transpose::NoTrans,
            100,
            100,
            1.0,
            &a,
            100,
            1,
            &mut b,
            100,
            1
        )
        .is_err()
    );
}

#[test]
fn strided_batches_are_checked() {
    let mut c = [];
    assert!(
        gemm_batch_strided::<f64>(
            COL,
            Transpose::NoTrans,
            Transpose::NoTrans,
            8,
            8,
            8,
            1.0,
            &[],
            8,
            64,
            &[],
            8,
            64,
            0.0,
            &mut c,
            8,
            64,
            4
        )
        .is_err()
    );
    assert!(
        gemm3m_batch_strided::<num_complex::Complex64>(
            COL,
            Transpose::NoTrans,
            Transpose::NoTrans,
            8,
            8,
            8,
            Default::default(),
            &[],
            8,
            64,
            &[],
            8,
            64,
            Default::default(),
            &mut [],
            8,
            64,
            4
        )
        .is_err()
    );
    assert!(
        gemv_batch_strided::<f64>(
            COL,
            Transpose::NoTrans,
            8,
            8,
            1.0,
            &[],
            8,
            64,
            &[],
            1,
            8,
            0.0,
            &mut [],
            1,
            8,
            4
        )
        .is_err()
    );
    assert!(
        trsm_batch_strided::<f64>(
            COL,
            Side::Left,
            UpLo::Upper,
            Transpose::NoTrans,
            Diag::NonUnit,
            8,
            8,
            1.0,
            &[],
            8,
            64,
            &mut [],
            8,
            64,
            4
        )
        .is_err()
    );
    assert!(
        dgmm_batch_strided::<f64>(
            COL,
            Side::Left,
            8,
            8,
            &[],
            8,
            64,
            &[],
            1,
            8,
            &mut [],
            8,
            64,
            4
        )
        .is_err()
    );
    // Negative batch stride reaches before the start of the slice.
    let x = [0.0f64; 20];
    let mut y = [0.0f64; 20];
    assert!(axpy_batch_strided::<f64>(10, 1.0, &x, 1, -10, &mut y, 1, 10, 2).is_err());
}

#[test]
fn packed_compute_buffers_are_checked() {
    let src = vec![1.0f64; 64 * 64];
    let pa =
        PackedMatrix::<f64>::pack_a(COL, Transpose::NoTrans, 64, 64, 64, 1.0, &src, 64).unwrap();
    assert!(
        gemm_compute_packed_a::<f64>(
            COL,
            Transpose::NoTrans,
            64,
            64,
            64,
            &pa,
            &[],
            64,
            0.0,
            &mut [],
            64
        )
        .is_err()
    );
    let pb =
        PackedMatrix::<f64>::pack_b(COL, Transpose::NoTrans, 64, 64, 64, 1.0, &src, 64).unwrap();
    assert!(
        gemm_compute_packed_b::<f64>(
            COL,
            Transpose::NoTrans,
            64,
            64,
            64,
            &[],
            64,
            &pb,
            0.0,
            &mut [],
            64
        )
        .is_err()
    );
    // ld smaller than the contiguous dimension.
    assert!(
        PackedMatrix::<f64>::pack_a(
            Layout::RowMajor,
            Transpose::NoTrans,
            1,
            1,
            100,
            1.0,
            &[1.0],
            1
        )
        .is_err()
    );
}

#[test]
fn compact_gemm_buffers_are_checked() {
    assert!(
        compact_gemm::<f64>(
            COL,
            Transpose::NoTrans,
            Transpose::NoTrans,
            4,
            4,
            4,
            1.0,
            &[],
            4,
            &[],
            4,
            0.0,
            &mut [],
            4,
            CompactFormat::Avx,
            8
        )
        .is_err()
    );
}
