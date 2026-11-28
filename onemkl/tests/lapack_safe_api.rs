// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "lapack")]

//! Safe-API memory-safety regressions for the LAPACK wrappers: every
//! input below used to reach LAPACKE with a buffer shorter than the
//! routine reads or writes (or with an out-of-range pivot index). Each
//! call must now return `Err` *before* the FFI call.

use num_complex::Complex64;
use onemkl::lapack::{
    Job, gbsv, gbtrf, gbtrs, geev_real, gels, gesdd, getri, getrs, gtsv, gttrf, gttrs, hptrs,
    larfg, laswp, pbsv, pbtrf, pbtrs, ppsv, pptrs, ptsv, pttrs_real, spsv, sptrs,
};
use onemkl::{Layout, MatrixMut, MatrixRef, Transpose, UpLo};

const COL: Layout = Layout::ColMajor;

#[test]
fn banded_buffers_are_checked() {
    let mut ipiv = [0i32; 4];
    assert!(gbsv::<f64>(COL, 4, 1, 1, 1, &mut [], 4, &mut ipiv, &mut [], 4).is_err());
    assert!(gbtrf::<f64>(COL, 4, 4, 1, 1, &mut [], 4, &mut ipiv).is_err());
    assert!(
        gbtrs::<f64>(
            COL,
            Transpose::NoTrans,
            4,
            1,
            1,
            1,
            &[],
            4,
            &[1, 2, 3, 4],
            &mut [],
            4
        )
        .is_err()
    );
    assert!(pbsv::<f64>(COL, UpLo::Upper, 4, 1, 1, &mut [], 2, &mut [], 4).is_err());
    assert!(pbtrf::<f64>(COL, UpLo::Upper, 4, 1, &mut [], 2).is_err());
    assert!(pbtrs::<f64>(COL, UpLo::Upper, 4, 1, 1, &[], 2, &mut [], 4).is_err());
}

#[test]
fn tridiagonal_buffers_are_checked() {
    assert!(
        gtsv::<f64>(
            COL,
            3,
            1,
            &mut [0.0; 2],
            &mut [1.0; 3],
            &mut [0.0; 2],
            &mut [],
            3
        )
        .is_err()
    );
    assert!(
        gtsv::<f64>(
            COL,
            3,
            1000,
            &mut [0.0; 2],
            &mut [1.0; 3],
            &mut [0.0; 2],
            &mut [0.0; 3],
            3
        )
        .is_err()
    );
    assert!(gttrf::<f64>(4, &mut [], &mut [], &mut [], &mut [0.0; 2], &mut [0; 4]).is_err());
    assert!(
        gttrs::<f64>(
            COL,
            Transpose::NoTrans,
            4,
            1,
            &[],
            &[],
            &[],
            &[],
            &[],
            &mut [],
            4
        )
        .is_err()
    );
    assert!(ptsv::<f64>(COL, 3, 1, &mut [1.0; 3], &mut [0.0; 2], &mut [], 3).is_err());
    assert!(pttrs_real::<f64>(COL, 3, 1, &[], &[], &mut [], 3).is_err());
}

#[test]
fn packed_buffers_are_checked() {
    assert!(
        spsv::<f64>(
            COL,
            UpLo::Upper,
            3,
            1,
            &mut [0.0; 6],
            &mut [1, 2, 3],
            &mut [],
            3
        )
        .is_err()
    );
    assert!(sptrs::<f64>(COL, UpLo::Upper, 3, 1, &[], &[], &mut [], 3).is_err());
    assert!(ppsv::<f64>(COL, UpLo::Upper, 3, 1, &mut [0.0; 6], &mut [], 3).is_err());
    assert!(pptrs::<f64>(COL, UpLo::Upper, 3, 1, &[], &mut [], 3).is_err());
    assert!(hptrs::<Complex64>(COL, UpLo::Upper, 3, 1, &[], &[], &mut [], 3).is_err());
}

#[test]
fn pivot_contents_are_checked() {
    let mut a = [1.0f64];
    let mut b = [1.0f64];
    {
        let a = MatrixRef::new(&a, 1, 1, COL).unwrap();
        let mut b = MatrixMut::new(&mut b, 1, 1, COL).unwrap();
        assert!(getrs::<f64>(Transpose::NoTrans, &a, &[2], &mut b).is_err());
    }
    {
        let mut am = MatrixMut::new(&mut a, 1, 1, COL).unwrap();
        assert!(getri::<f64>(&mut am, &[5]).is_err());
        assert!(laswp::<f64>(&mut am, 1, 2, &[1, 1], 1).is_err());
        assert!(laswp::<f64>(&mut am, 1, 1, &[1000000], 1).is_err());
    }
}

#[test]
fn matrix_view_shapes_are_checked() {
    // gels: B must have max(m, n) rows.
    let mut a = [0.0f64; 10];
    let mut b = [0.0f64; 1];
    let mut am = MatrixMut::new(&mut a, 5, 2, COL).unwrap();
    let mut bm = MatrixMut::with_lda(&mut b, 1, 1, COL, 5).unwrap();
    assert!(gels::<f64>(Transpose::NoTrans, &mut am, &mut bm).is_err());

    // geev: Compute without a vector view.
    let mut a4 = [0.0f64; 16];
    let (mut wr, mut wi) = ([0.0f64; 4], [0.0f64; 4]);
    let mut am4 = MatrixMut::new(&mut a4, 4, 4, COL).unwrap();
    assert!(
        geev_real::<f64>(
            Job::Compute,
            Job::None,
            &mut am4,
            &mut wr,
            &mut wi,
            None,
            None
        )
        .is_err()
    );

    // gesdd: Job::All without U / Vt.
    let mut a44 = [0.0f64; 16];
    let mut s = [0.0f64; 4];
    let mut am44 = MatrixMut::new(&mut a44, 4, 4, COL).unwrap();
    assert!(gesdd::<f64>(Job::All, &mut am44, &mut s, None, None).is_err());
}

#[test]
fn larfg_stride_is_checked() {
    let mut alpha = 1.0f64;
    assert!(larfg::<f64>(3, &mut alpha, &mut [1.0, 2.0], 5).is_err());
}
