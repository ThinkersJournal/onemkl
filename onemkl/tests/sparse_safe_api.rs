// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(all(
    feature = "sparse",
    feature = "dss",
    feature = "pardiso",
    feature = "feast",
    feature = "iss"
))]

//! Safe-API memory-safety regressions for the sparse / DSS / PARDISO /
//! ILU / FEAST / ISS wrappers: inputs below used to reach oneMKL with index
//! arrays or dense buffers that do not match the dimensions they were
//! passed with. Each call must now return `Err` *before* the FFI call.

use onemkl::UpLo;
use onemkl::dss::{Definite, Dss, IndexBase as DssBase, Symmetry};
use onemkl::feast::eigh_real_csr;
use onemkl::iss::{FgmresSession, IssOptions};
use onemkl::pardiso::{IndexBase as PBase, Pardiso, PardisoMatrixType};
use onemkl::preconditioners::{apply_ilu, ilu0, ilut};
use onemkl::sparse::{DenseLayout, IndexBase, MatrixDescr, Operation, SparseMatrix};

#[test]
fn csr_content_is_validated() {
    // row_ptr[1] = 5 but only one stored entry.
    assert!(
        SparseMatrix::<f64>::from_csr(1, 1, IndexBase::Zero, vec![0, 5], vec![0], vec![1.0])
            .is_err()
    );
    // column index out of range.
    assert!(
        SparseMatrix::<f64>::from_csr(1, 1, IndexBase::Zero, vec![0, 1], vec![7], vec![1.0])
            .is_err()
    );
    // 1-based indexing with 0-based arrays.
    assert!(
        SparseMatrix::<f64>::from_csr(1, 1, IndexBase::One, vec![0, 1], vec![0], vec![1.0])
            .is_err()
    );
    // non-monotonic row_ptr.
    assert!(
        SparseMatrix::<f64>::from_csr(
            2,
            2,
            IndexBase::Zero,
            vec![0, 2, 1],
            vec![0, 1],
            vec![1.0, 1.0]
        )
        .is_err()
    );
    // COO index out of range.
    assert!(
        SparseMatrix::<f64>::from_coo(1, 1, IndexBase::Zero, vec![9], vec![0], vec![1.0]).is_err()
    );
}

#[test]
fn dense_operands_of_mm_are_checked() {
    let a = SparseMatrix::<f64>::from_csr(
        3,
        3,
        IndexBase::Zero,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0; 3],
    )
    .unwrap();
    let x = [0.0f64; 1];
    let mut y = [0.0f64; 1];
    assert!(
        a.mm(
            Operation::NoTrans,
            1.0,
            MatrixDescr::general(),
            DenseLayout::ColMajor,
            &x,
            3,
            3,
            0.0,
            &mut y,
            3
        )
        .is_err()
    );
}

#[test]
fn dss_buffers_are_checked() {
    let mut d = Dss::<f64>::new().unwrap();
    assert!(
        d.define_structure(Symmetry::NonSymmetric, 1, 1, &[1, 100], &[1], DssBase::One)
            .is_err()
    );
    d.define_structure(
        Symmetry::NonSymmetric,
        3,
        3,
        &[1, 2, 3, 4],
        &[1, 2, 3],
        DssBase::One,
    )
    .unwrap();
    d.reorder().unwrap();
    assert!(d.factor_real(Definite::Indefinite, &[1.0]).is_err());
    d.factor_real(Definite::Indefinite, &[1.0, 1.0, 1.0])
        .unwrap();
    assert!(d.solve_real(&[0.0; 3], 1000, &mut [0.0; 3]).is_err());
    assert!(d.solve_real(&[0.0; 3], -1, &mut [0.0; 3]).is_err());
}

#[test]
fn pardiso_buffers_are_checked() {
    let mut p = Pardiso::<f64>::new(PardisoMatrixType::RealStructSym).with_indexing(PBase::One);
    // ia says 5 stored entries; a / ja hold one.
    assert!(
        p.solve(3, &[1.0], &[1, 2, 3, 6], &[1], &[0.0; 3], &mut [0.0; 3])
            .is_err()
    );
    // x too short for n * nrhs.
    let ia = [1, 2, 3, 4];
    let ja = [1, 2, 3];
    assert!(
        p.solve(3, &[1.0; 3], &ia, &ja, &[0.0; 3], &mut [0.0; 1])
            .is_err()
    );
    // get_diagonal with an n other than the factorized order.
    let mut q = Pardiso::<f64>::new(PardisoMatrixType::RealStructSym).with_diagonal_enabled();
    q.analyze_and_factorize(3, &[1.0; 3], &ia, &ja).unwrap();
    assert!(q.get_diagonal(1).is_err());
}

#[test]
fn ilu_and_feast_csr_content_is_validated() {
    assert!(ilu0(2, &[1.0], &[1, 5, 9], &[1]).is_err());
    assert!(ilut(2, &[1.0], &[1, 5, 9], &[1], 0.0, 1).is_err());
    assert!(apply_ilu(2, &[2.0], &[1, 2, 9], &[1], &[1.0, 1.0]).is_err());
    assert!(eigh_real_csr::<f64>(UpLo::Upper, 1, &[1.0], &[1, 50], &[1], 0.0, 1.0, 1).is_err());
    assert!(eigh_real_csr::<f64>(UpLo::Upper, 0, &[], &[1], &[], 0.0, 1.0, 1).is_err());
}

#[test]
fn fgmres_step_buffers_are_checked() {
    let b3 = [1.0f64; 3];
    let x3 = [0.0f64; 3];
    let mut s = FgmresSession::new(3, &b3, &x3, IssOptions::default()).unwrap();
    let mut b1 = [1.0f64];
    let mut x1 = [0.0f64];
    assert!(s.step(&mut b1, &mut x1).is_err());
}
