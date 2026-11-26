// SPDX-License-Identifier: MIT OR Apache-2.0
//! CBWR must be set before MKL's first dispatch, so this file holds a
//! single test that runs in its own process. It needs `blas` only to
//! force a dispatch.
#![cfg(feature = "blas")]

use onemkl::service::{self, CbwrBranch, CbwrError};

#[test]
fn cbwr_default_set_query_and_late_set_fails() {
    // Default state (nothing set, nothing dispatched): CBWR is off.
    assert_eq!(service::cbwr_branch().unwrap(), CbwrBranch::Off);
    assert!(!service::cbwr_strict());

    // Queryable. Measured: oneMKL 2026.2.1 on Linux x86_64 reports
    // `Auto` itself here, not a concrete branch, so only `Ok` is asserted.
    service::cbwr_auto_branch().unwrap();

    // `Compatible` is supported on every x86-64 CPU.
    service::cbwr_set_strict(CbwrBranch::Compatible).unwrap();
    assert_eq!(service::cbwr_branch().unwrap(), CbwrBranch::Compatible);
    assert!(service::cbwr_strict());

    // Dispatch something, then the mode is locked.
    let x = [1.0f64, 2.0, 3.0];
    let y = [4.0f64, 5.0, 6.0];
    assert_eq!(onemkl::blas::level1::dot(&x, &y).unwrap(), 32.0);

    // Re-set a *supported* branch (so the failure is the lock, not an
    // unsupported CPU feature) with a different strictness.
    assert_eq!(
        service::cbwr_set(CbwrBranch::Compatible),
        Err(CbwrError::ModeLocked)
    );
    assert_eq!(service::cbwr_branch().unwrap(), CbwrBranch::Compatible);
    assert!(service::cbwr_strict());
}
