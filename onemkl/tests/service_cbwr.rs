//! CBWR must be set before MKL's first dispatch, so this file holds a
//! single test that runs in its own process.

use onemkl::service::{self, CbwrBranch};

#[test]
fn cbwr_set_then_query_then_late_set_fails() {
    // Queryable. Measured: oneMKL 2026.2.1 on Linux x86_64 reports
    // `Auto` itself here, not a concrete branch, so only `Ok` is asserted.
    service::cbwr_auto_branch().unwrap();

    service::cbwr_set(CbwrBranch::Compatible, true).unwrap();
    assert_eq!(service::cbwr_branch().unwrap(), CbwrBranch::Compatible);
    assert!(service::cbwr_strict());

    // Dispatch something, then the mode is locked.
    let _ = service::version();
    let x = [1.0f64, 2.0, 3.0];
    let y = [4.0f64, 5.0, 6.0];
    let d = onemkl::blas::level1::dot(&x, &y).unwrap();
    assert_eq!(d, 32.0);
    assert!(service::cbwr_set(CbwrBranch::Avx2, false).is_err());
    assert_eq!(service::cbwr_branch().unwrap(), CbwrBranch::Compatible);
}
