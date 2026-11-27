// SPDX-License-Identifier: MIT OR Apache-2.0
//! `set_xerbla_handler`: process-global, so one test in its own process.
//! Triggers `xerbla` through the Fortran-style `dpotrf` with `n = -1`.

use core::ffi::c_char;
use std::sync::Mutex;

use onemkl::service;

static SEEN: Mutex<Vec<(String, i32)>> = Mutex::new(Vec::new());

fn record(routine: &str, argument: i32) {
    SEEN.lock().unwrap().push((routine.to_owned(), argument));
}

fn bad_dpotrf() -> onemkl_sys::MKL_INT {
    let uplo = b'U' as c_char;
    let n: onemkl_sys::MKL_INT = -1;
    let lda: onemkl_sys::MKL_INT = 1;
    let mut a = [0.0f64; 1];
    let mut info: onemkl_sys::MKL_INT = 0;
    unsafe { onemkl_sys::dpotrf(&uplo, &n, a.as_mut_ptr(), &lda, &mut info) };
    info
}

#[test]
fn handler_sees_routine_and_info_then_can_be_removed() {
    service::set_xerbla_handler(Some(record));
    let info = bad_dpotrf();
    // The routine reports -2 in `info`; `xerbla` is told argument 2.
    assert_eq!(info, -2);
    assert_eq!(SEEN.lock().unwrap().as_slice(), &[("DPOTRF".to_owned(), 2)]);

    // Removed: no further calls reach the handler, and MKL still works.
    service::set_xerbla_handler(None);
    assert_eq!(bad_dpotrf(), -2);
    assert_eq!(SEEN.lock().unwrap().len(), 1);
}
