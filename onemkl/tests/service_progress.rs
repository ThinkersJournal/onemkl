// SPDX-License-Identifier: MIT OR Apache-2.0
//! `set_progress_handler`: process-global, so one test in its own process.
//! Records whether MKL calls the handler during a LAPACK factorization.

use std::sync::atomic::{AtomicUsize, Ordering};

use onemkl::service::{self, ProgressControl};

static CALLS: AtomicUsize = AtomicUsize::new(0);

fn count(_thread: i32, _step: i32, _stage: &str) -> ProgressControl {
    CALLS.fetch_add(1, Ordering::SeqCst);
    ProgressControl::Continue
}

fn stop(_t: i32, _s: i32, _st: &str) -> ProgressControl {
    CALLS.fetch_add(1, Ordering::SeqCst);
    ProgressControl::Stop
}

fn factor(n: usize) -> i32 {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        a[i * n + i] = n as f64 + 1.0;
    }
    let (uplo, nn, lda) = (
        b'U' as core::ffi::c_char,
        n as core::ffi::c_int,
        n as core::ffi::c_int,
    );
    let mut info: core::ffi::c_int = 0;
    unsafe { onemkl_sys::dpotrf(&uplo, &nn, a.as_mut_ptr(), &lda, &mut info) };
    info
}

#[test]
fn handler_install_remove_and_observed_calls() {
    service::set_progress_handler(Some(count));
    assert_eq!(factor(512), 0);
    let with_handler = CALLS.load(Ordering::SeqCst);
    println!("progress calls with handler = {with_handler}");

    service::set_progress_handler(None);
    assert_eq!(factor(512), 0);
    assert_eq!(
        CALLS.load(Ordering::SeqCst),
        with_handler,
        "handler still called after removal"
    );

    // `Stop`: MKL abandons the factorization after the first callback
    // and the routine reports a nonzero info (measured: -1002).
    service::set_progress_handler(Some(stop));
    let info = factor(512);
    assert_eq!(CALLS.load(Ordering::SeqCst), with_handler + 1);
    assert_eq!(info, -1002);
    service::set_progress_handler(None);
}
