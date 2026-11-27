// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "data-fitting")]

//! `IntegrateCallbacks::search` (the `search_cb` of `df?IntegrateEx1D`).
//! Behaviour measured on oneMKL 2026.2 with `tools/size-probes/df_search2.c`
//! and `df_search3.c`: called twice (left limits, then right limits), exact
//! cells out of range give silent zeros, a positive status is swallowed.

use std::cell::RefCell;

use onemkl::Error;
use onemkl::data_fitting::{IntegrateCallbacks, LinearSpline1d, SearchRequest};

const LEFT: [f64; 3] = [0.5, 1.5, -1.0];
const RIGHT: [f64; 3] = [2.5, 4.5, 7.0];
const EXPECT: [f64; 3] = [3.0, 9.0, 24.0]; // y = x on [0, 5], linear extrapolation outside

fn spline() -> LinearSpline1d<f64> {
    let x: Vec<f64> = (0..6).map(f64::from).collect();
    LinearSpline1d::new(x.clone(), x).unwrap()
}

fn run(
    cb: &mut dyn FnMut(SearchRequest<'_, f64>) -> i32,
) -> Result<Vec<f64>, Error> {
    spline().integrate_with_callbacks(
        &LEFT,
        &RIGHT,
        IntegrateCallbacks::new().search(cb),
    )
}

/// The true 1-based cell of each site: 0 left of the knots, 6 right of them.
fn fill(r: &mut SearchRequest<'_, f64>, base: i64) {
    for (i, s) in r.sites.iter().enumerate() {
        let truth = if *s < 0.0 {
            0
        } else if *s >= 5.0 {
            6
        } else {
            *s as i64 + 1
        };
        r.cells[i] = truth - 1 + base;
    }
}

#[test]
fn called_twice_left_limits_then_right_limits_and_result_is_correct() {
    let seen = RefCell::new(Vec::new());
    let mut cb = |mut r: SearchRequest<'_, f64>| {
        seen.borrow_mut().push(r.sites.to_vec());
        fill(&mut r, 1);
        r.exact.fill(1);
        0
    };
    let out = run(&mut cb).unwrap();
    assert_eq!(out, EXPECT);
    let seen = seen.borrow();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].len(), 3);
    assert_eq!(seen[1].len(), 3);
    // Measured in df_search2.c: left limits (site 0.5 first), then right (2.5 first).
    assert_eq!(seen[0][0], 0.5);
    assert_eq!(seen[1][0], 2.5);
}

#[test]
fn out_of_range_exact_cell_is_an_error_not_silent_zeros() {
    // Unguarded, oneMKL 2026.2 returned integrals [0, 0, 0] with status 0 for
    // this call-back (df_search3.c, mode 2): the control that fails on the
    // unguarded path. The wrapper rejects it before oneMKL uses the cells.
    let mut cb = |r: SearchRequest<'_, f64>| {
        r.cells.fill(99);
        r.exact.fill(1);
        0
    };
    assert_eq!(
        run(&mut cb),
        Err(Error::SearchCell {
            index: 0,
            cell: 99,
            max_cell: 6
        })
    );
}

#[test]
fn shifted_exact_cells_are_rejected() {
    let mut cb = |mut r: SearchRequest<'_, f64>| {
        fill(&mut r, 0); // each cell one too small
        r.exact.fill(1);
        0
    };
    assert!(matches!(run(&mut cb), Err(Error::SearchCell { .. })));
}

#[test]
fn approximate_cells_are_not_checked_and_the_library_refines_them() {
    let mut cb = |r: SearchRequest<'_, f64>| {
        r.cells.fill(99);
        r.exact.fill(0);
        0
    };
    assert_eq!(run(&mut cb).unwrap(), EXPECT);
}

#[test]
fn negative_status_is_the_error_positive_status_is_a_typed_error() {
    let mut neg = |_: SearchRequest<'_, f64>| -5;
    assert!(matches!(run(&mut neg), Err(Error::VslStatus(-5))));
    // Unguarded, oneMKL swallowed this status and returned correct results
    // with status 0 (df_search3.c).
    let mut pos = |mut r: SearchRequest<'_, f64>| {
        fill(&mut r, 1);
        r.exact.fill(1);
        3
    };
    assert_eq!(run(&mut pos), Err(Error::SearchCallbackStatus(3)));
}

#[test]
fn panic_in_the_search_callback_is_resumed() {
    let r = std::panic::catch_unwind(|| {
        let mut cb = |_: SearchRequest<'_, f64>| -> i32 { panic!("boom") };
        let _ = run(&mut cb);
    });
    assert!(r.is_err());
}
