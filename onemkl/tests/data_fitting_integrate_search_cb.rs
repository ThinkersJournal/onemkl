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
        IntegrateCallbacks {
            search: Some(cb),
            ..Default::default()
        },
    )
}

fn fill(r: &mut SearchRequest<'_, f64>, base: i64) {
    for (i, s) in r.sites.iter().enumerate() {
        r.cells[i] = (*s as i64).clamp(0, 4) + base;
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
fn out_of_range_exact_cells_give_zeros_not_an_error_vacuity_mutant() {
    // Proves the call-back's cells reach the integrator: cell 99 declared
    // exact changes the result, from the correct integrals to silent zeros.
    let mut cb = |r: SearchRequest<'_, f64>| {
        r.cells.fill(99);
        r.exact.fill(1);
        0
    };
    let out = run(&mut cb).unwrap();
    assert_eq!(out, [0.0, 0.0, 0.0]);
    assert_ne!(out, EXPECT);
}

#[test]
fn negative_status_is_the_error_positive_status_is_swallowed() {
    let mut neg = |_: SearchRequest<'_, f64>| -5;
    assert!(matches!(run(&mut neg), Err(Error::VslStatus(-5))));
    let mut pos = |mut r: SearchRequest<'_, f64>| {
        fill(&mut r, 1);
        r.exact.fill(1);
        3
    };
    assert_eq!(run(&mut pos).unwrap(), EXPECT);
}

#[test]
fn panic_in_the_search_callback_is_resumed() {
    let r = std::panic::catch_unwind(|| {
        let mut cb = |_: SearchRequest<'_, f64>| -> i32 { panic!("boom") };
        let _ = run(&mut cb);
    });
    assert!(r.is_err());
}
