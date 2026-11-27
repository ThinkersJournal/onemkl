// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "data-fitting")]

//! `find_cells_with_callback` (`df?SearchCellsEx1D`). Behaviour measured on
//! oneMKL 2026.2 with `tools/size-probes/df_search.c`: 1-based cells, `exact`
//! flag, `DF_STATUS_EXACT_RESULT`, call-back status becomes the routine status.

use std::cell::Cell;

use onemkl::Error;
use onemkl::data_fitting::{DF_STATUS_EXACT_RESULT, LinearSpline1d, SearchRequest};

fn spline() -> LinearSpline1d<f64> {
    let x: Vec<f64> = (0..6).map(f64::from).collect();
    LinearSpline1d::new(x.clone(), x).unwrap()
}

const SITES: [f64; 4] = [0.5, 1.5, 2.5, 4.5];

/// Fills `cells` with `floor(site) + base`, capped to the last interval.
fn fill(req: &mut SearchRequest<'_, f64>, base: i64) {
    for (i, s) in req.sites.iter().enumerate() {
        req.cells[i] = (*s as i64).clamp(0, 4) + base;
    }
}

#[test]
fn exact_one_based_callback_is_used_and_called_once() {
    let s = spline();
    let calls = Cell::new(0);
    let mut cb = |mut r: SearchRequest<'_, f64>| {
        calls.set(calls.get() + 1);
        assert_eq!(r.sites.len(), 4);
        fill(&mut r, 1);
        r.exact.fill(1);
        0
    };
    let cells = s.find_cells_with_callback(&SITES, &mut cb).unwrap();
    assert_eq!(cells, [0, 1, 2, 4]); // 0-based result of 1-based cells 1, 2, 3, 5
    assert_eq!(calls.get(), 1);
    assert_eq!(cells, vec![0, 1, 2, 4]);
}

#[test]
fn exact_flag_makes_the_library_trust_the_callback_vacuity_mutant() {
    // Mutant: 0-based cells declared exact are copied through, so the result
    // is shifted. This proves the call-back's output is what comes back (the
    // tests above and below could not pass if oneMKL ignored it).
    let s = spline();
    let mut cb = |mut r: SearchRequest<'_, f64>| {
        fill(&mut r, 0);
        r.exact.fill(1);
        0
    };
    let cells = s.find_cells_with_callback(&SITES, &mut cb).unwrap();
    assert_eq!(cells, [0, 0, 1, 3]); // raw 0, 1, 2, 4 -> minus the 1-based offset
    assert_ne!(cells, vec![0, 1, 2, 4]); // the true cells
}

#[test]
fn approximate_guesses_are_refined_by_the_library() {
    let s = spline();
    let mut cb = |mut r: SearchRequest<'_, f64>| {
        fill(&mut r, 0); // wrong by one
        r.exact.fill(0);
        0
    };
    assert_eq!(
        s.find_cells_with_callback(&SITES, &mut cb).unwrap(),
        vec![0, 1, 2, 4]
    );
}

#[test]
fn exact_result_status_needs_no_flags() {
    let s = spline();
    let mut cb = |mut r: SearchRequest<'_, f64>| {
        fill(&mut r, 1);
        DF_STATUS_EXACT_RESULT
    };
    assert_eq!(
        s.find_cells_with_callback(&SITES, &mut cb).unwrap(),
        [0, 1, 2, 4]
    );
}

#[test]
fn negative_status_is_an_error_positive_is_a_warning() {
    let s = spline();
    let mut neg = |_: SearchRequest<'_, f64>| -5;
    assert!(matches!(
        s.find_cells_with_callback(&SITES, &mut neg),
        Err(Error::VslStatus(-5))
    ));
    let mut pos = |mut r: SearchRequest<'_, f64>| {
        fill(&mut r, 1);
        r.exact.fill(1);
        3
    };
    assert_eq!(
        s.find_cells_with_callback(&SITES, &mut pos).unwrap(),
        [0, 1, 2, 4]
    );
}

#[test]
fn callback_panic_is_resumed_after_the_call() {
    let s = spline();
    let r = std::panic::catch_unwind(|| {
        let mut cb = |_: SearchRequest<'_, f64>| -> i32 { panic!("boom") };
        let _ = s.find_cells_with_callback(&SITES, &mut cb);
    });
    assert!(r.is_err());
}
