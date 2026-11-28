// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "data-fitting")]

//! `interpolate_with_search_callback` (the `search_cb` of
//! `df?InterpolateEx1D`). Behaviour measured on oneMKL 2026.2 with
//! `tools/size-probes/df_search2.c` and `df_search3.c`: called once with every
//! site, exact cells out of range give silent zeros, a positive status is
//! swallowed.

use std::cell::RefCell;

use onemkl::Error;
use onemkl::data_fitting::{CubicSpline1d, LinearSpline1d, QuadraticSpline1d, SearchRequest};

const SITES: [f64; 3] = [0.5, 2.5, 4.5];

fn knots() -> Vec<f64> {
    (0..6).map(f64::from).collect()
}

fn spline() -> LinearSpline1d<f64> {
    LinearSpline1d::new(knots(), knots()).unwrap() // y = x
}

fn run(cb: &mut dyn FnMut(SearchRequest<'_, f64>) -> i32) -> Result<Vec<f64>, Error> {
    spline().interpolate_with_search_callback(&SITES, cb)
}

/// The true 1-based cell of each site, shifted by `shift`.
fn fill(r: &mut SearchRequest<'_, f64>, shift: i64) {
    for (i, s) in r.sites.iter().enumerate() {
        r.cells[i] = *s as i64 + 1 + shift;
    }
}

#[test]
fn called_once_with_every_site_and_result_is_correct() {
    let seen = RefCell::new(Vec::new());
    let mut cb = |mut r: SearchRequest<'_, f64>| {
        seen.borrow_mut().push(r.sites.to_vec());
        fill(&mut r, 0);
        r.exact.fill(1);
        0
    };
    assert_eq!(run(&mut cb).unwrap(), SITES);
    assert_eq!(*seen.borrow(), vec![SITES.to_vec()]);
}

#[test]
fn out_of_range_exact_cell_is_an_error_not_silent_zeros() {
    // Unguarded, oneMKL 2026.2 returned [0, 0, 0] with status 0 for cell 99
    // (df_search3.c): the wrapper rejects it before oneMKL uses it.
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
        fill(&mut r, -1);
        r.exact.fill(1);
        0
    };
    assert!(matches!(run(&mut cb), Err(Error::SearchCell { .. })));
}

#[test]
fn approximate_cells_are_refined_by_the_library() {
    let mut cb = |r: SearchRequest<'_, f64>| {
        r.cells.fill(99);
        r.exact.fill(0);
        0
    };
    assert_eq!(run(&mut cb).unwrap(), SITES);
}

#[test]
fn exact_result_status_is_accepted() {
    let mut cb = |mut r: SearchRequest<'_, f64>| {
        fill(&mut r, 0);
        onemkl::data_fitting::DF_STATUS_EXACT_RESULT
    };
    assert_eq!(run(&mut cb).unwrap(), SITES);
}

#[test]
fn negative_status_is_the_error_positive_status_is_a_typed_error() {
    let mut neg = |_: SearchRequest<'_, f64>| -5;
    assert!(matches!(run(&mut neg), Err(Error::VslStatus(-5))));
    // Unguarded, oneMKL swallowed this status (df_search3.c).
    let mut pos = |mut r: SearchRequest<'_, f64>| {
        fill(&mut r, 0);
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

#[test]
fn cubic_and_quadratic_match_their_plain_interpolate() {
    let mut cb = |mut r: SearchRequest<'_, f64>| {
        fill(&mut r, 0);
        r.exact.fill(1);
        0
    };
    let y: Vec<f64> = knots().iter().map(|x| x * x).collect();
    let c = CubicSpline1d::natural(knots(), y.clone()).unwrap();
    assert_eq!(
        c.interpolate_with_search_callback(&SITES, &mut cb).unwrap(),
        c.interpolate(&SITES).unwrap()
    );
    let q = QuadraticSpline1d::with_midpoint_value(knots(), y, 0.25).unwrap();
    assert_eq!(
        q.interpolate_with_search_callback(&SITES, &mut cb).unwrap(),
        q.interpolate(&SITES).unwrap()
    );
}

#[test]
fn f32_works_too() {
    let x: Vec<f32> = (0..6).map(|v| v as f32).collect();
    let s = LinearSpline1d::new(x.clone(), x).unwrap();
    let mut cb = |r: SearchRequest<'_, f32>| {
        for (i, v) in r.sites.iter().enumerate() {
            r.cells[i] = *v as i64 + 1;
        }
        r.exact.fill(1);
        0
    };
    let out = s
        .interpolate_with_search_callback(&[0.5, 2.5], &mut cb)
        .unwrap();
    assert_eq!(out, vec![0.5, 2.5]);
}
