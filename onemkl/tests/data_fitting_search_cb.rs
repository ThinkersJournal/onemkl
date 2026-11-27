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
fn shifted_exact_cells_are_rejected_not_silently_used() {
    // Mutant: 0-based cells declared exact are copied through, so the result
    // is shifted. This proves the call-back's output is what comes back (the
    // tests above and below could not pass if oneMKL ignored it).
    let s = spline();
    let mut cb = |mut r: SearchRequest<'_, f64>| {
        fill(&mut r, 0);
        r.exact.fill(1);
        0
    };
    // Unguarded, oneMKL returned the shifted cells [0, 0, 1, 3] (measured:
    // df_search.c); the wrapper rejects them before oneMKL uses them.
    assert_eq!(
        s.find_cells_with_callback(&SITES, &mut cb),
        Err(Error::SearchCell {
            index: 0,
            cell: 0,
            max_cell: 6
        })
    );
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
fn negative_status_is_an_error_positive_status_is_an_error_too() {
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
        s.find_cells_with_callback(&SITES, &mut pos),
        Err(Error::SearchCallbackStatus(3))
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

// ---- QuadraticSpline1d: same contract, measured on the same build ----

mod quadratic {
    use super::*;
    use onemkl::data_fitting::QuadraticSpline1d;

    fn qspline() -> QuadraticSpline1d<f64> {
        let x: Vec<f64> = (0..6).map(f64::from).collect();
        QuadraticSpline1d::with_midpoint_value(x.clone(), x, 0.5).unwrap()
    }

    #[test]
    fn exact_one_based_callback_is_used_and_called_once() {
        let s = qspline();
        let calls = Cell::new(0);
        let mut cb = |mut r: SearchRequest<'_, f64>| {
            calls.set(calls.get() + 1);
            fill(&mut r, 1);
            r.exact.fill(1);
            0
        };
        assert_eq!(
            s.find_cells_with_callback(&SITES, &mut cb).unwrap(),
            [0, 1, 2, 4]
        );
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn shifted_exact_cells_are_rejected() {
        // Proves the call-back's output is what comes back.
        let s = qspline();
        let mut cb = |mut r: SearchRequest<'_, f64>| {
            fill(&mut r, 0);
            r.exact.fill(1);
            0
        };
        assert_eq!(
            s.find_cells_with_callback(&SITES, &mut cb),
            Err(Error::SearchCell {
                index: 0,
                cell: 0,
                max_cell: 6
            })
        );
    }

    #[test]
    fn approximate_guesses_are_refined() {
        let s = qspline();
        let mut cb = |mut r: SearchRequest<'_, f64>| {
            fill(&mut r, 0);
            r.exact.fill(0);
            0
        };
        assert_eq!(
            s.find_cells_with_callback(&SITES, &mut cb).unwrap(),
            [0, 1, 2, 4]
        );
    }

    #[test]
    fn negative_and_positive_statuses_are_errors() {
        let s = qspline();
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
            s.find_cells_with_callback(&SITES, &mut pos),
            Err(Error::SearchCallbackStatus(3))
        );
    }
}

#[test]
fn out_of_range_exact_cell_is_a_typed_error() {
    let s = spline();
    let mut cb = |r: SearchRequest<'_, f64>| {
        r.cells.fill(99);
        r.exact.fill(1);
        0
    };
    assert_eq!(
        s.find_cells_with_callback(&SITES, &mut cb),
        Err(Error::SearchCell {
            index: 0,
            cell: 99,
            max_cell: 6
        })
    );
}
