// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "data-fitting")]

//! `integrate_with_callbacks` (`df?IntegrateEx1D`). Behaviour measured on
//! oneMKL 2026.2 with `tools/size-probes/df_probe3.c` (see
//! `docs/DATA-FITTING-FACTS.md`): regions, 1-based cell arguments, status
//! pass-through.

use std::cell::RefCell;

use onemkl::Error;
use onemkl::data_fitting::{IntegrateCallbacks, IntegrationPieces, LinearSpline1d};

fn spline() -> LinearSpline1d<f64> {
    // y = x on [0, 7].
    let x: Vec<f64> = (0..8).map(f64::from).collect();
    LinearSpline1d::new(x.clone(), x).unwrap()
}

#[test]
fn interior_callback_gets_the_interior_piece_with_one_based_cells() {
    let s = spline();
    let seen = RefCell::new(Vec::new());
    let mut interior = |p: IntegrationPieces<'_, f64>| {
        seen.borrow_mut().push((
            p.left_limits[0],
            p.right_limits[0],
            p.left_cells[0],
            p.right_cells[0],
        ));
        p.out[0] = 100.0 + (p.right_limits[0] - p.left_limits[0]);
        0
    };
    let r = s
        .integrate_with_callbacks(
            &[1.0],
            &[3.0],
            IntegrateCallbacks::new().interior(&mut interior),
        )
        .unwrap();
    assert_eq!(r, vec![102.0]);
    // Measured: [1,3] arrives as cells (2, 4), 1-based.
    assert_eq!(*seen.borrow(), vec![(1.0, 3.0, 2, 4)]);
}

#[test]
fn limits_outside_the_interval_are_split_between_the_three_callbacks() {
    let s = spline();
    let order = RefCell::new(Vec::new());
    let mut left = |p: IntegrationPieces<'_, f64>| {
        order
            .borrow_mut()
            .push(("left", p.left_limits[0], p.right_limits[0]));
        p.out[0] = 1000.0;
        0
    };
    let mut right = |p: IntegrationPieces<'_, f64>| {
        order
            .borrow_mut()
            .push(("right", p.left_limits[0], p.right_limits[0]));
        p.out[0] = 10.0;
        0
    };
    let r = s
        .integrate_with_callbacks(
            &[-2.0],
            &[9.0],
            IntegrateCallbacks::new().left(&mut left).right(&mut right),
        )
        .unwrap();
    // Interior piece [0, 7] uses the library: integral of x = 24.5.
    assert!((r[0] - (1000.0 + 24.5 + 10.0)).abs() < 1e-9, "{}", r[0]);
    let order = order.borrow();
    assert!(order.contains(&("left", -2.0, 0.0)), "{order:?}");
    assert!(order.contains(&("right", 7.0, 9.0)), "{order:?}");
}

#[test]
fn a_positive_callback_status_is_a_warning_and_the_result_is_kept() {
    let s = spline();
    let mut interior = |p: IntegrationPieces<'_, f64>| {
        p.out[0] = 5.0;
        5
    };
    let r = s
        .integrate_with_callbacks(
            &[1.0],
            &[3.0],
            IntegrateCallbacks::new().interior(&mut interior),
        )
        .unwrap();
    assert_eq!(r, vec![5.0]);
}

#[test]
fn a_negative_callback_status_is_an_error() {
    let s = spline();
    let mut interior = |_p: IntegrationPieces<'_, f64>| -3;
    let e = s
        .integrate_with_callbacks(
            &[1.0],
            &[3.0],
            IntegrateCallbacks::new().interior(&mut interior),
        )
        .unwrap_err();
    assert!(matches!(e, Error::VslStatus(-3)), "{e:?}");
}

#[test]
#[should_panic(expected = "boom in callback")]
fn a_panic_in_a_callback_is_resumed_after_the_call() {
    let s = spline();
    let mut interior = |_p: IntegrationPieces<'_, f64>| -> i32 { panic!("boom in callback") };
    let _ = s.integrate_with_callbacks(
        &[1.0],
        &[3.0],
        IntegrateCallbacks::new().interior(&mut interior),
    );
}

#[test]
fn no_callbacks_equals_plain_integrate() {
    let s = spline();
    let a = s
        .integrate_with_callbacks(&[1.0, 0.0], &[3.0, 7.0], IntegrateCallbacks::default())
        .unwrap();
    let b = s.integrate(&[1.0, 0.0], &[3.0, 7.0]).unwrap();
    assert_eq!(a, b);
}

#[test]
fn the_cubic_spline_has_the_same_call_back_form() {
    use onemkl::data_fitting::CubicSpline1d;
    let x: Vec<f64> = (0..8).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|v| v * v).collect();
    let s = CubicSpline1d::natural(x, y).unwrap();
    let mut right = |p: IntegrationPieces<'_, f64>| {
        p.out[0] = 42.0;
        0
    };
    let with = s
        .integrate_with_callbacks(
            &[1.0],
            &[9.0],
            IntegrateCallbacks::new().right(&mut right),
        )
        .unwrap();
    let inside = s.integrate(&[1.0], &[7.0]).unwrap();
    assert!(
        (with[0] - (inside[0] + 42.0)).abs() < 1e-9,
        "{with:?} {inside:?}"
    );
}
