// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "data-fitting")]

//! `QuadraticSpline1d`: the default quadratic (`DF_PP_QUADRATIC`,
//! `DF_PP_DEFAULT`, `DF_BC_Q_VAL`) and the Subbotin quadratic
//! (`DF_PP_SUBBOTIN`, `DF_BC_FREE_END`, `DF_IC_Q_KNOT`). The conditions were
//! measured with `tools/size-probes/df_probe3.c`: `DF_NO_BC` fails with
//! -1014 for the default quadratic, and Subbotin fails with -1012 without a
//! knot array.

use onemkl::data_fitting::QuadraticSpline1d;

fn data() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..8).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|v| v * v).collect();
    (x, y)
}

#[test]
fn default_quadratic_reproduces_a_parabola_given_its_value_at_the_first_midpoint() {
    let (x, y) = data();
    // y = x^2, so the value at (x0 + x1) / 2 = 0.5 is 0.25.
    let s = QuadraticSpline1d::with_midpoint_value(x, y, 0.25).unwrap();
    let v = s.interpolate(&[0.5, 2.5, 6.25]).unwrap();
    for (got, want) in v.iter().zip([0.25, 6.25, 39.0625]) {
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    }
}

#[test]
fn default_quadratic_interpolates_the_knots() {
    let (x, y) = data();
    let s = QuadraticSpline1d::with_midpoint_value(x.clone(), y.clone(), 0.25).unwrap();
    let v = s.interpolate(&x).unwrap();
    for (a, b) in v.iter().zip(&y) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }
}

#[test]
fn subbotin_interpolates_the_knots_and_integrates_a_parabola() {
    let (x, y) = data();
    let s = QuadraticSpline1d::subbotin(x.clone(), y.clone()).unwrap();
    let v = s.interpolate(&x).unwrap();
    for (a, b) in v.iter().zip(&y) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }
    let i = s.integrate(&[0.0], &[7.0]).unwrap();
    assert!((i[0] - 343.0 / 3.0).abs() < 1.0, "{}", i[0]);
}

#[test]
fn subbotin_with_explicit_knots_checks_their_shape() {
    let (x, y) = data();
    // n + 1 = 9 knots required: t_0 = x_0, t_n = x_{n-1}, interior knots
    // strictly inside their cells.
    let mut t: Vec<f64> = vec![x[0]];
    t.extend((1..8).map(|i| x[i] - 0.5));
    t.push(x[7]);
    assert!(QuadraticSpline1d::subbotin_with_knots(x.clone(), y.clone(), t).is_ok());
    assert!(QuadraticSpline1d::subbotin_with_knots(x.clone(), y.clone(), vec![0.0, 7.0]).is_err());
    let mut bad: Vec<f64> = vec![x[0]];
    bad.extend((1..8).map(|i| x[i] + 0.5)); // outside the cell
    bad.push(x[7]);
    assert!(QuadraticSpline1d::subbotin_with_knots(x, y, bad).is_err());
}

#[test]
fn too_few_knots_is_an_error() {
    assert!(QuadraticSpline1d::<f64>::with_midpoint_value(vec![0.0], vec![0.0], 0.0).is_err());
    assert!(QuadraticSpline1d::<f64>::subbotin(vec![0.0], vec![0.0]).is_err());
}
