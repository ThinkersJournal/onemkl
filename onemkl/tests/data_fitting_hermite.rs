// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "data-fitting")]

//! `CubicSpline1d::hermite` must honour the supplied derivatives at every
//! knot, checked at non-knot sites against an independent cubic Hermite
//! evaluation (the earlier test only looked at the knots).

use approx::assert_abs_diff_eq;

use onemkl::data_fitting::CubicSpline1d;

/// Independent piecewise cubic Hermite interpolation.
fn reference(x: &[f64], y: &[f64], d: &[f64], t: f64) -> f64 {
    let k = (0..x.len() - 1)
        .rev()
        .find(|&k| t >= x[k])
        .unwrap_or(0)
        .min(x.len() - 2);
    let h = x[k + 1] - x[k];
    let s = (t - x[k]) / h;
    let h00 = 2.0 * s.powi(3) - 3.0 * s * s + 1.0;
    let h10 = s.powi(3) - 2.0 * s * s + s;
    let h01 = -2.0 * s.powi(3) + 3.0 * s * s;
    let h11 = s.powi(3) - s * s;
    h00 * y[k] + h10 * h * d[k] + h01 * y[k + 1] + h11 * h * d[k + 1]
}

fn check(x: Vec<f64>, y: Vec<f64>, d: Vec<f64>, eps: f64) {
    let spline = CubicSpline1d::hermite(x.clone(), y.clone(), d.clone()).unwrap();
    // Sites strictly between knots, plus the knots themselves.
    let mut sites: Vec<f64> = Vec::new();
    for w in x.windows(2) {
        for f in [0.1, 0.37, 0.5, 0.83] {
            sites.push(w[0] + f * (w[1] - w[0]));
        }
    }
    sites.extend(x.iter().copied());
    let got = spline.interpolate(&sites).unwrap();
    for (s, g) in sites.iter().zip(&got) {
        assert_abs_diff_eq!(*g, reference(&x, &y, &d, *s), epsilon = eps);
    }
}

#[test]
fn hermite_reproduces_a_cubic_exactly() {
    let x: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|t| t * t * t).collect();
    let d: Vec<f64> = x.iter().map(|t| 3.0 * t * t).collect();
    check(x.clone(), y.clone(), d, 1e-9);
    // and equals x^3 itself
    let spline =
        CubicSpline1d::hermite(x.clone(), y, x.iter().map(|t| 3.0 * t * t).collect()).unwrap();
    let v = spline.interpolate(&[0.5, 2.25, 4.75]).unwrap();
    assert_abs_diff_eq!(v[0], 0.125, epsilon = 1e-9);
    assert_abs_diff_eq!(v[1], 11.390625, epsilon = 1e-9);
    assert_abs_diff_eq!(v[2], 107.171875, epsilon = 1e-9);
}

#[test]
fn hermite_reproduces_a_parabola_exactly() {
    let x: Vec<f64> = (0..5).map(|i| i as f64 * 0.5).collect();
    let y: Vec<f64> = x.iter().map(|t| t * t).collect();
    let d: Vec<f64> = x.iter().map(|t| 2.0 * t).collect();
    check(x, y, d, 1e-9);
}

#[test]
fn hermite_matches_reference_for_a_non_polynomial() {
    let x: Vec<f64> = vec![0.0, 0.4, 1.1, 1.5, 2.6, 3.0, 4.2];
    let y: Vec<f64> = x.iter().map(|t| t.sin()).collect();
    let d: Vec<f64> = x.iter().map(|t| t.cos()).collect();
    check(x, y, d, 1e-9);
}

#[test]
fn hermite_with_three_knots_and_end_slopes() {
    let x = vec![0.0_f64, 2.0, 3.0];
    let y = vec![1.0_f64, 5.0, 4.0];
    let d = vec![0.5_f64, 0.0, -1.0];
    check(x, y, d, 1e-9);
}

#[test]
fn hermite_rejects_two_knots_with_a_typed_error() {
    let r = CubicSpline1d::hermite(vec![0.0_f64, 2.0], vec![1.0, 5.0], vec![0.5, -1.0]);
    assert!(r.is_err());
}
