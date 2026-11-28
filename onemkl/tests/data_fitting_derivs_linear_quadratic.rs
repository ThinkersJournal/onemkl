// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "data-fitting")]

//! Derivative evaluation (`interpolate_derivatives`, with and without
//! call-backs) on linear and quadratic splines. Measured on oneMKL 2026.2
//! under WSL (Linux); the call-back layout is site-major with the orders
//! ascending, as for the cubic spline.

use approx::assert_abs_diff_eq;

use onemkl::data_fitting::{
    InterpolateCallbacks, InterpolationPieces, LinearSpline1d, QuadraticSpline1d,
};

fn knots() -> Vec<f64> {
    (0..6).map(f64::from).collect()
}

#[test]
fn linear_derivatives_are_exact() {
    // y = 2x + 1: value 2x + 1, first derivative 2.
    let x = knots();
    let y: Vec<f64> = x.iter().map(|t| 2.0 * t + 1.0).collect();
    let s = LinearSpline1d::new(x, y).unwrap();
    let sites = [0.5_f64, 1.25, 2.5, 4.75];
    let d = s.interpolate_derivatives(&sites, &[1, 0]).unwrap();
    for (k, &t) in sites.iter().enumerate() {
        assert_abs_diff_eq!(d[0][k], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(d[1][k], 2.0 * t + 1.0, epsilon = 1e-9);
    }
    assert!(s.interpolate_derivatives(&sites, &[2]).is_err());
    assert!(s.interpolate_derivatives(&sites, &[0, 0]).is_err());
}

#[test]
fn quadratic_derivatives_are_exact() {
    // y = x^2 with the midpoint value of the first cell fixed to 0.25.
    let x = knots();
    let y: Vec<f64> = x.iter().map(|t| t * t).collect();
    let s = QuadraticSpline1d::with_midpoint_value(x, y, 0.25).unwrap();
    let sites = [0.5_f64, 1.25, 2.5, 4.75];
    let d = s.interpolate_derivatives(&sites, &[0, 1, 2]).unwrap();
    for (k, &t) in sites.iter().enumerate() {
        assert_abs_diff_eq!(d[0][k], t * t, epsilon = 1e-9);
        assert_abs_diff_eq!(d[1][k], 2.0 * t, epsilon = 1e-9);
        assert_abs_diff_eq!(d[2][k], 2.0, epsilon = 1e-9);
    }
    assert!(s.interpolate_derivatives(&sites, &[3]).is_err());
}

/// An interior call-back that writes the exact values site-major
/// (`out[i * orders + k]`, orders ascending) must give the same result as the
/// library's own evaluation: this is the layout measurement.
#[test]
fn callbacks_use_the_site_major_layout_for_linear_and_quadratic() {
    let x = knots();
    let sites = [0.5_f64, 1.25, 2.5, 4.75];

    let ylin: Vec<f64> = x.iter().map(|t| 2.0 * t + 1.0).collect();
    let lin = LinearSpline1d::new(x.clone(), ylin).unwrap();
    let control = lin.interpolate_derivatives(&sites, &[0, 1]).unwrap();
    let mut cb = |p: InterpolationPieces<'_, f64>| {
        assert_eq!(p.orders, 2);
        for (i, s) in p.sites.iter().enumerate() {
            p.out[i * 2] = 2.0 * s + 1.0;
            p.out[i * 2 + 1] = 2.0;
        }
        0
    };
    let got = lin
        .interpolate_derivatives_with_callbacks(
            &sites,
            &[0, 1],
            InterpolateCallbacks::new().interior(&mut cb),
        )
        .unwrap();
    assert_eq!(got, control);

    let yq: Vec<f64> = x.iter().map(|t| t * t).collect();
    let quad = QuadraticSpline1d::with_midpoint_value(x, yq, 0.25).unwrap();
    let control = quad.interpolate_derivatives(&sites, &[0, 2]).unwrap();
    let mut cb = |p: InterpolationPieces<'_, f64>| {
        assert_eq!(p.orders, 2);
        for (i, s) in p.sites.iter().enumerate() {
            p.out[i * 2] = s * s;
            p.out[i * 2 + 1] = 2.0;
        }
        0
    };
    let got = quad
        .interpolate_derivatives_with_callbacks(
            &sites,
            &[2, 0],
            InterpolateCallbacks::new().interior(&mut cb),
        )
        .unwrap();
    // `got` is in the requested order [2, 0]; `control` was [0, 2].
    assert_eq!(got[0], control[1]);
    assert_eq!(got[1], control[0]);
}
