#![cfg(feature = "data-fitting")]

//! Derivative evaluation and cell search on 1-D cubic splines.

use approx::assert_abs_diff_eq;

use onemkl::data_fitting::CubicSpline1d;

/// A Hermite cubic with exact derivatives of y = x^3 reproduces x^3
/// exactly, so every derivative order has a closed form.
fn cube() -> CubicSpline1d<f64> {
    let x: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|t| t * t * t).collect();
    let dy: Vec<f64> = x.iter().map(|t| 3.0 * t * t).collect();
    CubicSpline1d::hermite(x, y, dy).unwrap()
}

#[test]
fn derivatives_of_a_cubic_are_exact() {
    let s = cube();
    let sites = [0.5_f64, 1.25, 2.5, 4.75];
    // One vector per requested derivative order, each over `sites`.
    let d = s.interpolate_derivatives(&sites, &[0, 1, 2, 3]).unwrap();
    assert_eq!(d.len(), 4);
    for (k, &t) in sites.iter().enumerate() {
        assert_abs_diff_eq!(d[0][k], t * t * t, epsilon = 1e-9);
        assert_abs_diff_eq!(d[1][k], 3.0 * t * t, epsilon = 1e-9);
        assert_abs_diff_eq!(d[2][k], 6.0 * t, epsilon = 1e-9);
        assert_abs_diff_eq!(d[3][k], 6.0, epsilon = 1e-9);
    }
}

#[test]
fn derivatives_can_skip_orders_and_keep_request_order() {
    let s = cube();
    let sites = [1.5_f64, 3.5];
    let d = s.interpolate_derivatives(&sites, &[2, 0]).unwrap();
    assert_eq!(d.len(), 2);
    for (k, &t) in sites.iter().enumerate() {
        assert_abs_diff_eq!(d[0][k], 6.0 * t, epsilon = 1e-9);
        assert_abs_diff_eq!(d[1][k], t * t * t, epsilon = 1e-9);
    }
}

#[test]
fn derivative_order_above_three_is_rejected() {
    let s = cube();
    assert!(s.interpolate_derivatives(&[1.0], &[4]).is_err());
}

#[test]
fn cells_report_the_interval_of_each_site() {
    let s = cube();
    // Knots are 0..=5, so there are 5 intervals; sites strictly inside.
    let sites = [0.5_f64, 1.5, 4.5, 2.25];
    let cells = s.find_cells(&sites).unwrap();
    assert_eq!(cells, vec![0, 1, 4, 2]);
}
