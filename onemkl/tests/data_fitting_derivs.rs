#![cfg(feature = "data-fitting")]

//! Derivative evaluation and cell search on 1-D cubic splines.

use approx::assert_abs_diff_eq;

use onemkl::data_fitting::CubicSpline1d;

/// A Bessel spline of y = x^2 reproduces the parabola exactly (the knot
/// derivative is the slope of the parabola through three points), so
/// every derivative order has a closed form: x^2, 2x, 2, 0.
fn quad() -> CubicSpline1d<f64> {
    let x: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|t| t * t).collect();
    CubicSpline1d::bessel(x, y).unwrap()
}

#[test]
fn derivatives_of_a_cubic_are_exact() {
    let s = quad();
    let sites = [0.5_f64, 1.25, 2.5, 4.75];
    // One vector per requested derivative order, each over `sites`.
    let d = s.interpolate_derivatives(&sites, &[0, 1, 2, 3]).unwrap();
    assert_eq!(d.len(), 4);
    for (k, &t) in sites.iter().enumerate() {
        assert_abs_diff_eq!(d[0][k], t * t, epsilon = 1e-9);
        assert_abs_diff_eq!(d[1][k], 2.0 * t, epsilon = 1e-9);
        assert_abs_diff_eq!(d[2][k], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(d[3][k], 0.0, epsilon = 1e-9);
    }
}

#[test]
fn derivatives_can_skip_orders_and_keep_request_order() {
    let s = quad();
    let sites = [1.5_f64, 3.5];
    let d = s.interpolate_derivatives(&sites, &[2, 0]).unwrap();
    assert_eq!(d.len(), 2);
    for (k, &t) in sites.iter().enumerate() {
        assert_abs_diff_eq!(d[0][k], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(d[1][k], t * t, epsilon = 1e-9);
    }
}

#[test]
fn derivative_order_above_three_is_rejected() {
    let s = quad();
    assert!(s.interpolate_derivatives(&[1.0], &[4]).is_err());
}

#[test]
fn cells_report_the_interval_of_each_site() {
    let s = quad();
    // Knots are 0..=5, so there are 5 intervals; sites strictly inside.
    let sites = [0.5_f64, 1.5, 4.5, 2.25];
    let cells = s.find_cells(&sites).unwrap();
    assert_eq!(cells, vec![0, 1, 4, 2]);
}
