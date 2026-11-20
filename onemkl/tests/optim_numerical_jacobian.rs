#![cfg(feature = "optim")]

//! `numerical_jacobian` (djacobi RCI): what the closure sees and the result.

use approx::assert_abs_diff_eq;
use onemkl::optim::numerical_jacobian;

#[test]
fn closure_sees_perturbed_x_result_is_analytic_and_callers_x_is_untouched() {
    let x = [2.0f64, 3.0];
    let mut jac = vec![0.0; 6];
    let mut seen: Vec<Vec<f64>> = Vec::new();
    numerical_jacobian(2, 3, &x, 1e-6, &mut jac, |xs, f| {
        seen.push(xs.to_vec());
        f[0] = xs[0] * xs[0] + xs[1];
        f[1] = 3.0 * xs[0] + xs[1] * xs[1] * xs[1];
        f[2] = xs[0] * xs[1];
    })
    .unwrap();

    // Measured (oneMKL 2026.2.1, Linux x86_64, debug and release): four
    // calls, each with one coordinate moved by +/- 2e-6 (+/- 3e-6 for x1).
    assert_eq!(seen.len(), 4);
    assert!(seen.iter().all(|v| v.as_slice() != [2.0, 3.0]));

    // Column-major m x n, columns = inputs.
    let expect = [4.0, 3.0, 3.0, 1.0, 27.0, 2.0];
    for (g, e) in jac.iter().zip(expect) {
        assert_abs_diff_eq!(*g, e, epsilon = 1e-6);
    }
    assert_eq!(x, [2.0, 3.0]);
}
