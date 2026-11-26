// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "optim")]

//! `numerical_jacobian` (djacobi RCI): what the closure sees and the result.

use approx::assert_abs_diff_eq;
use onemkl::optim::numerical_jacobian;

#[test]
fn closure_sees_perturbed_x_result_is_analytic_and_callers_x_is_untouched() {
    let x = [2.0f64, 3.0];
    let mut jac = vec![0.0; 6];
    let mut seen: Vec<Vec<f64>> = Vec::new();
    let mut ptrs: Vec<usize> = Vec::new();
    numerical_jacobian(2, 3, &x, 1e-6, &mut jac, |xs, f| {
        seen.push(xs.to_vec());
        ptrs.push(xs.as_ptr() as usize);
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

    // MKL must perturb an owned copy, never the caller's slice: the closure's
    // buffer is a different allocation from `x`.
    assert!(ptrs.iter().all(|p| *p != x.as_ptr() as usize));
}

#[test]
fn dimension_product_overflow_is_an_error_not_a_wrap() {
    let x = [1.0f64, 2.0];
    let mut jac = vec![0.0; 4];
    let r = numerical_jacobian(2, usize::MAX, &x, 1e-6, &mut jac, |_, _| {});
    assert!(matches!(r, Err(onemkl::Error::DimensionOverflow)));
}
