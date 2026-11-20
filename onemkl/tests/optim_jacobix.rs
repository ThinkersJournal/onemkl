#![cfg(feature = "optim")]

//! `numerical_jacobian_central` (`djacobix`): layout, accuracy, panics.

use approx::assert_abs_diff_eq;
use onemkl::optim::numerical_jacobian_central;

// F: R^2 -> R^3.  F = [x0^2 + x1, 3 x0 + x1^3, x0 * x1]
// J(2, 3) = [[4, 1], [3, 27], [3, 2]]  (rows = outputs, cols = inputs)
fn f(x: &[f64], out: &mut [f64]) {
    out[0] = x[0] * x[0] + x[1];
    out[1] = 3.0 * x[0] + x[1] * x[1] * x[1];
    out[2] = x[0] * x[1];
}

#[test]
fn central_difference_jacobian_matches_analytic_and_reports_layout() {
    let x = [2.0, 3.0];
    let mut jac = vec![0.0; 6];
    numerical_jacobian_central(2, 3, &x, 1e-8, &mut jac, f).unwrap();
    println!("jac = {jac:?}");
    // Column-major m x n (columns = inputs): [4,3,3, 1,27,2].
    let expect = [4.0, 3.0, 3.0, 1.0, 27.0, 2.0];
    for (g, e) in jac.iter().zip(expect) {
        assert_abs_diff_eq!(*g, e, epsilon = 1e-5);
    }
    // The caller's x is untouched.
    assert_eq!(x, [2.0, 3.0]);
}

#[test]
fn rejects_bad_arguments() {
    let mut jac = vec![0.0; 6];
    assert!(numerical_jacobian_central(2, 3, &[1.0], 1e-6, &mut jac, f).is_err());
    assert!(numerical_jacobian_central(2, 3, &[1.0, 2.0], 1e-6, &mut jac[..5], f).is_err());
    assert!(numerical_jacobian_central(2, 3, &[1.0, 2.0], 0.0, &mut jac, f).is_err());
    assert!(numerical_jacobian_central(2, 3, &[1.0, 2.0], f64::NAN, &mut jac, f).is_err());
}

#[test]
fn a_panic_in_the_residual_is_reraised_after_mkl_returns() {
    let mut jac = vec![0.0; 6];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        numerical_jacobian_central(2, 3, &[2.0, 3.0], 1e-6, &mut jac, |_, _| panic!("boom"))
    }));
    let payload = r.expect_err("panic must propagate");
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"boom"));
}
