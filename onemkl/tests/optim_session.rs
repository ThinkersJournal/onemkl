#![cfg(feature = "optim")]

//! Step-by-step TRNLS session API and `?trnlsp_check`.

use approx::assert_abs_diff_eq;

use onemkl::optim::{TrnlsOptions, TrnlsSession, TrnlsStep};

// Fit y = a + b*x through (1, 2) and (2, 3): solution a = b = 1.
fn residual(x: &[f64], f: &mut [f64]) {
    f[0] = x[0] + x[1] - 2.0;
    f[1] = x[0] + 2.0 * x[1] - 3.0;
}
fn jacobian(j: &mut [f64]) {
    j.copy_from_slice(&[1.0, 1.0, 1.0, 2.0]); // column-major 2x2
}

#[test]
fn session_drives_the_solver_step_by_step() {
    let mut s = TrnlsSession::new(2, 2, &[0.0, 0.0], TrnlsOptions::default()).unwrap();
    let mut residual_requests = 0;
    let result = loop {
        assert!(residual_requests < 1000, "solver did not converge");
        match s.step().unwrap() {
            TrnlsStep::NeedResidual => {
                residual_requests += 1;
                let x = s.x().to_vec();
                residual(&x, s.residual_mut());
            }
            TrnlsStep::NeedJacobian => jacobian(s.jacobian_mut()),
            TrnlsStep::Done(r) => break r,
        }
    };
    assert!(residual_requests > 0);
    assert_abs_diff_eq!(s.x()[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(s.x()[1], 1.0, epsilon = 1e-6);
    assert!(result.final_residual_norm < 1e-6);
}

#[test]
fn check_accepts_good_data_and_flags_a_nan_jacobian() {
    let mut s = TrnlsSession::new(2, 2, &[0.0, 0.0], TrnlsOptions::default()).unwrap();
    let x = s.x().to_vec();
    residual(&x, s.residual_mut());
    jacobian(s.jacobian_mut());
    assert!(s.check().unwrap().is_valid());

    s.jacobian_mut()[2] = f64::NAN;
    let c = s.check().unwrap();
    assert!(!c.is_valid(), "NaN in the Jacobian must be reported: {c:?}");
}
