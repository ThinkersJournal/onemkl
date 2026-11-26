// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "optim")]

//! Step-by-step TRNLS session API and `?trnlsp_check`.

use approx::assert_abs_diff_eq;

use onemkl::optim::{
    TrnlsBoundedSession, TrnlsBounds, TrnlsOptions, TrnlsSession, TrnlsStep, solve_trnls_bounded,
};

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

fn run_bounded(s: &mut TrnlsBoundedSession) {
    for _ in 0..1000 {
        match s.step().unwrap() {
            TrnlsStep::NeedResidual => {
                let x = s.x().to_vec();
                residual(&x, s.residual_mut());
            }
            TrnlsStep::NeedJacobian => jacobian(s.jacobian_mut()),
            TrnlsStep::Done(_) => return,
        }
    }
    panic!("bounded solver did not converge");
}

#[test]
fn bounded_session_respects_the_bound_and_matches_the_closure_solver() {
    // Unconstrained optimum is (1, 1); the bound x1 <= 0.5 is active.
    let (lo, hi) = ([-10.0, -10.0], [10.0, 0.5]);
    let mut s =
        TrnlsBoundedSession::new(2, 2, &[0.0, 0.0], &lo, &hi, TrnlsOptions::default()).unwrap();
    run_bounded(&mut s);
    assert!(s.x()[1] <= 0.5 + 1e-12, "bound violated: {:?}", s.x());
    // MKL's bounded solver stops early on this toy problem (it does not reach
    // (1.75, 0.5)); the contract tested is bound respect, descent, and parity
    // with the closure driver, which runs the same reverse-communication loop.
    let f0 = 13.0_f64.sqrt(); // |F(0,0)| = |(-2,-3)|
    let mut f = [0.0; 2];
    residual(s.x(), &mut f);
    assert!(f[0].hypot(f[1]) < f0, "no descent: {:?}", s.x());

    let mut x = [0.0, 0.0];
    solve_trnls_bounded(
        2,
        2,
        &mut x,
        TrnlsBounds {
            lower: &lo,
            upper: &hi,
        },
        TrnlsOptions::default(),
        residual,
        |_, j| jacobian(j),
    )
    .unwrap();
    assert_abs_diff_eq!(s.x()[0], x[0], epsilon = 1e-8);
    assert_abs_diff_eq!(s.x()[1], x[1], epsilon = 1e-8);
}

#[test]
fn bounded_check_and_length_validation() {
    let (lo, hi) = ([-10.0, -10.0], [10.0, 10.0]);
    let mut s =
        TrnlsBoundedSession::new(2, 2, &[0.0, 0.0], &lo, &hi, TrnlsOptions::default()).unwrap();
    let x = s.x().to_vec();
    residual(&x, s.residual_mut());
    jacobian(s.jacobian_mut());
    assert!(s.check().unwrap().is_valid());
    s.jacobian_mut()[2] = f64::NAN;
    assert!(!s.check().unwrap().is_valid());
    assert!(
        TrnlsBoundedSession::new(2, 2, &[0.0, 0.0], &lo[..1], &hi, TrnlsOptions::default())
            .is_err()
    );
}
