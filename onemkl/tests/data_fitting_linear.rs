#![cfg(feature = "data-fitting")]

//! `LinearSpline1d` (`DF_PP_LINEAR`, default type, no boundary or internal
//! conditions: the combination that constructs on oneMKL 2026.2, see
//! `tools/size-probes/df_types.c`).

use approx::assert_abs_diff_eq;

use onemkl::data_fitting::LinearSpline1d;

fn knots() -> (Vec<f64>, Vec<f64>) {
    (vec![0.0, 1.0, 3.0, 4.0, 7.0], vec![0.0, 1.0, 7.0, 5.0, 5.0])
}

#[test]
fn interpolation_is_piecewise_linear_between_knots() {
    let (x, y) = knots();
    let s = LinearSpline1d::new(x, y).unwrap();
    let v = s.interpolate(&[0.5, 2.0, 3.5, 5.0, 7.0]).unwrap();
    // 0.5 on [0,1]: 0.5; 2.0 on [1,3]: 1 + 3*(1) = 4; 3.5 on [3,4]: 6; flat 5.
    let want = [0.5, 4.0, 6.0, 5.0, 5.0];
    for (a, b) in v.iter().zip(want) {
        assert_abs_diff_eq!(*a, b, epsilon = 1e-12);
    }
}

#[test]
fn integral_over_all_knots_is_the_trapezoid_sum() {
    let (x, y) = knots();
    let mut trap = 0.0;
    for k in 0..x.len() - 1 {
        trap += 0.5 * (y[k] + y[k + 1]) * (x[k + 1] - x[k]);
    }
    let s = LinearSpline1d::new(x, y).unwrap();
    let i = s.integrate(&[0.0], &[7.0]).unwrap();
    assert_abs_diff_eq!(i[0], trap, epsilon = 1e-12);
}

#[test]
fn fewer_than_two_knots_is_an_error() {
    assert!(LinearSpline1d::new(vec![1.0], vec![2.0]).is_err());
    assert!(LinearSpline1d::new(vec![0.0, 1.0], vec![0.0]).is_err());
}
