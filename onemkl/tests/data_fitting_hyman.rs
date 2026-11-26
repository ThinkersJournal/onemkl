#![cfg(feature = "data-fitting")]

//! `CubicSpline1d::hyman` (`DF_PP_HYMAN` with `DF_BC_FREE_END`).
//!
//! Measured on oneMKL 2026.2 with `tools/size-probes/df_probe3.c`: on the
//! plateau data below a natural cubic dips by 0.0243 between samples and
//! Hyman with not-a-knot ends still dips by 0.0140, while Hyman with
//! free-end conditions never goes down. `DF_NO_BC` fails with -1014 although
//! the Intel reference lists it as supported (see `docs/DATA-FITTING-FACTS.md`).

use onemkl::data_fitting::CubicSpline1d;

fn plateau() -> (Vec<f64>, Vec<f64>) {
    (
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        vec![0.0, 0.0, 0.1, 1.0, 1.0, 1.1, 2.0, 2.0],
    )
}

fn sites() -> Vec<f64> {
    (0..=70).map(|i| f64::from(i) * 0.1).collect()
}

fn worst_drop(v: &[f64]) -> f64 {
    v.windows(2).map(|w| w[0] - w[1]).fold(0.0, f64::max)
}

#[test]
fn hyman_keeps_monotone_plateau_data_monotone() {
    let (x, y) = plateau();
    let s = CubicSpline1d::hyman(x, y).unwrap();
    let v = s.interpolate(&sites()).unwrap();
    assert!(worst_drop(&v) <= 1e-12, "worst drop {}", worst_drop(&v));
}

#[test]
fn the_data_discriminates_a_natural_spline_dips() {
    // Control: the same data makes a natural cubic go down, so the test above
    // can fail.
    let (x, y) = plateau();
    let s = CubicSpline1d::natural(x, y).unwrap();
    let v = s.interpolate(&sites()).unwrap();
    assert!(worst_drop(&v) > 1e-3, "worst drop {}", worst_drop(&v));
}

#[test]
fn hyman_interpolates_the_knots() {
    let (x, y) = plateau();
    let s = CubicSpline1d::hyman(x.clone(), y.clone()).unwrap();
    let v = s.interpolate(&x).unwrap();
    for (a, b) in v.iter().zip(&y) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
}

#[test]
fn hyman_needs_four_knots() {
    let r = CubicSpline1d::hyman(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]);
    assert!(r.is_err());
}
