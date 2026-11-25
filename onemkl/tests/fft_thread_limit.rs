#![cfg(feature = "fft")]

//! `DFTI_THREAD_LIMIT` on complex and real plans: the limit is applied at
//! plan time, read back with `thread_limit()`, and never changes results.

use approx::assert_abs_diff_eq;
use num_complex::Complex;
use onemkl::fft::{FftPlan, RealFftPlan};

fn data(n: usize) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| Complex::new((i as f64 * 0.7).sin(), (i as f64 * 0.2).cos()))
        .collect()
}

#[test]
fn complex_limit_one_reads_back_and_matches_default() {
    let dims = [8usize, 6usize];
    let mut limited = FftPlan::<f64>::complex_nd_with_thread_limit(&dims, 1).unwrap();
    assert_eq!(limited.thread_limit().unwrap(), 1);
    let mut plain = FftPlan::<f64>::complex_nd(&dims).unwrap();
    let mut a = data(48);
    let mut b = a.clone();
    limited.forward_in_place(&mut a).unwrap();
    plain.forward_in_place(&mut b).unwrap();
    for (x, y) in a.iter().zip(&b) {
        assert_abs_diff_eq!(x.re, y.re, epsilon = 1e-12);
        assert_abs_diff_eq!(x.im, y.im, epsilon = 1e-12);
    }
}

#[test]
fn requested_limit_is_clamped_not_exceeded() {
    let plan = FftPlan::<f64>::complex_nd_with_thread_limit(&[16], 4).unwrap();
    let got = plan.thread_limit().unwrap();
    assert!((1..=4).contains(&got), "limit read back as {got}");
}

#[test]
fn real_limit_one_reads_back_and_round_trips() {
    let dims = [4usize, 6usize];
    let mut plan = RealFftPlan::<f64>::real_nd_with_thread_limit(&dims, 1).unwrap();
    assert_eq!(plan.thread_limit().unwrap(), 1);
    let x: Vec<f64> = (0..24).map(|i| ((i * 5) % 7) as f64 - 2.0).collect();
    let mut spec = vec![Complex::new(0.0, 0.0); plan.complex_len()];
    plan.forward(&x, &mut spec).unwrap();
    let mut back = vec![0.0; 24];
    plan.backward(&spec, &mut back).unwrap();
    for (b, o) in back.iter().zip(&x) {
        assert_abs_diff_eq!(*b / 24.0, *o, epsilon = 1e-12);
    }
}

#[test]
fn plain_plans_report_a_positive_limit() {
    assert!(
        FftPlan::<f64>::complex_1d(8)
            .unwrap()
            .thread_limit()
            .unwrap()
            >= 1
    );
}
