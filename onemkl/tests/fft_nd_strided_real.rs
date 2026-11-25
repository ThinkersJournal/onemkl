#![cfg(feature = "fft")]

//! N-D real FFT (out-of-place, CCE output) over strided layouts on both
//! sides, checked against a naive 2-D real DFT. The spectrum keeps
//! `n1 / 2 + 1` columns (last dimension halved).

use approx::assert_abs_diff_eq;
use num_complex::Complex;
use onemkl::fft::StridedRealFftPlan;

fn signal(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| (i as f64 * 0.43).sin() + 0.2 * (i as f64 * 0.07).cos())
        .collect()
}

/// Naive spectrum bin (k0, k1) of the logical n0 x n1 real input read
/// through the real strides.
fn bin(x: &[f64], n: [usize; 2], s: [usize; 2], k: [usize; 2]) -> Complex<f64> {
    let mut acc = Complex::new(0.0, 0.0);
    for j0 in 0..n[0] {
        for j1 in 0..n[1] {
            let a = -2.0
                * std::f64::consts::PI
                * ((j0 * k[0]) as f64 / n[0] as f64 + (j1 * k[1]) as f64 / n[1] as f64);
            acc += Complex::new(a.cos(), a.sin()) * x[j0 * s[0] + j1 * s[1]];
        }
    }
    acc
}

fn check(n: [usize; 2], rs: [usize; 2], cs: [usize; 2]) {
    let m = n[1] / 2 + 1;
    let mut plan = StridedRealFftPlan::<f64>::new(&n, &rs, &cs).unwrap();
    assert_eq!(
        plan.real_buffer_len(),
        (n[0] - 1) * rs[0] + (n[1] - 1) * rs[1] + 1
    );
    assert_eq!(
        plan.complex_buffer_len(),
        (n[0] - 1) * cs[0] + (m - 1) * cs[1] + 1
    );
    let x = signal(plan.real_buffer_len());
    let mut out = vec![Complex::new(f64::NAN, f64::NAN); plan.complex_buffer_len()];
    plan.forward(&x, &mut out).unwrap();
    for k0 in 0..n[0] {
        for k1 in 0..m {
            let want = bin(&x, n, rs, [k0, k1]);
            let got = out[k0 * cs[0] + k1 * cs[1]];
            assert_abs_diff_eq!(got.re, want.re, epsilon = 1e-9);
            assert_abs_diff_eq!(got.im, want.im, epsilon = 1e-9);
        }
    }
}

#[test]
fn padded_real_dense_complex() {
    check([4, 6], [8, 1], [4, 1]);
}

#[test]
fn dense_real_padded_complex() {
    check([4, 6], [6, 1], [7, 1]);
}

#[test]
fn column_major_both_sides() {
    check([4, 5], [1, 4], [1, 4]);
}

#[test]
fn round_trip_strided() {
    let (n, rs, cs) = ([4usize, 6usize], [9usize, 1usize], [5usize, 1usize]);
    let mut plan = StridedRealFftPlan::<f64>::with_scales(&n, &rs, &cs, 1.0, 1.0 / 24.0).unwrap();
    let x = signal(plan.real_buffer_len());
    let mut spec = vec![Complex::new(0.0, 0.0); plan.complex_buffer_len()];
    plan.forward(&x, &mut spec).unwrap();
    let mut back = vec![-7.0; plan.real_buffer_len()];
    plan.backward(&spec, &mut back).unwrap();
    for j0 in 0..n[0] {
        for j1 in 0..n[1] {
            let i = j0 * rs[0] + j1 * rs[1];
            assert_abs_diff_eq!(back[i], x[i], epsilon = 1e-12);
        }
    }
}

#[test]
fn rejects_bad_arguments() {
    assert!(
        StridedRealFftPlan::<f64>::new(&[4, 6], &[6, 1], &[4]).is_err(),
        "rank mismatch"
    );
    assert!(
        StridedRealFftPlan::<f64>::new(&[], &[], &[]).is_err(),
        "empty"
    );
    assert!(
        StridedRealFftPlan::<f64>::new(&[4, 6], &[6, 2], &[4, 1]).is_err(),
        "real overlap"
    );
    // spectrum has 4 columns, so a column stride of 1 with row stride 3 overlaps
    assert!(
        StridedRealFftPlan::<f64>::new(&[4, 6], &[6, 1], &[3, 1]).is_err(),
        "complex overlap"
    );
    let mut p = StridedRealFftPlan::<f64>::new(&[4, 6], &[6, 1], &[4, 1]).unwrap();
    let mut out = vec![Complex::new(0.0, 0.0); 16];
    assert!(p.forward(&[0.0; 3], &mut out).is_err(), "short real input");
    assert!(
        p.forward(&[0.0; 24], &mut out[..3]).is_err(),
        "short complex output"
    );
}
