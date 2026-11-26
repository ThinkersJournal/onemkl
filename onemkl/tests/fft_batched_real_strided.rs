#![cfg(feature = "fft")]

//! Batched 1-D real FFT with strides within each transform as well as
//! distances between them (interleaved batches included), out of place,
//! CCE spectrum, checked against a naive DFT read through the layouts.

use approx::assert_abs_diff_eq;
use num_complex::Complex;
use onemkl::fft::StridedBatchedRealFftPlan;

fn signal(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| (i as f64 * 0.31).sin() + 0.1 * (i % 5) as f64)
        .collect()
}

/// (n, howmany, real stride, real distance, complex stride, complex distance)
type Layout = (usize, usize, usize, usize, usize, usize);

fn check((n, b, rs, rd, cs, cd): Layout) {
    let h = n / 2 + 1;
    let mut plan = StridedBatchedRealFftPlan::<f64>::new(n, b, rs, rd, cs, cd).unwrap();
    assert_eq!(plan.real_buffer_len(), (b - 1) * rd + (n - 1) * rs + 1);
    assert_eq!(plan.complex_buffer_len(), (b - 1) * cd + (h - 1) * cs + 1);
    let x = signal(plan.real_buffer_len());
    let mut out = vec![Complex::new(f64::NAN, f64::NAN); plan.complex_buffer_len()];
    plan.forward(&x, &mut out).unwrap();
    for t in 0..b {
        for k in 0..h {
            let (mut re, mut im) = (0.0, 0.0);
            for j in 0..n {
                let a = -2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
                let v = x[t * rd + j * rs];
                re += v * a.cos();
                im += v * a.sin();
            }
            let got = out[t * cd + k * cs];
            assert_abs_diff_eq!(got.re, re, epsilon = 1e-9);
            assert_abs_diff_eq!(got.im, im, epsilon = 1e-9);
        }
    }
}

#[test]
fn interleaved_real_and_complex() {
    // element j of transform t at j * 3 + t, in both domains
    check((8, 3, 3, 1, 3, 1));
}

#[test]
fn contiguous_real_interleaved_complex() {
    check((6, 4, 1, 6, 4, 1));
}

#[test]
fn padded_strides_and_distances() {
    check((7, 3, 2, 20, 3, 13));
}

#[test]
fn round_trip_interleaved() {
    let (n, b) = (8usize, 3usize);
    let mut plan =
        StridedBatchedRealFftPlan::<f64>::with_scales(n, b, 3, 1, 3, 1, 1.0, 1.0 / n as f64)
            .unwrap();
    let x = signal(plan.real_buffer_len());
    let mut spec = vec![Complex::new(0.0, 0.0); plan.complex_buffer_len()];
    plan.forward(&x, &mut spec).unwrap();
    let mut back = vec![-9.0; plan.real_buffer_len()];
    plan.backward(&spec, &mut back).unwrap();
    for i in 0..plan.real_buffer_len() {
        assert_abs_diff_eq!(back[i], x[i], epsilon = 1e-12);
    }
}

#[test]
fn rejects_bad_arguments() {
    let new = |n, b, rs, rd, cs, cd| StridedBatchedRealFftPlan::<f64>::new(n, b, rs, rd, cs, cd);
    assert!(new(0, 3, 1, 8, 1, 5).is_err(), "zero length");
    assert!(new(8, 0, 1, 8, 1, 5).is_err(), "zero batch");
    assert!(new(8, 3, 0, 8, 1, 5).is_err(), "zero stride");
    assert!(new(8, 3, 1, 4, 1, 5).is_err(), "real transforms overlap");
    assert!(new(8, 3, 1, 8, 1, 3).is_err(), "complex transforms overlap");
    let mut p = new(8, 3, 1, 8, 1, 5).unwrap();
    let mut out = vec![Complex::new(0.0, 0.0); 15];
    assert!(p.forward(&[0.0; 5], &mut out).is_err(), "short real input");
    assert!(
        p.forward(&[0.0; 24], &mut out[..4]).is_err(),
        "short complex output"
    );
}
