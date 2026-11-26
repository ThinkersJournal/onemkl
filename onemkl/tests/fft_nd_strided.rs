// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "fft")]

//! N-D complex in-place FFT over a strided layout (padded leading
//! dimension, column-major), checked against a naive N-D DFT that reads
//! elements through the same strides.

use approx::assert_abs_diff_eq;
use num_complex::Complex;
use onemkl::fft::StridedFftPlan;

/// Naive 2-D forward DFT over logical (j0, j1) with element strides.
fn dft2(x: &[Complex<f64>], n: [usize; 2], s: [usize; 2]) -> Vec<Complex<f64>> {
    let mut out = vec![Complex::new(0.0, 0.0); x.len()];
    for k0 in 0..n[0] {
        for k1 in 0..n[1] {
            let mut acc = Complex::new(0.0, 0.0);
            for j0 in 0..n[0] {
                for j1 in 0..n[1] {
                    let a = -2.0
                        * std::f64::consts::PI
                        * ((j0 * k0) as f64 / n[0] as f64 + (j1 * k1) as f64 / n[1] as f64);
                    acc += x[j0 * s[0] + j1 * s[1]] * Complex::new(a.cos(), a.sin());
                }
            }
            out[k0 * s[0] + k1 * s[1]] = acc;
        }
    }
    out
}

fn fill(len: usize) -> Vec<Complex<f64>> {
    (0..len)
        .map(|i| Complex::new((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos()))
        .collect()
}

fn check(n: [usize; 2], s: [usize; 2]) {
    let mut plan = StridedFftPlan::<f64>::new(&n, &s).unwrap();
    assert_eq!(plan.buffer_len(), (n[0] - 1) * s[0] + (n[1] - 1) * s[1] + 1);
    let mut x = fill(plan.buffer_len());
    let want = dft2(&x, n, s);
    plan.forward_in_place(&mut x).unwrap();
    for k0 in 0..n[0] {
        for k1 in 0..n[1] {
            let i = k0 * s[0] + k1 * s[1];
            assert_abs_diff_eq!(x[i].re, want[i].re, epsilon = 1e-9);
            assert_abs_diff_eq!(x[i].im, want[i].im, epsilon = 1e-9);
        }
    }
}

#[test]
fn row_major_padded_leading_dimension() {
    check([4, 5], [7, 1]);
}

#[test]
fn column_major() {
    check([4, 5], [1, 4]);
}

#[test]
fn column_major_padded() {
    check([6, 3], [1, 9]);
}

#[test]
fn round_trip_with_backward() {
    let (n, s) = ([4usize, 6usize], [8usize, 1usize]);
    let mut plan = StridedFftPlan::<f64>::with_scales(&n, &s, 1.0, 1.0 / 24.0).unwrap();
    let orig = fill(plan.buffer_len());
    let mut x = orig.clone();
    plan.forward_in_place(&mut x).unwrap();
    plan.backward_in_place(&mut x).unwrap();
    for j0 in 0..n[0] {
        for j1 in 0..n[1] {
            let i = j0 * s[0] + j1;
            assert_abs_diff_eq!(x[i].re, orig[i].re, epsilon = 1e-12);
            assert_abs_diff_eq!(x[i].im, orig[i].im, epsilon = 1e-12);
        }
    }
}

#[test]
fn rejects_bad_arguments() {
    assert!(
        StridedFftPlan::<f64>::new(&[4, 5], &[1]).is_err(),
        "rank mismatch"
    );
    assert!(StridedFftPlan::<f64>::new(&[], &[]).is_err(), "empty");
    assert!(
        StridedFftPlan::<f64>::new(&[4, 5], &[1, 2]).is_err(),
        "overlapping strides"
    );
    let mut p = StridedFftPlan::<f64>::new(&[4, 5], &[5, 1]).unwrap();
    let mut small = vec![Complex::new(0.0, 0.0); 3];
    assert!(p.forward_in_place(&mut small).is_err());
}
