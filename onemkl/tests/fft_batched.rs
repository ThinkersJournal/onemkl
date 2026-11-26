// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "fft")]

//! Batched, strided 1-D complex FFT (`DFTI_NUMBER_OF_TRANSFORMS`,
//! distance, stride), checked against a naive O(n^2) DFT (epsilon 1e-9).

use approx::assert_abs_diff_eq;
use num_complex::Complex64;
use onemkl::fft::BatchedFftPlan;

fn dft(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    (0..n)
        .map(|k| {
            x.iter()
                .enumerate()
                .map(|(j, &v)| {
                    let a = -2.0 * std::f64::consts::PI * ((j * k) % n) as f64 / n as f64;
                    v * Complex64::new(a.cos(), a.sin())
                })
                .sum()
        })
        .collect()
}

fn val(t: usize, j: usize) -> Complex64 {
    Complex64::new(
        ((t * 5 + j * 3) % 7) as f64 - 2.0,
        ((t * 2 + j) % 5) as f64 - 1.5,
    )
}

/// Fill a buffer with a sentinel, put transform `t` element `j` at
/// `t*dist + j*stride`, run forward, and compare against the naive DFT.
fn check(n: usize, howmany: usize, dist: usize, stride: usize) {
    let mut plan = BatchedFftPlan::<f64>::new(n, howmany, dist, stride).unwrap();
    let len = plan.buffer_len();
    assert_eq!(len, (howmany - 1) * dist + (n - 1) * stride + 1);
    let sentinel = Complex64::new(777.0, -777.0);
    let mut buf = vec![sentinel; len];
    for t in 0..howmany {
        for j in 0..n {
            buf[t * dist + j * stride] = val(t, j);
        }
    }
    plan.forward_in_place(&mut buf).unwrap();
    let mut touched = vec![false; len];
    for t in 0..howmany {
        let x: Vec<_> = (0..n).map(|j| val(t, j)).collect();
        let want = dft(&x);
        for k in 0..n {
            let g = buf[t * dist + k * stride];
            touched[t * dist + k * stride] = true;
            assert_abs_diff_eq!(g.re, want[k].re, epsilon = 1e-9);
            assert_abs_diff_eq!(g.im, want[k].im, epsilon = 1e-9);
        }
    }
    for (i, b) in buf.iter().enumerate() {
        if !touched[i] {
            assert_eq!(*b, sentinel, "padding at {i} was modified");
        }
    }
}

#[test]
fn batched_contiguous_matches_naive_dft() {
    check(8, 3, 8, 1); // even n
    check(9, 4, 9, 1); // odd n
}

#[test]
fn batched_distance_padding_is_untouched() {
    check(8, 3, 11, 1);
    check(9, 2, 13, 1);
}

#[test]
fn batched_stride_matches_naive_dft() {
    check(8, 3, 17, 2);
    check(9, 3, 28, 3);
}

#[test]
fn batched_round_trip_scaled() {
    let (n, h, d, s) = (10, 3, 24, 2);
    let mut plan = BatchedFftPlan::<f64>::with_scales(n, h, d, s, 1.0, 1.0 / n as f64).unwrap();
    let mut buf = vec![Complex64::new(0.0, 0.0); plan.buffer_len()];
    for t in 0..h {
        for j in 0..n {
            buf[t * d + j * s] = val(t, j);
        }
    }
    plan.forward_in_place(&mut buf).unwrap();
    plan.backward_in_place(&mut buf).unwrap();
    for t in 0..h {
        for j in 0..n {
            let g = buf[t * d + j * s];
            assert_abs_diff_eq!(g.re, val(t, j).re, epsilon = 1e-12);
            assert_abs_diff_eq!(g.im, val(t, j).im, epsilon = 1e-12);
        }
    }
}

#[test]
fn batched_rejects_bad_arguments() {
    assert!(BatchedFftPlan::<f64>::new(0, 1, 1, 1).is_err());
    assert!(BatchedFftPlan::<f64>::new(8, 0, 8, 1).is_err());
    assert!(BatchedFftPlan::<f64>::new(8, 2, 8, 0).is_err());
    // distance smaller than one transform's extent overlaps the next one
    assert!(BatchedFftPlan::<f64>::new(8, 2, 7, 1).is_err());
    assert!(BatchedFftPlan::<f64>::new(8, 2, 14, 2).is_err());
    let mut plan = BatchedFftPlan::<f64>::new(8, 2, 8, 1).unwrap();
    let mut short = vec![Complex64::new(0.0, 0.0); 15];
    assert!(plan.forward_in_place(&mut short).is_err());
}
