#![cfg(feature = "fft")]

//! Batched out-of-place 1-D real FFT (real <-> CCE complex) with
//! per-side distances, against a naive O(n^2) DFT (epsilon 1e-9).

use approx::assert_abs_diff_eq;
use num_complex::Complex64;
use onemkl::fft::BatchedRealFftPlan;

fn dft_half(x: &[f64]) -> Vec<Complex64> {
    let n = x.len();
    (0..=n / 2)
        .map(|k| {
            x.iter()
                .enumerate()
                .map(|(j, &v)| {
                    let a = -2.0 * std::f64::consts::PI * ((j * k) % n) as f64 / n as f64;
                    Complex64::new(v * a.cos(), v * a.sin())
                })
                .sum()
        })
        .collect()
}

fn val(t: usize, j: usize) -> f64 {
    ((t * 5 + j * 3) % 7) as f64 - 2.5
}

fn check(n: usize, howmany: usize, rd: usize, cd: usize) {
    let mut plan = BatchedRealFftPlan::<f64>::new(n, howmany, rd, cd).unwrap();
    let h = n / 2 + 1;
    assert_eq!(plan.real_buffer_len(), (howmany - 1) * rd + n);
    assert_eq!(plan.complex_buffer_len(), (howmany - 1) * cd + h);
    let rs = -123.0;
    let cs = Complex64::new(777.0, -777.0);
    let mut input = vec![rs; plan.real_buffer_len()];
    let mut out = vec![cs; plan.complex_buffer_len()];
    for t in 0..howmany {
        for j in 0..n {
            input[t * rd + j] = val(t, j);
        }
    }
    plan.forward(&input, &mut out).unwrap();
    let mut touched = vec![false; out.len()];
    for t in 0..howmany {
        let x: Vec<f64> = (0..n).map(|j| val(t, j)).collect();
        let want = dft_half(&x);
        for k in 0..h {
            let g = out[t * cd + k];
            touched[t * cd + k] = true;
            assert_abs_diff_eq!(g.re, want[k].re, epsilon = 1e-9);
            assert_abs_diff_eq!(g.im, want[k].im, epsilon = 1e-9);
        }
    }
    for (i, b) in out.iter().enumerate() {
        if !touched[i] {
            assert_eq!(*b, cs, "complex padding at {i} was modified");
        }
    }
}

#[test]
fn batched_real_contiguous_matches_naive_dft() {
    check(8, 3, 8, 5); // even n
    check(9, 4, 9, 5); // odd n
}

#[test]
fn batched_real_distances_and_padding() {
    check(8, 3, 11, 7);
    check(9, 3, 12, 9);
}

#[test]
fn batched_real_round_trip_scaled_with_padding() {
    let (n, h, rd, cd) = (10usize, 3usize, 13usize, 8usize);
    let mut plan =
        BatchedRealFftPlan::<f64>::with_scales(n, h, rd, cd, 1.0, 1.0 / n as f64).unwrap();
    let mut input = vec![-123.0; plan.real_buffer_len()];
    for t in 0..h {
        for j in 0..n {
            input[t * rd + j] = val(t, j);
        }
    }
    let mut spec = vec![Complex64::new(0.0, 0.0); plan.complex_buffer_len()];
    plan.forward(&input, &mut spec).unwrap();
    let mut back = vec![-55.0; plan.real_buffer_len()];
    plan.backward(&spec, &mut back).unwrap();
    for t in 0..h {
        for j in 0..n {
            assert_abs_diff_eq!(back[t * rd + j], val(t, j), epsilon = 1e-12);
        }
        if t + 1 < h {
            for j in n..rd {
                assert_eq!(back[t * rd + j], -55.0, "real padding modified");
            }
        }
    }
}

#[test]
fn batched_real_rejects_bad_arguments() {
    assert!(BatchedRealFftPlan::<f64>::new(0, 1, 1, 1).is_err());
    assert!(BatchedRealFftPlan::<f64>::new(8, 0, 8, 5).is_err());
    assert!(BatchedRealFftPlan::<f64>::new(8, 2, 7, 5).is_err()); // real overlap
    assert!(BatchedRealFftPlan::<f64>::new(8, 2, 8, 4).is_err()); // complex overlap (need 5)
    let mut plan = BatchedRealFftPlan::<f64>::new(8, 2, 8, 5).unwrap();
    let short_in = vec![0.0; 15];
    let mut out = vec![Complex64::new(0.0, 0.0); plan.complex_buffer_len()];
    assert!(plan.forward(&short_in, &mut out).is_err());
    let input = vec![0.0; 16];
    let mut short_out = vec![Complex64::new(0.0, 0.0); 9];
    assert!(plan.forward(&input, &mut short_out).is_err());
}
