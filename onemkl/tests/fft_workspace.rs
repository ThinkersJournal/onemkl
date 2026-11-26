// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "fft")]

//! `DFTI_WORKSPACE` on complex and real plans: the mode is applied at plan
//! time, read back with `workspace()`, and (measured on oneMKL 2026.2, see
//! `tools/size-probes/fft_flags.c`) never changes the result.

use approx::assert_abs_diff_eq;
use num_complex::Complex;
use onemkl::fft::{FftPlan, FftWorkspace, RealFftPlan};

fn data(n: usize) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| Complex::new((i as f64 * 0.7).sin(), (i as f64 * 0.2).cos()))
        .collect()
}

#[test]
fn complex_workspace_modes_read_back_and_match_default() {
    let dims = [8usize, 6usize];
    for mode in [FftWorkspace::Allow, FftWorkspace::Avoid] {
        let mut p = FftPlan::<f64>::complex_nd_with_workspace(&dims, mode).unwrap();
        assert_eq!(p.workspace().unwrap(), mode);
        let mut plain = FftPlan::<f64>::complex_nd(&dims).unwrap();
        let mut a = data(48);
        let mut b = a.clone();
        p.forward_in_place(&mut a).unwrap();
        plain.forward_in_place(&mut b).unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert_abs_diff_eq!(x.re, y.re, epsilon = 1e-12);
            assert_abs_diff_eq!(x.im, y.im, epsilon = 1e-12);
        }
    }
}

#[test]
fn real_workspace_modes_read_back_and_match_default() {
    let dims = [4usize, 10usize];
    for mode in [FftWorkspace::Allow, FftWorkspace::Avoid] {
        let mut p = RealFftPlan::<f64>::real_nd_with_workspace(&dims, mode).unwrap();
        assert_eq!(p.workspace().unwrap(), mode);
        let mut plain = RealFftPlan::<f64>::real_nd(&dims).unwrap();
        let x: Vec<f64> = (0..40).map(|i| (i as f64 * 0.3).sin()).collect();
        let mut a = vec![Complex::new(0.0, 0.0); p.complex_len()];
        let mut b = a.clone();
        p.forward(&x, &mut a).unwrap();
        plain.forward(&x, &mut b).unwrap();
        for (u, v) in a.iter().zip(&b) {
            assert_abs_diff_eq!(u.re, v.re, epsilon = 1e-12);
            assert_abs_diff_eq!(u.im, v.im, epsilon = 1e-12);
        }
    }
}
