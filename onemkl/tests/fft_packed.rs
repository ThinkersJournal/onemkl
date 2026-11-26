#![cfg(feature = "fft")]

//! 1-D real FFT in the packed storage formats (CCS, PACK, PERM),
//! checked against a naive O(n^2) DFT laid out by hand per the oneMKL
//! "Packed Formats" description, plus round-tripping.

use approx::assert_abs_diff_eq;
use onemkl::fft::{PackedRealFftPlan, RealPackedFormat};

/// Naive forward DFT of a real signal: X[k] = sum x[j] e^{-2 pi i jk/n}.
fn dft(x: &[f64]) -> Vec<(f64, f64)> {
    let n = x.len();
    (0..=n / 2)
        .map(|k| {
            let (mut re, mut im) = (0.0, 0.0);
            for (j, &v) in x.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
                re += v * a.cos();
                im += v * a.sin();
            }
            (re, im)
        })
        .collect()
}

/// Expected packed layout of the spectrum `s` (k = 0..=n/2) of an n-point
/// real signal.
fn expected(fmt: RealPackedFormat, n: usize, s: &[(f64, f64)]) -> Vec<f64> {
    let mut v = Vec::new();
    match fmt {
        RealPackedFormat::Ccs => {
            for &(re, im) in s {
                v.push(re);
                v.push(im);
            }
        }
        RealPackedFormat::Pack => {
            v.push(s[0].0);
            for (k, &(re, im)) in s.iter().enumerate().skip(1) {
                v.push(re);
                if !(n % 2 == 0 && k == n / 2) {
                    v.push(im);
                }
            }
        }
        RealPackedFormat::Perm => {
            v.push(s[0].0);
            if n % 2 == 0 {
                v.push(s[n / 2].0);
            }
            for (k, &(re, im)) in s.iter().enumerate().skip(1) {
                if n % 2 == 0 && k == n / 2 {
                    continue;
                }
                v.push(re);
                v.push(im);
            }
        }
    }
    v
}

fn signal(n: usize) -> Vec<f64> {
    (0..n).map(|j| ((j * 7 + 3) % 11) as f64 - 4.5).collect()
}

#[test]
fn packed_forward_matches_naive_dft_all_formats() {
    for fmt in [
        RealPackedFormat::Ccs,
        RealPackedFormat::Pack,
        RealPackedFormat::Perm,
    ] {
        for n in [4usize, 8, 9, 15] {
            let x = signal(n);
            let want = expected(fmt, n, &dft(&x));
            let mut plan = PackedRealFftPlan::<f64>::new(n, fmt).unwrap();
            assert_eq!(plan.buffer_len(), want.len(), "{fmt:?} n={n}");
            let mut buf = vec![0.0; plan.buffer_len()];
            buf[..n].copy_from_slice(&x);
            plan.forward_in_place(&mut buf).unwrap();
            for (i, (g, w)) in buf.iter().zip(&want).enumerate() {
                assert_abs_diff_eq!(*g, *w, epsilon = 1e-9);
                let _ = i;
            }
        }
    }
}

#[test]
fn packed_round_trip_scaled() {
    for fmt in [
        RealPackedFormat::Ccs,
        RealPackedFormat::Pack,
        RealPackedFormat::Perm,
    ] {
        let n = 12;
        let x = signal(n);
        let mut plan = PackedRealFftPlan::<f64>::with_scales(n, fmt, 1.0, 1.0 / n as f64).unwrap();
        let mut buf = vec![0.0; plan.buffer_len()];
        buf[..n].copy_from_slice(&x);
        plan.forward_in_place(&mut buf).unwrap();
        plan.backward_in_place(&mut buf).unwrap();
        for (g, w) in buf.iter().zip(&x) {
            assert_abs_diff_eq!(*g, *w, epsilon = 1e-12);
        }
    }
}

#[test]
fn packed_rejects_short_buffer_and_zero_length() {
    let mut plan = PackedRealFftPlan::<f32>::new(8, RealPackedFormat::Ccs).unwrap();
    let mut short = vec![0.0f32; 8]; // CCS needs 10
    assert!(plan.forward_in_place(&mut short).is_err());
    assert!(PackedRealFftPlan::<f32>::new(0, RealPackedFormat::Pack).is_err());
}
