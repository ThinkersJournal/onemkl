// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "rng")]

//! Verify summary statistics tasks.

use approx::assert_abs_diff_eq;

use onemkl::rng::summary_stats::SummaryStats;

#[test]
fn mean_of_arithmetic_sequence() {
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let mut ss = SummaryStats::<f64>::new(&data, 1, 5).unwrap();
    let mean = ss.mean().unwrap();
    assert_eq!(mean.len(), 1);
    assert_abs_diff_eq!(mean[0], 3.0, epsilon = 1e-12);
}

#[test]
fn variance_of_known_sample() {
    // Sample variance of [1, 2, 3, 4, 5] is 10 / (5 - 1) = 2.5.
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let mut ss = SummaryStats::<f64>::new(&data, 1, 5).unwrap();
    let var = ss.variance().unwrap();
    assert_eq!(var.len(), 1);
    assert_abs_diff_eq!(var[0], 2.5, epsilon = 1e-12);
}

#[test]
fn min_max_sum() {
    let data = [3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
    let mut ss = SummaryStats::<f64>::new(&data, 1, 7).unwrap();
    let mn = ss.min().unwrap();
    let mx = ss.max().unwrap();
    let sm = ss.sum().unwrap();
    assert_abs_diff_eq!(mn[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(mx[0], 9.0, epsilon = 1e-12);
    assert_abs_diff_eq!(sm[0], 25.0, epsilon = 1e-12);
}

#[test]
fn multivariate_means() {
    // 2 variables × 4 observations, row-major:
    //   var 0: [1, 2, 3, 4]  → mean 2.5
    //   var 1: [10, 20, 30, 40] → mean 25.0
    let data = [1.0_f64, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let mut ss = SummaryStats::<f64>::new(&data, 2, 4).unwrap();
    let mean = ss.mean().unwrap();
    assert_eq!(mean.len(), 2);
    assert_abs_diff_eq!(mean[0], 2.5, epsilon = 1e-12);
    assert_abs_diff_eq!(mean[1], 25.0, epsilon = 1e-12);
}

#[test]
fn f32_variant() {
    let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let mut ss = SummaryStats::<f32>::new(&data, 1, 5).unwrap();
    let mean = ss.mean().unwrap();
    assert_abs_diff_eq!(mean[0], 3.0, epsilon = 1e-6);
}

#[test]
fn rejects_undersized_buffer() {
    // Want 2 vars × 3 obs but supply only 5 elements.
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    assert!(SummaryStats::<f64>::new(&data, 2, 3).is_err());
}

#[test]
fn quantiles_at_quartiles() {
    // Sample [1..=11]; expected median = 6 exactly. Q1 / Q3 depend
    // on the interpolation convention; just check they're in band.
    let data: Vec<f64> = (1..=11).map(|x| x as f64).collect();
    let mut ss = SummaryStats::<f64>::new(&data, 1, data.len()).unwrap();
    let q = ss.quantiles(&[0.25, 0.5, 0.75]).unwrap();
    assert_eq!(q.len(), 3);
    assert_abs_diff_eq!(q[1], 6.0, epsilon = 1e-9);
    assert!((q[0] - 3.5).abs() < 0.5);
    assert!((q[2] - 8.5).abs() < 0.5);
}

#[test]
fn quantiles_rejects_empty_orders() {
    let data = [1.0_f64, 2.0, 3.0];
    let mut ss = SummaryStats::<f64>::new(&data, 1, 3).unwrap();
    assert!(ss.quantiles(&[]).is_err());
}

#[test]
fn robust_covariance_tbs_runs() {
    // TBS (Tukey biweight S-estimator) is the algorithm MKL exposes
    // for VSL_SS_ROBUST_COV. Needs a multivariate, sufficiently large
    // sample.
    const N: usize = 200;
    let var1: Vec<f64> = (0..N).map(|x| x as f64).collect();
    let var2: Vec<f64> = (0..N).map(|x| (x as f64) * 2.0).collect();
    let mut data = Vec::with_capacity(2 * N);
    data.extend_from_slice(&var1);
    data.extend_from_slice(&var2);
    let mut ss = SummaryStats::<f64>::new(&data, 2, N).unwrap();
    let (mean, cov) = ss.robust_covariance_tbs(&[]).unwrap();
    assert_eq!(mean.len(), 2);
    assert_eq!(cov.len(), 4);
    // Robust mean of an evenly-spaced sequence sits in its middle.
    assert!(mean[0] > 50.0 && mean[0] < 150.0);
}

#[test]
fn outliers_bacon_flags_extreme() {
    // 5 normal points + 1 huge outlier.
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 1000.0];
    let mut ss = SummaryStats::<f64>::new(&data, 1, data.len()).unwrap();
    let params = SummaryStats::<f64>::bacon_default_params();
    let weights = ss.outliers_bacon(&params).unwrap();
    assert_eq!(weights.len(), data.len());
    assert!(
        weights[5] < 1.0,
        "expected last sample flagged, got weight {}",
        weights[5]
    );
}

#[test]
fn impute_missing_fills_marked_cells() {
    use onemkl::rng::summary_stats::{ImputeOptions, SsScalar, impute_missing};
    const P: usize = 3;
    const N: usize = 40;
    let mut state = 12345_u64;
    let mut data: Vec<f64> = (0..P * N)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect();
    for j in 0..P {
        data[j * N + 3 + j] = <f64 as SsScalar>::MISSING;
    }
    let opts = ImputeOptions::<f64> {
        sets: 2,
        ..Default::default()
    };
    let out = impute_missing(&mut data, P, N, &opts).unwrap();
    assert_eq!(out.missing_count, 3);
    assert_eq!(out.simulated.len(), 6);
    assert_eq!(out.estimates.len(), 2 * 30 * (P + (P * P + P) / 2));
    assert!(
        out.simulated
            .iter()
            .all(|v| v.is_finite() && (-2.0..3.0).contains(v))
    );
    // an ordinary NaN is not the marker
    let mut plain = vec![1.0_f64, f64::NAN, 3.0, 4.0];
    assert!(impute_missing(&mut plain, 1, 4, &opts).is_err());
}

#[test]
fn impute_missing_f32() {
    use onemkl::rng::summary_stats::{ImputeOptions, SsScalar, impute_missing};
    let mut data: Vec<f32> = (0..2 * 30)
        .map(|i| ((i * 37 % 101) as f32) / 101.0)
        .collect();
    data[5] = <f32 as SsScalar>::MISSING;
    let out = impute_missing(&mut data, 2, 30, &ImputeOptions::<f32>::default()).unwrap();
    assert_eq!(out.simulated.len(), 2);
    assert!(out.simulated.iter().all(|v| v.is_finite()));
}

#[test]
fn robust_covariance_tbs_needs_more_than_2p_observations() {
    // oneMKL returns -4002 for p = 3, n = 6 (probe ss5.c); the wrapper refuses first.
    let data: Vec<f64> = (0..18).map(|i| (i * 7 % 11) as f64).collect();
    let mut ss = SummaryStats::<f64>::new(&data, 3, 6).unwrap();
    assert!(ss.robust_covariance_tbs(&[]).is_err());
    let mut state = 99_u64;
    let data7: Vec<f64> = (0..36)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect();
    let mut ss7 = SummaryStats::<f64>::new(&data7, 3, 12).unwrap();
    assert!(
        ss7.robust_covariance_tbs(&[0.25, 0.001, 0.001, 10.0])
            .is_ok()
    );
}

#[test]
fn stream_quantiles_zw_accuracy_and_refusals() {
    let n = 1000_usize;
    let data: Vec<f64> = (0..n).map(|i| ((i * 7919) % n) as f64 / n as f64).collect();
    let mut ss = SummaryStats::<f64>::new(&data, 1, n).unwrap();
    let q = ss.stream_quantiles(&[0.25, 0.5, 0.9], &[0.01]).unwrap();
    for (got, want) in q.iter().zip([0.25, 0.5, 0.9]) {
        assert!((got - want).abs() < 0.02, "{got} vs {want}");
    }
    // oneMKL refuses 0 or 2 parameters and accuracy 0 (probe ss6.c): the wrapper and the library agree
    assert!(ss.stream_quantiles(&[0.5], &[]).is_err());
    assert!(ss.stream_quantiles(&[0.5], &[0.01, 0.01]).is_err());
    assert!(ss.stream_quantiles(&[0.5], &[0.0]).is_err());
}

// ---- with_weights: values measured on oneMKL 2026.2 (tools/size-probes/ss7.c) ----

const WX: [f64; 12] = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 2.0, 2.0, 4.0, 4.0, 8.0, 0.0];
const WW: [f64; 6] = [1.0, 2.0, 3.0, 1.0, 1.0, 2.0];

#[test]
fn weighted_mean_and_variance_match_the_measured_formulas() {
    let mut ss = SummaryStats::<f64>::with_weights(&WX, 2, 6, &WW).unwrap();
    let mean = ss.mean().unwrap();
    assert_abs_diff_eq!(mean[0], 4.3, epsilon = 1e-12);
    assert_abs_diff_eq!(mean[1], 3.0, epsilon = 1e-12);
    // sum(w (x - mean)^2) / (W - sum(w^2) / W), W = 10, sum(w^2) = 20.
    let var = ss.variance().unwrap();
    assert_abs_diff_eq!(var[0], 11.5125, epsilon = 1e-10);
    assert_abs_diff_eq!(var[1], 6.25, epsilon = 1e-10);
}

#[test]
fn weights_change_the_result_vacuity_mutant() {
    // Without the weights the same data has mean 25/6, so the weighted
    // result above is not what an ignored weights argument would give.
    let mut ss = SummaryStats::<f64>::new(&WX, 2, 6).unwrap();
    let mean = ss.mean().unwrap();
    assert_abs_diff_eq!(mean[0], 25.0 / 6.0, epsilon = 1e-12);
    assert!((mean[0] - 4.3).abs() > 0.1);
}

#[test]
fn weighted_f32() {
    let x: Vec<f32> = WX.iter().map(|v| *v as f32).collect();
    let w: Vec<f32> = WW.iter().map(|v| *v as f32).collect();
    let mut ss = SummaryStats::<f32>::with_weights(&x, 2, 6, &w).unwrap();
    assert_abs_diff_eq!(ss.mean().unwrap()[0], 4.3_f32, epsilon = 1e-5);
}

#[test]
fn bad_weights_are_refused_before_onemkl_computes_with_them() {
    // Unguarded, oneMKL computed a mean of 5.8333 from a negative weight and
    // 0 from all-zero weights, both with status 0 (ss7.c).
    use onemkl::Error;
    let neg = [1.0, -2.0, 3.0, 1.0, 1.0, 2.0];
    let zero = [0.0; 6];
    let nan = [1.0, f64::NAN, 1.0, 1.0, 1.0, 1.0];
    let short = [1.0; 5];
    for w in [&neg[..], &zero[..], &nan[..], &short[..]] {
        let r = SummaryStats::<f64>::with_weights(&WX, 2, 6, w);
        assert!(matches!(r, Err(Error::InvalidArgument(_))), "{w:?}");
    }
}

// ---- partial_covariance: values from tools/size-probes/ss9.c on oneMKL 2026.2 ----

mod partial {
    use super::*;
    use onemkl::Error;
    use onemkl::rng::summary_stats::PartialRole::{Conditioning as Z, Target as Y};

    const PX: [f64; 30] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, //
        2.0, 1.0, 4.0, 3.0, 7.0, 5.0, 8.0, 9.0, 12.0, 11.0, //
        5.0, 3.0, 6.0, 8.0, 2.0, 9.0, 4.0, 12.0, 7.0, 10.0,
    ];

    /// `S_YY - S_YZ S_ZZ^-1 S_ZY` for Z = variable 0, computed here.
    fn analytic(z: usize, ys: [usize; 2]) -> [f64; 4] {
        let m = |v: usize| PX[v * 10..v * 10 + 10].iter().sum::<f64>() / 10.0;
        let s = |a: usize, b: usize| {
            (0..10)
                .map(|i| (PX[a * 10 + i] - m(a)) * (PX[b * 10 + i] - m(b)))
                .sum::<f64>()
                / 9.0
        };
        let pc = |a: usize, b: usize| s(a, b) - s(a, z) * s(z, b) / s(z, z);
        let (a, b) = (ys[0], ys[1]);
        [pc(a, a), pc(a, b), pc(b, a), pc(b, b)]
    }

    #[test]
    fn partial_covariance_matches_the_schur_complement() {
        let mut ss = SummaryStats::<f64>::new(&PX, 3, 10).unwrap();
        let r = ss.partial_covariance(&[Z, Y, Y]).unwrap();
        assert_eq!(r.num_targets, 2);
        let want = analytic(0, [1, 2]);
        for (g, w) in r.covariance.iter().zip(want) {
            assert_abs_diff_eq!(*g, w, epsilon = 1e-10);
        }
        // Measured: 1.46532, -1.48956, 7.033; off-diagonal correlation -0.464005.
        assert_abs_diff_eq!(r.covariance[0], 1.46532, epsilon = 1e-5);
        assert_abs_diff_eq!(r.correlation[1], -0.464005, epsilon = 1e-5);
        // Measured quirk: the correlation diagonal is the partial variance.
        assert_abs_diff_eq!(r.correlation[0], r.covariance[0], epsilon = 1e-12);
    }

    #[test]
    fn the_conditioning_variable_matters_vacuity_mutant() {
        let mut ss = SummaryStats::<f64>::new(&PX, 3, 10).unwrap();
        let a = ss.partial_covariance(&[Z, Y, Y]).unwrap();
        let b = ss.partial_covariance(&[Y, Z, Y]).unwrap();
        // Measured: conditioning on variable 1 gives 0.932785 for the first target.
        assert_abs_diff_eq!(b.covariance[0], 0.932785, epsilon = 1e-5);
        assert!((a.covariance[0] - b.covariance[0]).abs() > 0.1);
    }

    #[test]
    fn bad_role_arrays_are_refused_before_onemkl_rejects_them() {
        // oneMKL returned -4073 for "no targets" (ss9.c).
        let mut ss = SummaryStats::<f64>::new(&PX, 3, 10).unwrap();
        for roles in [&[Z, Z, Z][..], &[Y, Y, Y][..], &[Z, Y][..]] {
            assert!(matches!(
                ss.partial_covariance(roles),
                Err(Error::InvalidArgument(_))
            ));
        }
    }
}
