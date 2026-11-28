// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "data-fitting")]

//! `InterpolateCallbacks` (the `le_cb`, `re_cb` and `i_cb` of
//! `df?InterpolateEx1D`). Behaviour measured on oneMKL 2026.2 with
//! `tools/size-probes/df_interp_cb.c`: each call-back is called once with the
//! sites of its region (left of the first breakpoint, inside, right of the
//! last), cells are 0-based, results land at the sites' positions also for
//! unsorted sites, a negative status is the routine's status, a positive one
//! is returned with the result kept.

use std::cell::RefCell;

use onemkl::Error;
use onemkl::data_fitting::{InterpolateCallbacks, InterpolationPieces, LinearSpline1d};

fn spline() -> LinearSpline1d<f64> {
    let x: Vec<f64> = (0..6).map(f64::from).collect();
    LinearSpline1d::new(x.clone(), x).unwrap() // y = x
}

fn tag(base: f64) -> impl FnMut(InterpolationPieces<'_, f64>) -> i32 {
    move |p| {
        for (i, s) in p.sites.iter().enumerate() {
            p.out[i] = base + *s;
        }
        0
    }
}

#[test]
fn each_region_goes_to_its_call_back() {
    let (mut l, mut r, mut i) = (tag(100.0), tag(200.0), tag(300.0));
    let s = spline();
    let sites = [-1.0, 0.5, 2.5, 5.0, 7.0];
    let out = s
        .interpolate_with_callbacks(
            &sites,
            InterpolateCallbacks::new()
                .left(&mut l)
                .right(&mut r)
                .interior(&mut i),
        )
        .unwrap();
    assert_eq!(out, vec![99.0, 300.5, 302.5, 305.0, 207.0]);
}

#[test]
fn none_uses_the_library_and_unsorted_sites_keep_their_positions() {
    let mut l = tag(100.0);
    let s = spline();
    let sites = [2.5, -1.0, 7.0, 0.5, 5.0];
    let out = s
        .interpolate_with_callbacks(&sites, InterpolateCallbacks::new().left(&mut l))
        .unwrap();
    assert_eq!(out, vec![2.5, 99.0, 7.0, 0.5, 5.0]);
}

#[test]
fn call_backs_get_sites_in_order_and_zero_based_cells() {
    let seen = RefCell::new(Vec::new());
    let mut i = |p: InterpolationPieces<'_, f64>| {
        seen.borrow_mut().push((p.sites.to_vec(), p.cells.to_vec()));
        p.out.copy_from_slice(p.sites);
        0
    };
    spline()
        .interpolate_with_callbacks(
            &[2.5, 0.5, 5.0],
            InterpolateCallbacks::new().interior(&mut i),
        )
        .unwrap();
    assert_eq!(
        *seen.borrow(),
        vec![(vec![2.5, 0.5, 5.0], vec![2, 0, 4])] // measured: cell of 5.0 is the last, 4
    );
}

#[test]
fn negative_status_is_the_error() {
    let mut l = |_: InterpolationPieces<'_, f64>| -5;
    let r =
        spline().interpolate_with_callbacks(&[-1.0, 1.0], InterpolateCallbacks::new().left(&mut l));
    assert!(matches!(r, Err(Error::VslStatus(-5))));
}

#[test]
fn positive_status_is_a_warning_and_the_result_is_kept() {
    let mut i = |p: InterpolationPieces<'_, f64>| {
        p.out.fill(42.0);
        3
    };
    let out = spline()
        .interpolate_with_callbacks(&[1.0, 2.0], InterpolateCallbacks::new().interior(&mut i))
        .unwrap();
    assert_eq!(out, vec![42.0, 42.0]);
}

#[test]
fn panic_in_a_call_back_is_resumed() {
    let r = std::panic::catch_unwind(|| {
        let mut i = |_: InterpolationPieces<'_, f64>| -> i32 { panic!("boom") };
        let _ = spline()
            .interpolate_with_callbacks(&[1.0], InterpolateCallbacks::new().interior(&mut i));
    });
    assert!(r.is_err());
}

// Derivative orders: `CubicSpline1d::interpolate_derivatives_with_callbacks`.
// Measured with `tools/size-probes/df_interp_cb.c` (two orders): oneMKL puts
// each call-back's block at its first site's position and does not rearrange
// it; the library's own sites are site-major (`r[i * orders + k]`, orders
// ascending), so a call-back must write that layout to match.
mod derivative_orders {
    use onemkl::data_fitting::{CubicSpline1d, InterpolateCallbacks, InterpolationPieces};

    fn cubic() -> CubicSpline1d<f64> {
        let x: Vec<f64> = (0..6).map(f64::from).collect();
        CubicSpline1d::natural(x.clone(), x).unwrap() // y = x: value s, 1st derivative 1
    }

    #[test]
    fn callback_blocks_are_site_major_and_mix_with_the_library() {
        // Interior sites by the call-back (value: 300 + s, derivative: 1300 + s);
        // the sites outside the knot range by the library.
        let mut i = |p: InterpolationPieces<'_, f64>| {
            assert_eq!(p.orders, 2);
            assert_eq!(p.out.len(), 2 * p.sites.len());
            for (k, s) in p.sites.iter().enumerate() {
                p.out[k * 2] = 300.0 + *s;
                p.out[k * 2 + 1] = 1300.0 + *s;
            }
            0
        };
        let sites = [-1.0, 0.5, 2.5, 7.0];
        // Requested out of order: results follow the request order.
        let r = cubic()
            .interpolate_derivatives_with_callbacks(
                &sites,
                &[1, 0],
                InterpolateCallbacks::new().interior(&mut i),
            )
            .unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0][1..3], [1300.5, 1302.5]); // derivative (order 1) at the interior sites
        assert_eq!(r[1][1..3], [300.5, 302.5]); // value (order 0) at the interior sites
        // Library-served sites: y = x extrapolated.
        for (idx, want_value) in [(0usize, -1.0f64), (3, 7.0)] {
            assert!((r[1][idx] - want_value).abs() < 1e-9, "{r:?}");
            assert!((r[0][idx] - 1.0).abs() < 1e-9, "{r:?}");
        }
    }

    #[test]
    fn invalid_orders_are_rejected_before_the_call() {
        let c = cubic();
        for orders in [&[4usize][..], &[1, 1][..]] {
            let r = c.interpolate_derivatives_with_callbacks(
                &[0.5],
                orders,
                InterpolateCallbacks::new(),
            );
            assert!(r.is_err(), "{orders:?}");
        }
        let empty = c
            .interpolate_derivatives_with_callbacks(&[0.5], &[], InterpolateCallbacks::new())
            .unwrap();
        assert!(empty.is_empty());
    }
}
