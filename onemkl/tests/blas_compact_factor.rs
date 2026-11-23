#![cfg(feature = "blas")]

//! Compact TRSM / POTRF / GETRFNP against hand-computed results.

use approx::assert_abs_diff_eq;
use onemkl::blas::compact::{
    getrfnp, getrinp, gepack, geunpack, get_size, potrf, trsm, CompactFormat,
};
use onemkl::{Diag, Layout, Side, Transpose, UpLo};

const FMT: CompactFormat = CompactFormat::Avx;

fn pack(ms: &[Vec<f64>], ld: usize, sd: usize) -> Vec<f64> {
    let mut buf = vec![0.0; get_size::<f64>(ld, sd, FMT, ms.len()).unwrap()];
    let ptrs: Vec<*const f64> = ms.iter().map(|m| m.as_ptr()).collect();
    unsafe { gepack::<f64>(Layout::ColMajor, ld, sd, &ptrs, ld, &mut buf, ld, FMT).unwrap() };
    buf
}

fn unpack(buf: &[f64], nm: usize, ld: usize, sd: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; ld * sd]; nm];
    let ptrs: Vec<*mut f64> = out.iter_mut().map(|m| m.as_mut_ptr()).collect();
    unsafe { geunpack::<f64>(Layout::ColMajor, ld, sd, &ptrs, ld, buf, ld, FMT).unwrap() };
    out
}

#[test]
fn trsm_solves_two_lower_systems() {
    // Column-major 2x2 lower triangulars: A0 = [[2,0],[1,4]], A1 = [[1,0],[3,2]].
    let a = vec![vec![2.0, 1.0, 0.0, 4.0], vec![1.0, 3.0, 0.0, 2.0]];
    // B = 2x1 column vectors.
    let b = vec![vec![2.0, 9.0], vec![1.0, 7.0]];
    let ap = pack(&a, 2, 2);
    let mut bp = pack(&b, 2, 1);
    trsm::<f64>(
        Layout::ColMajor, Side::Left, UpLo::Lower, Transpose::NoTrans, Diag::NonUnit,
        2, 1, 1.0, &ap, 2, &mut bp, 2, FMT, 2,
    )
    .unwrap();
    let x = unpack(&bp, 2, 2, 1);
    // A0 x = [2,9]: x = [1, 2]; A1 x = [1,7]: x = [1, 2].
    for m in &x {
        assert_abs_diff_eq!(m[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(m[1], 2.0, epsilon = 1e-12);
    }
}

#[test]
fn potrf_factors_spd_and_getrfnp_factors_lu() {
    // SPD: [[4,2],[2,3]] -> L = [[2,0],[1,sqrt(2)]].
    let a = vec![vec![4.0, 2.0, 2.0, 3.0]];
    let mut ap = pack(&a, 2, 2);
    let info = potrf::<f64>(Layout::ColMajor, UpLo::Lower, 2, &mut ap, 2, FMT, 1).unwrap();
    assert_eq!(info, 0);
    let l = unpack(&ap, 1, 2, 2);
    assert_abs_diff_eq!(l[0][0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(l[0][1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(l[0][3], 2.0_f64.sqrt(), epsilon = 1e-12);

    // LU no pivot: [[4,3],[6,3]] -> L=[[1,0],[1.5,1]], U=[[4,3],[0,-1.5]].
    let a = vec![vec![4.0, 6.0, 3.0, 3.0]];
    let mut ap = pack(&a, 2, 2);
    let info = getrfnp::<f64>(Layout::ColMajor, 2, 2, &mut ap, 2, FMT, 1).unwrap();
    assert_eq!(info, 0);
    let lu = unpack(&ap, 1, 2, 2);
    assert_abs_diff_eq!(lu[0][0], 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(lu[0][1], 1.5, epsilon = 1e-12);
    assert_abs_diff_eq!(lu[0][2], 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(lu[0][3], -1.5, epsilon = 1e-12);
}

#[test]
fn undersized_buffers_are_rejected() {
    let mut ap = vec![0.0f64; 1];
    let r = potrf::<f64>(Layout::ColMajor, UpLo::Lower, 2, &mut ap, 2, FMT, 4);
    assert!(r.is_err());
    let r = potrf::<f64>(Layout::ColMajor, UpLo::Lower, 4, &mut ap, 2, FMT, 1);
    assert!(r.is_err(), "ldap < n");
}

#[test]
fn geqrf_r_factor_matches_hand_computed() {
    use onemkl::blas::compact::geqrf;
    // A = [[3,1],[4,2]] (col-major [3,4,1,2]); |R00| = 5, R01 = +-(3*1+4*2)/5 = 2.2.
    let a = vec![vec![3.0, 4.0, 1.0, 2.0]];
    let mut ap = pack(&a, 2, 2);
    let mut tau = vec![0.0; get_size::<f64>(2, 1, FMT, 1).unwrap()];
    let info = geqrf::<f64>(Layout::ColMajor, 2, 2, &mut ap, 2, &mut tau, FMT, 1).unwrap();
    assert_eq!(info, 0);
    let qr = unpack(&ap, 1, 2, 2);
    assert_abs_diff_eq!(qr[0][0].abs(), 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(qr[0][2].abs(), 2.2, epsilon = 1e-12);
    // |R11| = |det A| / 5 = 2/5 = 0.4.
    assert_abs_diff_eq!(qr[0][3].abs(), 0.4, epsilon = 1e-12);
}

#[test]
fn geqrf_rejects_small_tau() {
    use onemkl::blas::compact::geqrf;
    let mut ap = vec![0.0f64; get_size::<f64>(2, 2, FMT, 4).unwrap()];
    let mut tau = vec![0.0f64; 1];
    assert!(geqrf::<f64>(Layout::ColMajor, 2, 2, &mut ap, 2, &mut tau, FMT, 4).is_err());
}

#[test]
fn getrinp_inverts_lu_factored_matrices() {
    // [[4,3],[6,3]] (col-major 4,6,3,3): det -6, inverse [[-1/2, 1/2],[1, -2/3]].
    // [[2,1],[1,2]]: inverse 1/3 * [[2,-1],[-1,2]].
    let a = vec![vec![4.0, 6.0, 3.0, 3.0], vec![2.0, 1.0, 1.0, 2.0]];
    let mut ap = pack(&a, 2, 2);
    assert_eq!(getrfnp::<f64>(Layout::ColMajor, 2, 2, &mut ap, 2, FMT, 2).unwrap(), 0);
    assert_eq!(getrinp::<f64>(Layout::ColMajor, 2, &mut ap, 2, FMT, 2).unwrap(), 0);
    let inv = unpack(&ap, 2, 2, 2);
    let want = [
        [-0.5, 1.0, 0.5, -2.0 / 3.0],
        [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0],
    ];
    for (got, w) in inv.iter().zip(&want) {
        for (g, e) in got.iter().zip(w) {
            assert_abs_diff_eq!(g, e, epsilon = 1e-12);
        }
    }
}
