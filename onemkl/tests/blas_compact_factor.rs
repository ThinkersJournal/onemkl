#![cfg(feature = "blas")]

//! Compact TRSM / POTRF / GETRFNP against hand-computed results.

use approx::assert_abs_diff_eq;
use onemkl::blas::compact::{
    getrfnp, gepack, geunpack, get_size, potrf, trsm, CompactFormat,
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
