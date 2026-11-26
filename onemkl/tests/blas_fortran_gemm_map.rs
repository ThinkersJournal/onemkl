#![cfg(feature = "blas")]

//! The mapping from a Fortran-style `?gemm_` call (column-major, `transa` /
//! `transb` as `'N'|'T'|'C'`, `m n k`, `lda ldb ldc`) to onemkl, as a caller
//! such as Fuel's `sgemm_/dgemm_/hgemm_` wrappers needs it.
//!
//! The helpers below ARE the mapping; each test checks it against a plain
//! reference loop on asymmetric non-square data with leading dimensions larger
//! than the row counts, for both transposes, and checks the padding rows of C
//! are not written. `mutants_are_caught` runs deliberately wrong mappings on
//! the same data and requires each to differ, so a pass is not vacuous.

use onemkl::blas::level3::gemm;
use onemkl::blas::mixed_precision::hgemm;
use onemkl::matrix::{MatrixMut, MatrixRef};
use onemkl::{Layout, Transpose};

#[derive(Clone, Copy, PartialEq)]
enum Mutant {
    None,
    SwapTransA,
    RowMajor,
    LdaIsRows,
}

fn tr(c: u8) -> Transpose {
    match c.to_ascii_uppercase() {
        b'N' => Transpose::NoTrans,
        b'T' => Transpose::Trans,
        b'C' => Transpose::ConjTrans,
        other => panic!("bad transpose char {other}"),
    }
}

/// Stored shape of an operand whose op() is `rows x cols`.
fn stored(t: u8, rows: usize, cols: usize) -> (usize, usize) {
    if t.to_ascii_uppercase() == b'N' {
        (rows, cols)
    } else {
        (cols, rows)
    }
}

macro_rules! fortran_gemm_fn {
    ($name:ident, $t:ty) => {
        /// `?gemm_(transa, transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc)`.
        #[allow(clippy::too_many_arguments)]
        fn $name(
            mutant: Mutant,
            ta: u8,
            tb: u8,
            m: usize,
            n: usize,
            k: usize,
            alpha: $t,
            a: &[$t],
            lda: usize,
            b: &[$t],
            ldb: usize,
            beta: $t,
            c: &mut [$t],
            ldc: usize,
        ) {
            let (ar, ac) = stored(ta, m, k);
            let (br, bc) = stored(tb, k, n);
            if mutant == Mutant::RowMajor {
                // Wrong: describe the same buffers as row-major (same ld).
                let a_r = MatrixRef::with_lda(a, ac, ar, Layout::RowMajor, lda.max(ar)).unwrap();
                let b_r = MatrixRef::with_lda(b, bc, br, Layout::RowMajor, ldb).unwrap();
                let mut c_r = MatrixMut::with_lda(c, n, m, Layout::RowMajor, ldc).unwrap();
                gemm(tr(ta), tr(tb), alpha, &a_r, &b_r, beta, &mut c_r).unwrap();
                return;
            }
            let lda_eff = if mutant == Mutant::LdaIsRows { ar } else { lda };
            let a_ref = MatrixRef::with_lda(a, ar, ac, Layout::ColMajor, lda_eff).unwrap();
            let b_ref = MatrixRef::with_lda(b, br, bc, Layout::ColMajor, ldb).unwrap();
            let mut c_ref = MatrixMut::with_lda(c, m, n, Layout::ColMajor, ldc).unwrap();
            let ta_m = if mutant == Mutant::SwapTransA {
                if ta.to_ascii_uppercase() == b'N' {
                    Transpose::Trans
                } else {
                    Transpose::NoTrans
                }
            } else {
                tr(ta)
            };
            gemm(ta_m, tr(tb), alpha, &a_ref, &b_ref, beta, &mut c_ref).unwrap();
        }
    };
}
fortran_gemm_fn!(sgemm_map, f32);
fortran_gemm_fn!(dgemm_map, f64);

/// Reference: C(i,j) = alpha * sum_l opA(i,l) * opB(l,j) + beta * C(i,j), column-major with ld.
macro_rules! ref_gemm_fn {
    ($name:ident, $t:ty) => {
        #[allow(clippy::too_many_arguments)]
        fn $name(
            ta: u8,
            tb: u8,
            m: usize,
            n: usize,
            k: usize,
            alpha: $t,
            a: &[$t],
            lda: usize,
            b: &[$t],
            ldb: usize,
            beta: $t,
            c: &mut [$t],
            ldc: usize,
        ) {
            let ea = |i: usize, l: usize| {
                if ta.to_ascii_uppercase() == b'N' {
                    a[l * lda + i]
                } else {
                    a[i * lda + l]
                }
            };
            let eb = |l: usize, j: usize| {
                if tb.to_ascii_uppercase() == b'N' {
                    b[j * ldb + l]
                } else {
                    b[l * ldb + j]
                }
            };
            for j in 0..n {
                for i in 0..m {
                    let mut s: $t = 0.0;
                    for l in 0..k {
                        s += ea(i, l) * eb(l, j);
                    }
                    c[j * ldc + i] = alpha * s + beta * c[j * ldc + i];
                }
            }
        }
    };
}
ref_gemm_fn!(ref_s, f32);
ref_gemm_fn!(ref_d, f64);

const PAD: f64 = 777.0;
const M: usize = 3;
const N: usize = 4;
const K: usize = 5;
const PAD_ROWS: usize = 2;

fn fill_f64(len: usize, seed: f64) -> Vec<f64> {
    (0..len)
        .map(|i| ((i as f64) * 0.37 + seed).sin() * 2.0 + (i % 3) as f64 * 0.25)
        .collect()
}

macro_rules! map_test {
    ($test:ident, $t:ty, $mapf:ident, $reff:ident, $tol:expr) => {
        #[test]
        fn $test() {
            for ta in [b'N', b'T', b'C'] {
                for tb in [b'N', b'T', b'C'] {
                    let (ar, ac) = stored(ta, M, K);
                    let (br, bc) = stored(tb, K, N);
                    let (lda, ldb, ldc) = (ar + PAD_ROWS, br + 1, M + 3);
                    let a: Vec<$t> = fill_f64(lda * ac, 1.0)
                        .into_iter()
                        .map(|v| v as $t)
                        .collect();
                    let b: Vec<$t> = fill_f64(ldb * bc, 2.0)
                        .into_iter()
                        .map(|v| v as $t)
                        .collect();
                    let mut c0: Vec<$t> = fill_f64(ldc * N, 3.0)
                        .into_iter()
                        .map(|v| v as $t)
                        .collect();
                    for j in 0..N {
                        for i in M..ldc {
                            c0[j * ldc + i] = PAD as $t;
                        }
                    }
                    let (alpha, beta): ($t, $t) = (1.5, -0.5);
                    let mut want = c0.clone();
                    $reff(
                        ta, tb, M, N, K, alpha, &a, lda, &b, ldb, beta, &mut want, ldc,
                    );
                    let mut got = c0.clone();
                    $mapf(
                        Mutant::None,
                        ta,
                        tb,
                        M,
                        N,
                        K,
                        alpha,
                        &a,
                        lda,
                        &b,
                        ldb,
                        beta,
                        &mut got,
                        ldc,
                    );
                    for idx in 0..got.len() {
                        assert!(
                            (got[idx] - want[idx]).abs() <= $tol * want[idx].abs().max(1.0),
                            "{}{} elem {idx}: got {} want {}",
                            ta as char,
                            tb as char,
                            got[idx],
                            want[idx]
                        );
                    }
                    // padding rows of C untouched
                    for j in 0..N {
                        for i in M..ldc {
                            assert_eq!(got[j * ldc + i], PAD as $t, "padding of C overwritten");
                        }
                    }
                }
            }
        }
    };
}
map_test!(sgemm_fortran_mapping, f32, sgemm_map, ref_s, 1e-4);
map_test!(dgemm_fortran_mapping, f64, dgemm_map, ref_d, 1e-12);

#[test]
fn mutants_are_caught() {
    // 'N','T' on A, 'N' on B, ld > rows: each wrong mapping must change C.
    let (ta, tb) = (b'N', b'T');
    let (ar, ac) = stored(ta, M, K);
    let (br, bc) = stored(tb, K, N);
    let (lda, ldb, ldc) = (ar + PAD_ROWS, br + 1, M + 3);
    let a = fill_f64(lda * ac, 1.0);
    let b = fill_f64(ldb * bc, 2.0);
    let c0 = fill_f64(ldc * N, 3.0);
    let mut want = c0.clone();
    ref_d(ta, tb, M, N, K, 1.5, &a, lda, &b, ldb, -0.5, &mut want, ldc);
    for (name, mutant) in [
        ("swap transa", Mutant::SwapTransA),
        ("row-major layout", Mutant::RowMajor),
        ("lda = rows", Mutant::LdaIsRows),
    ] {
        let mut got = c0.clone();
        // Mutants may legitimately be rejected by validation; a panic counts as caught.
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            dgemm_map(
                mutant, ta, tb, M, N, K, 1.5, &a, lda, &b, ldb, -0.5, &mut got, ldc,
            );
            got.clone()
        }));
        let differs = match r {
            Ok(g) => g.iter().zip(&want).any(|(x, y)| (x - y).abs() > 1e-9),
            Err(_) => true,
        };
        assert!(
            differs,
            "mutant `{name}` was NOT caught: data does not discriminate it"
        );
    }
}

// ---- half precision: raw IEEE binary16 bits, alpha/beta as bits ----------

fn f32_to_f16_bits(x: f32) -> u16 {
    // Exact for the multiples of 1/32 used here (normal range, <= 2048).
    if x == 0.0 {
        return 0;
    }
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    assert!(
        exp > 0 && exp < 0x1f,
        "value {x} outside the normal f16 range"
    );
    assert_eq!(
        bits & 0x1fff,
        0,
        "value {x} not exactly representable in f16"
    );
    sign | ((exp as u16) << 10) | ((bits & 0x7f_ffff) >> 13) as u16
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as f32;
    if exp == 0 {
        return sign * mant * 2.0_f32.powi(-24);
    }
    sign * (1.0 + mant / 1024.0) * 2.0_f32.powi(exp - 15)
}

#[test]
fn hgemm_fortran_mapping_nonsquare_ld() {
    for ta in [b'N', b'T'] {
        for tb in [b'N', b'T'] {
            let (ar, ac) = stored(ta, M, K);
            let (br, bc) = stored(tb, K, N);
            let (lda, ldb, ldc) = (ar + PAD_ROWS, br + 1, M + 3);
            // quarter-multiples in [-2, 2]: sums stay exact in f16
            let q = |i: usize, s: usize| (((i * 7 + s * 3) % 17) as f32 - 8.0) * 0.25;
            let a: Vec<u16> = (0..lda * ac).map(|i| f32_to_f16_bits(q(i, 1))).collect();
            let b: Vec<u16> = (0..ldb * bc).map(|i| f32_to_f16_bits(q(i, 2))).collect();
            let mut c: Vec<u16> = (0..ldc * N).map(|i| f32_to_f16_bits(q(i, 3))).collect();
            let c0 = c.clone();
            let (alpha, beta) = (1.5_f32, -0.5_f32);
            hgemm(
                Layout::ColMajor,
                tr(ta),
                tr(tb),
                M,
                N,
                K,
                f32_to_f16_bits(alpha),
                &a,
                lda,
                &b,
                ldb,
                f32_to_f16_bits(beta),
                &mut c,
                ldc,
            )
            .unwrap();
            let af: Vec<f32> = a.iter().map(|&h| f16_bits_to_f32(h)).collect();
            let bf: Vec<f32> = b.iter().map(|&h| f16_bits_to_f32(h)).collect();
            let mut want: Vec<f32> = c0.iter().map(|&h| f16_bits_to_f32(h)).collect();
            ref_s(
                ta, tb, M, N, K, alpha, &af, lda, &bf, ldb, beta, &mut want, ldc,
            );
            for j in 0..N {
                for i in 0..ldc {
                    let g = f16_bits_to_f32(c[j * ldc + i]);
                    let w = want[j * ldc + i];
                    assert!(
                        (g - w).abs() <= 2e-2 * w.abs().max(1.0),
                        "{}{} ({i},{j}): got {g} want {w}",
                        ta as char,
                        tb as char
                    );
                    if i >= M {
                        assert_eq!(c[j * ldc + i], c0[j * ldc + i], "padding of C overwritten");
                    }
                }
            }
        }
    }
}

#[test]
fn hgemm_rejects_a_tight_last_column_buffer() {
    // Fortran sgemm_ only reads (cols-1)*ld + rows elements of a column-major
    // operand; onemkl's hgemm demands the full cols*ld. Pin the stricter rule
    // so a caller with tight slices sees it here, not in production.
    let (m, n, k) = (2, 2, 2);
    let lda = 3;
    let a = vec![0_u16; (k - 1) * lda + m]; // 5 < 2*3
    let b = vec![0_u16; 2 * 2];
    let mut c = vec![0_u16; 2 * 2];
    let r = hgemm(
        Layout::ColMajor,
        Transpose::NoTrans,
        Transpose::NoTrans,
        m,
        n,
        k,
        0,
        &a,
        lda,
        &b,
        2,
        0,
        &mut c,
        2,
    );
    assert!(r.is_err());
}
