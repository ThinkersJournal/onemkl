#![cfg(all(feature = "blas", feature = "half"))]

//! Typed `half::f16` GEMM wrappers vs a naive f32 reference. Operands are
//! multiples of 0.25 below 8 (exact in f16, products and sums exact in f32),
//! so `gemm_f16_f32` must match the reference exactly.

use half::f16;
use onemkl::blas::mixed_precision::f16 as typed;
use onemkl::{Layout, Transpose};

const M: usize = 3;
const N: usize = 4;
const K: usize = 5;

fn data(len: usize, seed: usize) -> Vec<f16> {
    (0..len).map(|v| f16::from_f32(((v * 7 + seed) % 29) as f32 * 0.25 - 3.0)).collect()
}

// Row-major, no transpose: C = alpha*A*B + beta*C.
fn reference(a: &[f16], b: &[f16], c0: &[f32], alpha: f32, beta: f32, transpose_b: bool) -> Vec<f32> {
    let mut c = c0.to_vec();
    for i in 0..M {
        for j in 0..N {
            let mut s = 0.0f32;
            for p in 0..K {
                let bv = if transpose_b { b[j * K + p] } else { b[p * N + j] };
                s += a[i * K + p].to_f32() * bv.to_f32();
            }
            c[i * N + j] = alpha * s + beta * c0[i * N + j];
        }
    }
    c
}

fn run_f32acc() -> Vec<f32> {
    let a = data(M * K, 1);
    let b = data(K * N, 5);
    let c0: Vec<f32> = (0..M * N).map(|v| v as f32).collect();
    let mut c = c0.clone();
    typed::gemm_f16_f32(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans, M, N, K, 1.5, &a, K, &b, N, -0.5, &mut c, N,
    )
    .unwrap();
    assert_eq!(c, reference(&a, &b, &c0, 1.5, -0.5, false));
    c
}

#[test]
fn gemm_f16_f32_matches_reference() {
    run_f32acc();
}

#[test]
fn reference_distinguishes_a_wrong_op() {
    // Sabotage sibling: the reference with B transposed must differ, so the
    // test above cannot pass for an operand-layout mix-up.
    let a = data(M * K, 1);
    let b = data(K * N, 5);
    let c0 = vec![0.0f32; M * N];
    assert_ne!(reference(&a, &b, &c0, 1.0, 0.0, false), reference(&a, &b, &c0, 1.0, 0.0, true));
}

#[test]
fn hgemm_matches_reference() {
    let a = data(M * K, 1);
    let b = data(K * N, 5);
    let c0: Vec<f32> = (0..M * N).map(|v| v as f32).collect();
    let mut c: Vec<f16> = c0.iter().map(|&v| f16::from_f32(v)).collect();
    typed::hgemm(
        Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans, M, N, K,
        f16::from_f32(1.5), &a, K, &b, N, f16::from_f32(-0.5), &mut c, N,
    )
    .unwrap();
    let want = reference(&a, &b, &c0, 1.5, -0.5, false);
    for (g, w) in c.iter().zip(&want) {
        // f16 result: allow its rounding (rel 2^-10) plus accumulation slack.
        assert!((g.to_f32() - w).abs() <= 0.02 * w.abs().max(1.0), "{g} vs {w}");
    }
}
