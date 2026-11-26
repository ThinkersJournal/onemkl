// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "blas")]

//! `sgemm` (f32) through `blas::level3::gemm`: both layouts, transposes, beta,
//! checked against a naive reference. The older tests cover f64 only.

use onemkl::blas::level3::gemm;
use onemkl::matrix::{MatrixMut, MatrixRef};
use onemkl::{Layout, Transpose};

fn at(buf: &[f32], layout: Layout, rows: usize, cols: usize, i: usize, j: usize) -> f32 {
    match layout {
        Layout::RowMajor => buf[i * cols + j],
        Layout::ColMajor => buf[j * rows + i],
    }
}

fn run(layout: Layout, ta: Transpose, tb: Transpose) {
    let (m, n, k) = (3usize, 4usize, 5usize);
    let (ar, ac) = if ta == Transpose::NoTrans {
        (m, k)
    } else {
        (k, m)
    };
    let (br, bc) = if tb == Transpose::NoTrans {
        (k, n)
    } else {
        (n, k)
    };
    let a: Vec<f32> = (0..ar * ac).map(|v| (v as f32) * 0.5 - 3.0).collect();
    let b: Vec<f32> = (0..br * bc).map(|v| 2.0 - (v as f32) * 0.25).collect();
    let c0: Vec<f32> = (0..m * n).map(|v| v as f32).collect();
    let (alpha, beta) = (1.5_f32, -0.5_f32);
    let mut c = c0.clone();
    gemm(
        ta,
        tb,
        alpha,
        &MatrixRef::new(&a, ar, ac, layout).unwrap(),
        &MatrixRef::new(&b, br, bc, layout).unwrap(),
        beta,
        &mut MatrixMut::new(&mut c, m, n, layout).unwrap(),
    )
    .unwrap();
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f32;
            for l in 0..k {
                let av = if ta == Transpose::NoTrans {
                    at(&a, layout, ar, ac, i, l)
                } else {
                    at(&a, layout, ar, ac, l, i)
                };
                let bv = if tb == Transpose::NoTrans {
                    at(&b, layout, br, bc, l, j)
                } else {
                    at(&b, layout, br, bc, j, l)
                };
                s += av * bv;
            }
            let want = alpha * s + beta * at(&c0, layout, m, n, i, j);
            let got = at(&c, layout, m, n, i, j);
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "{layout:?} {ta:?} {tb:?} ({i},{j}): {got} vs {want}"
            );
        }
    }
}

#[test]
fn sgemm_all_layouts_and_transposes() {
    for layout in [Layout::RowMajor, Layout::ColMajor] {
        for ta in [Transpose::NoTrans, Transpose::Trans] {
            for tb in [Transpose::NoTrans, Transpose::Trans] {
                run(layout, ta, tb);
            }
        }
    }
}
