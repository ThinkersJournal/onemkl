#![cfg(feature = "vm")]

//! f32 (`vs*`) paths of the VM wrappers, plus `fmax` / `fmin` (both types).
//! The older `vm.rs` covers f64 only.

use onemkl::vm;

const X: [f32; 5] = [-2.0, -0.5, 0.0, 0.75, 3.0];
const POS: [f32; 5] = [0.25, 0.5, 1.0, 2.0, 10.0];

fn close(got: &[f32], want: impl Fn(usize) -> f32, rel: f32, what: &str) {
    for (i, &g) in got.iter().enumerate() {
        let w = want(i);
        assert!((g - w).abs() <= rel * w.abs().max(1e-30) + 1e-30, "{what}[{i}]: got {g}, want {w}");
    }
}

#[test]
fn unary_f32_matches_std_at_high_accuracy() {
    let mut y = [0.0_f32; 5];
    vm::tanh(&X, &mut y).unwrap();
    close(&y, |i| X[i].tanh(), 1e-6, "tanh");
    vm::exp(&X, &mut y).unwrap();
    close(&y, |i| X[i].exp(), 1e-6, "exp");
    vm::sin(&X, &mut y).unwrap();
    close(&y, |i| X[i].sin(), 1e-6, "sin");
    vm::cos(&X, &mut y).unwrap();
    close(&y, |i| X[i].cos(), 1e-6, "cos");
    vm::ln(&POS, &mut y).unwrap();
    close(&y, |i| POS[i].ln(), 1e-6, "ln");
    vm::sqrt(&POS, &mut y).unwrap();
    close(&y, |i| POS[i].sqrt(), 1e-6, "sqrt");
}

#[test]
fn binary_f32_matches_std() {
    let mut z = [0.0_f32; 5];
    vm::add(&X, &POS, &mut z).unwrap();
    close(&z, |i| X[i] + POS[i], 1e-6, "add");
    vm::sub(&X, &POS, &mut z).unwrap();
    close(&z, |i| X[i] - POS[i], 1e-6, "sub");
    vm::mul(&X, &POS, &mut z).unwrap();
    close(&z, |i| X[i] * POS[i], 1e-6, "mul");
    vm::div(&X, &POS, &mut z).unwrap();
    close(&z, |i| X[i] / POS[i], 1e-6, "div");
}

#[test]
fn length_mismatch_is_rejected() {
    let mut y = [0.0_f32; 4];
    assert!(vm::exp(&X, &mut y).is_err());
}

#[test]
fn fmax_fmin_f32_and_f64() {
    let a = [1.0_f32, -2.0, 5.0, 0.0];
    let b = [3.0_f32, -4.0, 5.0, -1.0];
    let mut z = [0.0_f32; 4];
    vm::fmax(&a, &b, &mut z).unwrap();
    assert_eq!(z, [3.0, -2.0, 5.0, 0.0]);
    vm::fmin(&a, &b, &mut z).unwrap();
    assert_eq!(z, [1.0, -4.0, 5.0, -1.0]);

    let a = [1.0_f64, -2.0, 5.0, 0.0];
    let b = [3.0_f64, -4.0, 5.0, -1.0];
    let mut z = [0.0_f64; 4];
    vm::fmax(&a, &b, &mut z).unwrap();
    assert_eq!(z, [3.0, -2.0, 5.0, 0.0]);
    vm::fmin(&a, &b, &mut z).unwrap();
    assert_eq!(z, [1.0, -4.0, 5.0, -1.0]);
}

#[test]
fn fmax_fmin_ignore_a_nan_operand_like_c_fmax() {
    // Measured on oneMKL 2026.2.1 (f64): a single NaN operand is ignored,
    // NaN only when both are NaN. Signed zeros were not measured.
    let a = [f64::NAN, 1.0, f64::NAN];
    let b = [1.0_f64, f64::NAN, f64::NAN];
    let mut z = [0.0_f64; 3];
    vm::fmax(&a, &b, &mut z).unwrap();
    assert_eq!(&z[..2], &[1.0, 1.0]);
    assert!(z[2].is_nan());
    vm::fmin(&a, &b, &mut z).unwrap();
    assert_eq!(&z[..2], &[1.0, 1.0]);
    assert!(z[2].is_nan());
}
