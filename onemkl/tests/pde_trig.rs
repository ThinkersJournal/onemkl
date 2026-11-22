#![cfg(feature = "pde")]

//! Verify trigonometric transforms (DCT / DST).

use approx::assert_abs_diff_eq;

use onemkl::pde::{TrigTransform, TrigTransformType};

#[test]
fn cosine_round_trip_recovers_input() {
    // Cosine transform expects buffer length n + 1.
    let n: usize = 8;
    let original: Vec<f64> = (0..=n).map(|i| (i as f64) * 0.5).collect();
    let mut buf = original.clone();

    let mut plan = TrigTransform::<f64>::new(TrigTransformType::Cosine, n).unwrap();
    plan.forward(&mut buf).unwrap();
    plan.backward(&mut buf).unwrap();

    for (after, before) in buf.iter().zip(&original) {
        assert_abs_diff_eq!(after, before, epsilon = 1e-10);
    }
}

#[test]
fn sine_round_trip_recovers_input() {
    let n: usize = 8;
    // Sine transform requires zero boundary samples. The interior
    // values are arbitrary.
    let original: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
    let mut buf = original.clone();

    let mut plan = TrigTransform::<f64>::new(TrigTransformType::Sine, n).unwrap();
    plan.forward(&mut buf).unwrap();
    plan.backward(&mut buf).unwrap();

    for (after, before) in buf.iter().zip(&original) {
        assert_abs_diff_eq!(after, before, epsilon = 1e-10);
    }
}

#[test]
fn staggered_cosine_round_trip() {
    let n: usize = 8;
    let original: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();
    let mut buf = original.clone();

    let mut plan =
        TrigTransform::<f64>::new(TrigTransformType::StaggeredCosine, n).unwrap();
    plan.forward(&mut buf).unwrap();
    plan.backward(&mut buf).unwrap();

    for (after, before) in buf.iter().zip(&original) {
        assert_abs_diff_eq!(after, before, epsilon = 1e-10);
    }
}

#[test]
fn buffer_len_matches_type() {
    assert_eq!(TrigTransformType::Cosine.buffer_len(8), 9);
    assert_eq!(TrigTransformType::Sine.buffer_len(8), 9);
    assert_eq!(TrigTransformType::StaggeredCosine.buffer_len(8), 8);
    assert_eq!(TrigTransformType::Staggered2Sine.buffer_len(8), 8);
}

#[test]
fn buffer_len_query_matches_plan() {
    let plan = TrigTransform::<f64>::new(TrigTransformType::Cosine, 8).unwrap();
    assert_eq!(plan.buffer_len(), 9);
}

#[test]
fn rejects_undersized_buffer() {
    let mut plan = TrigTransform::<f64>::new(TrigTransformType::Cosine, 8).unwrap();
    let mut too_small = vec![0.0_f64; 4];
    assert!(plan.forward(&mut too_small).is_err());
}

#[test]
fn f32_round_trip() {
    let n: usize = 8;
    let original: Vec<f32> = (0..=n).map(|i| (i as f32) * 0.25).collect();
    let mut buf = original.clone();

    let mut plan = TrigTransform::<f32>::new(TrigTransformType::Cosine, n).unwrap();
    plan.forward(&mut buf).unwrap();
    plan.backward(&mut buf).unwrap();

    for (after, before) in buf.iter().zip(&original) {
        assert_abs_diff_eq!(after, before, epsilon = 1e-5);
    }
}

use onemkl::pde::{Boundary, Helmholtz2D};

/// Solve `Δu − q u = f` on [0,1]² with homogeneous Dirichlet data and
/// u = sin(πx) sin(πy); return the max-norm error.
fn dirichlet_error(n: usize, q: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let h = 1.0 / n as f64;
    let exact = |x: f64, y: f64| (pi * x).sin() * (pi * y).sin();
    let mut f = vec![0.0; (n + 1) * (n + 1)];
    for j in 0..=n {
        for i in 0..=n {
            f[i + j * (n + 1)] = -(2.0 * pi * pi + q) * exact(i as f64 * h, j as f64 * h);
        }
    }
    let zeros = vec![0.0; n + 1];
    let mut s = Helmholtz2D::<f64>::new(
        (0.0, 1.0), (0.0, 1.0), n, n, [Boundary::Dirichlet; 4], q,
    )
    .unwrap();
    s.solve(&mut f, &zeros, &zeros, &zeros, &zeros).unwrap();
    let mut err = 0.0_f64;
    for j in 0..=n {
        for i in 0..=n {
            err = err.max((f[i + j * (n + 1)] - exact(i as f64 * h, j as f64 * h)).abs());
        }
    }
    err
}

#[test]
fn helmholtz_2d_dirichlet_is_second_order() {
    for q in [0.0, 3.0] {
        let e32 = dirichlet_error(32, q);
        let e64 = dirichlet_error(64, q);
        assert!(e64 < 5e-3, "q={q}: error {e64}");
        let ratio = e32 / e64;
        assert!((3.5..4.5).contains(&ratio), "q={q}: ratio {ratio}");
    }
}

#[test]
fn helmholtz_2d_rejects_bad_arguments() {
    let d = [Boundary::Dirichlet; 4];
    assert!(Helmholtz2D::<f64>::new((0.0, 1.0), (0.0, 1.0), 7, 8, d, 0.0).is_err());
    let mixed = [Boundary::Periodic, Boundary::Dirichlet, Boundary::Dirichlet, Boundary::Dirichlet];
    assert!(Helmholtz2D::<f64>::new((0.0, 1.0), (0.0, 1.0), 8, 8, mixed, 0.0).is_err());
}

#[test]
fn helmholtz_2d_nonzero_dirichlet_data_is_exact_for_quadratics() {
    // u = x^2 - y^2 + 1 is harmonic and exactly representable by the
    // 5-point stencil, so a rectangle with nx != ny and non-zero
    // Dirichlet data must be reproduced to rounding error.
    let (nx, ny) = (16usize, 8usize);
    let (lx, ly) = (2.0_f64, 1.0_f64);
    let (hx, hy) = (lx / nx as f64, ly / ny as f64);
    let u = |x: f64, y: f64| x * x - y * y + 1.0;
    let mut f = vec![0.0; (nx + 1) * (ny + 1)];
    let bd_ax: Vec<f64> = (0..=ny).map(|j| u(0.0, j as f64 * hy)).collect();
    let bd_bx: Vec<f64> = (0..=ny).map(|j| u(lx, j as f64 * hy)).collect();
    let bd_ay: Vec<f64> = (0..=nx).map(|i| u(i as f64 * hx, 0.0)).collect();
    let bd_by: Vec<f64> = (0..=nx).map(|i| u(i as f64 * hx, ly)).collect();
    let mut s = Helmholtz2D::<f64>::new(
        (0.0, lx), (0.0, ly), nx, ny, [Boundary::Dirichlet; 4], 0.0,
    )
    .unwrap();
    s.solve(&mut f, &bd_ax, &bd_bx, &bd_ay, &bd_by).unwrap();
    for j in 0..=ny {
        for i in 0..=nx {
            let want = u(i as f64 * hx, j as f64 * hy);
            assert_abs_diff_eq!(f[i + j * (nx + 1)], want, epsilon = 1e-9);
        }
    }
}



use onemkl::pde::Helmholtz3D;

#[test]
fn helmholtz_3d_dirichlet_is_second_order() {
    fn err(n: usize, q: f64) -> f64 {
        let pi = std::f64::consts::PI;
        let h = 1.0 / n as f64;
        let ex = |x: f64, y: f64, z: f64| (pi * x).sin() * (pi * y).sin() * (pi * z).sin();
        let w = n + 1;
        let mut f = vec![0.0; w * w * w];
        for k in 0..=n { for j in 0..=n { for i in 0..=n {
            f[i + j * w + k * w * w] =
                -(3.0 * pi * pi + q) * ex(i as f64 * h, j as f64 * h, k as f64 * h);
        }}}
        let z = vec![0.0; w * w];
        let mut s = Helmholtz3D::<f64>::new(
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), n, n, n, [Boundary::Dirichlet; 6], q,
        ).unwrap();
        s.solve(&mut f, [&z, &z, &z, &z, &z, &z]).unwrap();
        let mut e = 0.0_f64;
        for k in 0..=n { for j in 0..=n { for i in 0..=n {
            e = e.max((f[i + j * w + k * w * w] - ex(i as f64 * h, j as f64 * h, k as f64 * h)).abs());
        }}}
        e
    }
    for q in [0.0, 3.0] {
        let (e16, e32) = (err(16, q), err(32, q));
        assert!(e32 < 2e-3, "q={q}: error {e32}");
        let ratio = e16 / e32;
        assert!((3.5..4.5).contains(&ratio), "q={q}: ratio {ratio}");
    }
}

#[test]
fn helmholtz_3d_nonzero_dirichlet_data_is_exact_for_polynomials() {
    // u = x^2 - y^2 + z^2 + xyz + 1: Δu = 2, reproduced exactly by the
    // 7-point stencil, on a 2x1x1 box with nx=16, ny=8, nz=4.
    let (nx, ny, nz) = (16usize, 8usize, 4usize);
    let (lx, ly, lz) = (2.0_f64, 1.0_f64, 1.0_f64);
    let (hx, hy, hz) = (lx / nx as f64, ly / ny as f64, lz / nz as f64);
    let u = |x: f64, y: f64, z: f64| x * x - y * y + z * z + x * y * z + 1.0;
    let (wx, wy) = (nx + 1, ny + 1);
    let mut f = vec![2.0; wx * wy * (nz + 1)];
    let mut ax = vec![0.0; wy * (nz + 1)];
    let mut bx = ax.clone();
    for k in 0..=nz { for j in 0..=ny {
        ax[j + k * wy] = u(0.0, j as f64 * hy, k as f64 * hz);
        bx[j + k * wy] = u(lx, j as f64 * hy, k as f64 * hz);
    }}
    let mut ay = vec![0.0; wx * (nz + 1)];
    let mut by = ay.clone();
    for k in 0..=nz { for i in 0..=nx {
        ay[i + k * wx] = u(i as f64 * hx, 0.0, k as f64 * hz);
        by[i + k * wx] = u(i as f64 * hx, ly, k as f64 * hz);
    }}
    let mut az = vec![0.0; wx * wy];
    let mut bz = az.clone();
    for j in 0..=ny { for i in 0..=nx {
        az[i + j * wx] = u(i as f64 * hx, j as f64 * hy, 0.0);
        bz[i + j * wx] = u(i as f64 * hx, j as f64 * hy, lz);
    }}
    let mut s = Helmholtz3D::<f64>::new(
        (0.0, lx), (0.0, ly), (0.0, lz), nx, ny, nz, [Boundary::Dirichlet; 6], 0.0,
    ).unwrap();
    s.solve(&mut f, [&ax, &bx, &ay, &by, &az, &bz]).unwrap();
    for k in 0..=nz { for j in 0..=ny { for i in 0..=nx {
        let want = u(i as f64 * hx, j as f64 * hy, k as f64 * hz);
        assert_abs_diff_eq!(f[i + j * wx + k * wx * wy], want, epsilon = 1e-9);
    }}}
}
