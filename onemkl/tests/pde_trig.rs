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


#[test]
fn helmholtz_2d_neumann_data_is_exact_for_quadratics() {
    // x: Dirichlet, y: Neumann (outward normal derivative), q = 3,
    // u = x^2 - y^2 + xy + 1 (harmonic) on [0,2]x[0,1], 16x8.
    let (nx, ny) = (16usize, 8usize);
    let (lx, ly, q) = (2.0_f64, 1.0_f64, 3.0_f64);
    let (hx, hy) = (lx / nx as f64, ly / ny as f64);
    let w = nx + 1;
    let u = |x: f64, y: f64| x * x - y * y + x * y + 1.0;
    let mut f = vec![0.0; w * (ny + 1)];
    for j in 0..=ny { for i in 0..=nx {
        f[i + j * w] = -q * u(i as f64 * hx, j as f64 * hy);
    }}
    let ax: Vec<f64> = (0..=ny).map(|j| u(0.0, j as f64 * hy)).collect();
    let bx: Vec<f64> = (0..=ny).map(|j| u(lx, j as f64 * hy)).collect();
    // du/dy = -2y + x; outward normal at y = 0 is -y.
    let ay: Vec<f64> = (0..=nx).map(|i| -(i as f64 * hx)).collect();
    let by: Vec<f64> = (0..=nx).map(|i| -2.0 * ly + i as f64 * hx).collect();
    let bc = [Boundary::Dirichlet, Boundary::Dirichlet, Boundary::Neumann, Boundary::Neumann];
    let mut s = Helmholtz2D::<f64>::new((0.0, lx), (0.0, ly), nx, ny, bc, q).unwrap();
    s.solve(&mut f, &ax, &bx, &ay, &by).unwrap();
    for j in 0..=ny { for i in 0..=nx {
        assert_abs_diff_eq!(f[i + j * w], u(i as f64 * hx, j as f64 * hy), epsilon = 1e-9);
    }}
}

fn conv_2d(n: usize, bc: [Boundary; 4]) -> f64 {
    // x periodic (or Neumann), y Dirichlet (or Neumann); q = 3.
    let pi = std::f64::consts::PI;
    let h = 1.0 / n as f64;
    let w = n + 1;
    let q = 3.0;
    let (u, lam): (Box<dyn Fn(f64, f64) -> f64>, f64) = if bc[0] == Boundary::Periodic {
        (Box::new(move |x, y| (2.0 * pi * x).sin() * (pi * y).sin()), 5.0 * pi * pi)
    } else {
        (Box::new(move |x, y| (pi * x).cos() * (pi * y).cos()), 2.0 * pi * pi)
    };
    let mut f = vec![0.0; w * w];
    for j in 0..=n { for i in 0..=n {
        f[i + j * w] = -(lam + q) * u(i as f64 * h, j as f64 * h);
    }}
    let z = vec![0.0; w];
    let mut s = Helmholtz2D::<f64>::new((0.0, 1.0), (0.0, 1.0), n, n, bc, q).unwrap();
    s.solve(&mut f, &z, &z, &z, &z).unwrap();
    let mut e = 0.0_f64;
    for j in 0..=n { for i in 0..=n {
        e = e.max((f[i + j * w] - u(i as f64 * h, j as f64 * h)).abs());
    }}
    e
}

#[test]
fn helmholtz_2d_periodic_and_neumann_are_second_order() {
    use Boundary::*;
    for bc in [[Periodic, Periodic, Dirichlet, Dirichlet], [Neumann; 4], [Dirichlet, Dirichlet, Periodic, Periodic]] {
        if bc[2] == Periodic {
            continue; // the sin(pi y) factor above is not y-periodic
        }
        let (e16, e32) = (conv_2d(16, bc), conv_2d(32, bc));
        let ratio = e16 / e32;
        assert!(e32 < 5e-3, "{bc:?}: error {e32}");
        assert!((3.5..4.5).contains(&ratio), "{bc:?}: ratio {ratio}");
    }
}

#[test]
fn helmholtz_3d_neumann_data_is_exact_for_polynomials() {
    // x, z Dirichlet; y Neumann. u = x^2 - y^2 + z^2 + xyz + 1, Δu = 2,
    // q = 0 (nonsingular because of the Dirichlet faces).
    let (nx, ny, nz) = (8usize, 8usize, 4usize);
    let (lx, ly, lz) = (1.0_f64, 1.0_f64, 1.0_f64);
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
    // du/dy = -2y + xz; outward normal at y = 0 is -y.
    let mut ay = vec![0.0; wx * (nz + 1)];
    let mut by = ay.clone();
    for k in 0..=nz { for i in 0..=nx {
        let (x, z) = (i as f64 * hx, k as f64 * hz);
        ay[i + k * wx] = -(x * z);
        by[i + k * wx] = -2.0 * ly + x * z;
    }}
    let mut az = vec![0.0; wx * wy];
    let mut bz = az.clone();
    for j in 0..=ny { for i in 0..=nx {
        az[i + j * wx] = u(i as f64 * hx, j as f64 * hy, 0.0);
        bz[i + j * wx] = u(i as f64 * hx, j as f64 * hy, lz);
    }}
    use Boundary::*;
    let bc = [Dirichlet, Dirichlet, Neumann, Neumann, Dirichlet, Dirichlet];
    let mut s = Helmholtz3D::<f64>::new((0.0, lx), (0.0, ly), (0.0, lz), nx, ny, nz, bc, 0.0).unwrap();
    s.solve(&mut f, [&ax, &bx, &ay, &by, &az, &bz]).unwrap();
    for k in 0..=nz { for j in 0..=ny { for i in 0..=nx {
        assert_abs_diff_eq!(
            f[i + j * wx + k * wx * wy],
            u(i as f64 * hx, j as f64 * hy, k as f64 * hz),
            epsilon = 1e-9
        );
    }}}
}

#[test]
fn helmholtz_3d_mixed_periodic_dirichlet_neumann_is_second_order() {
    use Boundary::*;
    fn err(n: usize) -> f64 {
        let pi = std::f64::consts::PI;
        let h = 1.0 / n as f64;
        let w = n + 1;
        let q = 3.0;
        let u = |x: f64, y: f64, z: f64| (2.0 * pi * x).sin() * (pi * y).sin() * (pi * z).cos();
        let mut f = vec![0.0; w * w * w];
        for k in 0..=n { for j in 0..=n { for i in 0..=n {
            f[i + j * w + k * w * w] = -(6.0 * pi * pi + q) * u(i as f64 * h, j as f64 * h, k as f64 * h);
        }}}
        let z = vec![0.0; w * w];
        let bc = [Periodic, Periodic, Dirichlet, Dirichlet, Neumann, Neumann];
        let mut s = Helmholtz3D::<f64>::new((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), n, n, n, bc, q).unwrap();
        s.solve(&mut f, [&z, &z, &z, &z, &z, &z]).unwrap();
        let mut e = 0.0_f64;
        for k in 0..=n { for j in 0..=n { for i in 0..=n {
            e = e.max((f[i + j * w + k * w * w] - u(i as f64 * h, j as f64 * h, k as f64 * h)).abs());
        }}}
        e
    }
    let (e16, e32) = (err(16), err(32));
    assert!(e32 < 5e-2, "error {e32}");
    assert!((3.5..4.5).contains(&(e16 / e32)), "ratio {}", e16 / e32);
}

#[test]
fn helmholtz_f32_reproduces_polynomials_2d_and_3d() {
    use Boundary::*;
    // 2-D: x Dirichlet, y Neumann, q = 3, u = x^2 - y^2 + xy + 1.
    let (nx, ny) = (16usize, 8usize);
    let (lx, ly, q) = (2.0_f32, 1.0_f32, 3.0_f32);
    let (hx, hy) = (lx / nx as f32, ly / ny as f32);
    let w = nx + 1;
    let u = |x: f32, y: f32| x * x - y * y + x * y + 1.0;
    let mut f = vec![0.0_f32; w * (ny + 1)];
    for j in 0..=ny { for i in 0..=nx { f[i + j * w] = -q * u(i as f32 * hx, j as f32 * hy); } }
    let ax: Vec<f32> = (0..=ny).map(|j| u(0.0, j as f32 * hy)).collect();
    let bx: Vec<f32> = (0..=ny).map(|j| u(lx, j as f32 * hy)).collect();
    let ay: Vec<f32> = (0..=nx).map(|i| -(i as f32 * hx)).collect();
    let by: Vec<f32> = (0..=nx).map(|i| -2.0 * ly + i as f32 * hx).collect();
    let mut s = Helmholtz2D::<f32>::new(
        (0.0, lx), (0.0, ly), nx, ny, [Dirichlet, Dirichlet, Neumann, Neumann], q,
    ).unwrap();
    s.solve(&mut f, &ax, &bx, &ay, &by).unwrap();
    for j in 0..=ny { for i in 0..=nx {
        assert_abs_diff_eq!(f[i + j * w], u(i as f32 * hx, j as f32 * hy), epsilon = 2e-3);
    }}

    // 3-D: all Dirichlet, u = x^2 - y^2 + z^2 + xyz + 1, Δu = 2.
    let n = 8usize;
    let h = 1.0_f32 / n as f32;
    let u3 = |x: f32, y: f32, z: f32| x * x - y * y + z * z + x * y * z + 1.0;
    let w = n + 1;
    let mut f = vec![2.0_f32; w * w * w];
    let mut faces: Vec<Vec<f32>> = vec![vec![0.0; w * w]; 6];
    for b in 0..w { for a in 0..w {
        let (p, r) = (a as f32 * h, b as f32 * h);
        faces[0][a + b * w] = u3(0.0, p, r); faces[1][a + b * w] = u3(1.0, p, r);
        faces[2][a + b * w] = u3(p, 0.0, r); faces[3][a + b * w] = u3(p, 1.0, r);
        faces[4][a + b * w] = u3(p, r, 0.0); faces[5][a + b * w] = u3(p, r, 1.0);
    }}
    let mut s = Helmholtz3D::<f32>::new(
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), n, n, n, [Dirichlet; 6], 0.0,
    ).unwrap();
    s.solve(&mut f, [&faces[0], &faces[1], &faces[2], &faces[3], &faces[4], &faces[5]]).unwrap();
    for k in 0..=n { for j in 0..=n { for i in 0..=n {
        assert_abs_diff_eq!(
            f[i + j * w + k * w * w],
            u3(i as f32 * h, j as f32 * h, k as f32 * h),
            epsilon = 2e-3
        );
    }}}
}

use onemkl::pde::SphericalHelmholtz;

/// Max error of `Δ_S u - q u = F` for a sum of spherical harmonics
/// Y_1^0 (cos t), Y_2^1 (sin t cos t cos p) and Y_2^2 (sin^2 t cos 2p).
fn sph_error(np: usize, nt: usize) -> f64 {
    let pi = std::f64::consts::PI;
    let q = 3.0;
    let w = np + 1;
    let u = |t: f64, p: f64| {
        t.cos() + 0.5 * t.sin() * t.cos() * p.cos() + 0.25 * t.sin().powi(2) * (2.0 * p).cos()
    };
    // Laplace-Beltrami eigenvalues -l(l+1): -2, -6, -6.
    let rhs = |t: f64, p: f64| {
        -(2.0 + q) * t.cos()
            - (6.0 + q) * 0.5 * t.sin() * t.cos() * p.cos()
            - (6.0 + q) * 0.25 * t.sin().powi(2) * (2.0 * p).cos()
    };
    let mut f = vec![0.0; w * (nt + 1)];
    for j in 0..=nt { for i in 0..=np {
        f[i + j * w] = rhs(j as f64 * pi / nt as f64, i as f64 * 2.0 * pi / np as f64);
    }}
    let mut s = SphericalHelmholtz::<f64>::new(np, nt, q).unwrap();
    s.solve(&mut f).unwrap();
    let mut e = 0.0_f64;
    for j in 0..=nt { for i in 0..=np {
        let want = u(j as f64 * pi / nt as f64, i as f64 * 2.0 * pi / np as f64);
        e = e.max((f[i + j * w] - want).abs());
    }}
    e
}

#[test]
fn spherical_helmholtz_is_second_order() {
    let (e1, e2) = (sph_error(16, 8), sph_error(32, 16));
    let (e3, e4) = (sph_error(64, 32), sph_error(128, 64));
    eprintln!("sph errors {e1:e} {e2:e} {e3:e} {e4:e}");
    assert!(e4 < 1e-3, "error {e4}");
    for (a, b) in [(e1, e2), (e2, e3), (e3, e4)] {
        assert!((3.0..5.0).contains(&(a / b)), "ratio {}", a / b);
    }
}

use onemkl::pde::SphericalPatchHelmholtz;

const PATCH: (f64, f64, f64, f64) = (0.3, 2.0, 0.6, 2.4); // ap, bp, at, bt

fn patch_u(t: f64, p: f64) -> f64 {
    t.cos() + 0.5 * t.sin() * t.cos() * p.cos() + 0.25 * t.sin().powi(2) * (2.0 * p).cos()
}

/// Max interior error vs the continuous solution; boundary data = exact `u`.
fn patch_error<T: onemkl::pde::HelmholtzScalar>(np: usize, nt: usize) -> f64 {
    let (ap, bp, at, bt) = PATCH;
    let q = 3.0;
    let w = np + 1;
    let rhs = |t: f64, p: f64| {
        -(2.0 + q) * t.cos()
            - (6.0 + q) * 0.5 * t.sin() * t.cos() * p.cos()
            - (6.0 + q) * 0.25 * t.sin().powi(2) * (2.0 * p).cos()
    };
    let node = |i: usize, j: usize| (at + j as f64 * (bt - at) / nt as f64, ap + i as f64 * (bp - ap) / np as f64);
    let mut f = vec![T::default(); w * (nt + 1)];
    for j in 0..=nt {
        for i in 0..=np {
            let (t, p) = node(i, j);
            let boundary = i == 0 || i == np || j == 0 || j == nt;
            f[i + j * w] = T::from_f64(if boundary { patch_u(t, p) } else { rhs(t, p) });
        }
    }
    let mut s = SphericalPatchHelmholtz::<T>::new(ap, bp, at, bt, np, nt, T::from_f64(q)).unwrap();
    s.solve(&mut f).unwrap();
    let mut e = 0.0_f64;
    for j in 0..=nt {
        for i in 0..=np {
            let (t, p) = node(i, j);
            e = e.max((f[i + j * w].to_f64() - patch_u(t, p)).abs());
        }
    }
    e
}

#[test]
fn spherical_patch_is_second_order() {
    let e: Vec<f64> = [(16, 8), (32, 16), (64, 32), (128, 64)].iter().map(|&(a, b)| patch_error::<f64>(a, b)).collect();
    eprintln!("patch errors {e:?}");
    assert!(e[3] < 1e-3, "error {}", e[3]);
    for k in 0..3 {
        assert!((3.0..5.0).contains(&(e[k] / e[k + 1])), "ratio {}", e[k] / e[k + 1]);
    }
}

#[test]
fn spherical_patch_f32_converges() {
    let e = patch_error::<f32>(64, 32);
    assert!(e < 2e-3, "f32 error {e}");
}

/// Exact check of the wrapper's discrete model: apply the discrete operator
/// to an arbitrary grid function (non-zero boundary), solve, recover it.
#[test]
fn spherical_patch_reproduces_a_discrete_solution() {
    let (ap, bp, at, bt) = PATCH;
    let (np, nt, q) = (10usize, 7usize, 2.5_f64);
    let w = np + 1;
    let (hp, ht) = ((bp - ap) / np as f64, (bt - at) / nt as f64);
    let s = |j: usize| (at + j as f64 * ht).sin();
    let u: Vec<f64> = (0..w * (nt + 1)).map(|k| ((k * 37 % 101) as f64) / 50.0 - 1.0).collect();
    let mut f = u.clone();
    for j in 1..nt {
        let (sj, sp, sm) = (s(j), 0.5 * (s(j) + s(j + 1)), 0.5 * (s(j) + s(j - 1)));
        for i in 1..np {
            let k = i + j * w;
            let au = ht * ht * sj * q * u[k]
                + sp * (u[k] - u[k + w]) + sm * (u[k] - u[k - w])
                + ht * ht / (sj * hp * hp) * (2.0 * u[k] - u[k - 1] - u[k + 1]);
            f[k] = -au / (ht * ht * sj);
        }
    }
    let mut sol = SphericalPatchHelmholtz::<f64>::new(ap, bp, at, bt, np, nt, q).unwrap();
    sol.solve(&mut f).unwrap();
    for k in 0..f.len() {
        assert!((f[k] - u[k]).abs() < 1e-10, "node {k}: {} vs {}", f[k], u[k]);
    }
}

#[test]
fn spherical_patch_rejects_poles_and_bad_sizes() {
    let pi = std::f64::consts::PI;
    assert!(SphericalPatchHelmholtz::<f64>::new(0.0, 1.0, 0.0, 1.0, 8, 8, 1.0).is_err());
    assert!(SphericalPatchHelmholtz::<f64>::new(0.0, 1.0, 1.0, pi, 8, 8, 1.0).is_err());
    assert!(SphericalPatchHelmholtz::<f64>::new(1.0, 1.0, 0.5, 1.0, 8, 8, 1.0).is_err());
    assert!(SphericalPatchHelmholtz::<f64>::new(0.0, 1.0, 0.5, 1.0, 1, 8, 1.0).is_err());
}
