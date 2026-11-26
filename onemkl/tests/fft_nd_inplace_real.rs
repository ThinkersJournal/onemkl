#![cfg(feature = "fft")]

//! N-D in-place real FFT (CCE spectrum written over the padded real
//! array), checked against a naive N-D DFT. The last dimension is padded
//! to `2 * (n_last / 2 + 1)` reals so the spectrum fits in the same buffer.

use approx::assert_abs_diff_eq;
use onemkl::fft::InPlaceRealFftPlan;

/// Row-major strides of the complex spectrum (last dim halved), in complex
/// elements, and the matching real strides of the padded real array.
fn layouts(dims: &[usize]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut cdims = dims.to_vec();
    let last = cdims.len() - 1;
    cdims[last] = dims[last] / 2 + 1;
    let mut cs = vec![1usize; dims.len()];
    for i in (0..last).rev() {
        cs[i] = cs[i + 1] * cdims[i + 1];
    }
    let mut rs: Vec<usize> = cs.iter().map(|c| 2 * c).collect();
    rs[last] = 1;
    (cdims, cs, rs)
}

fn multi(mut flat: usize, dims: &[usize]) -> Vec<usize> {
    let mut idx = vec![0; dims.len()];
    for i in (0..dims.len()).rev() {
        idx[i] = flat % dims[i];
        flat /= dims[i];
    }
    idx
}

fn check(dims: &[usize]) {
    let (cdims, cs, rs) = layouts(dims);
    let mut plan = InPlaceRealFftPlan::<f64>::new(dims).unwrap();
    let total_c: usize = cdims.iter().product();
    assert_eq!(plan.buffer_len(), 2 * total_c);
    let n: usize = dims.iter().product();
    let mut buf = vec![f64::NAN; plan.buffer_len()];
    let mut x = vec![0.0; n];
    for (f, xv) in x.iter_mut().enumerate() {
        let idx = multi(f, dims);
        *xv = ((f * f + 3 * f) % 11) as f64 - 4.0;
        let off: usize = idx.iter().zip(&rs).map(|(i, s)| i * s).sum();
        buf[off] = *xv;
    }
    plan.forward_in_place(&mut buf).unwrap();
    for kf in 0..total_c {
        let k = multi(kf, &cdims);
        let (mut re, mut im) = (0.0, 0.0);
        for (f, xv) in x.iter().enumerate() {
            let j = multi(f, dims);
            let ph: f64 = (0..dims.len())
                .map(|d| (j[d] * k[d]) as f64 / dims[d] as f64)
                .sum();
            let a = -2.0 * std::f64::consts::PI * ph;
            re += xv * a.cos();
            im += xv * a.sin();
        }
        let off: usize = k.iter().zip(&cs).map(|(i, s)| i * s).sum::<usize>() * 2;
        assert_abs_diff_eq!(buf[off], re, epsilon = 1e-9);
        assert_abs_diff_eq!(buf[off + 1], im, epsilon = 1e-9);
    }
}

#[test]
fn forward_2d_even_last() {
    check(&[4, 6]);
}

#[test]
fn forward_2d_odd_last() {
    check(&[3, 5]);
}

#[test]
fn forward_3d() {
    check(&[3, 4, 5]);
}

#[test]
fn round_trip_2d() {
    let dims = [4usize, 6usize];
    let (_, _, rs) = layouts(&dims);
    let mut plan = InPlaceRealFftPlan::<f64>::with_scales(&dims, 1.0, 1.0 / 24.0).unwrap();
    let mut buf = vec![0.0; plan.buffer_len()];
    let mut orig = vec![0.0; plan.buffer_len()];
    for j0 in 0..dims[0] {
        for j1 in 0..dims[1] {
            let v = ((j0 * 7 + j1 * 3) % 5) as f64 - 1.5;
            buf[j0 * rs[0] + j1] = v;
            orig[j0 * rs[0] + j1] = v;
        }
    }
    plan.forward_in_place(&mut buf).unwrap();
    plan.backward_in_place(&mut buf).unwrap();
    for j0 in 0..dims[0] {
        for j1 in 0..dims[1] {
            assert_abs_diff_eq!(buf[j0 * rs[0] + j1], orig[j0 * rs[0] + j1], epsilon = 1e-12);
        }
    }
}

#[test]
fn rejects_bad_arguments() {
    assert!(InPlaceRealFftPlan::<f64>::new(&[]).is_err(), "empty");
    assert!(InPlaceRealFftPlan::<f64>::new(&[4, 0]).is_err(), "zero dim");
    let mut p = InPlaceRealFftPlan::<f64>::new(&[4, 6]).unwrap();
    let mut short = vec![0.0; 10];
    assert!(p.forward_in_place(&mut short).is_err());
    assert!(p.backward_in_place(&mut short).is_err());
}
