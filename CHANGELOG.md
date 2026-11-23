# Changelog

All notable changes to this project are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

### Fixed

- `onemkl-sys` build: `bindgen` regains its `runtime` feature (kept `default-features = false`, so `prettyplease` stays dropped). Without it `clang-sys` linked libclang into the build script, which failed to start on machines without libclang (Windows `STATUS_DLL_NOT_FOUND` 0xc0000135; needed a `libclang-dev` CI workaround). With it libclang is dlopen'd only when bindgen runs. Evidence (Windows, no libclang.dll): `DOCS_RS=1 cargo build -p onemkl-sys` exits 101/0xc0000135 on 0.3.6 and exits 0 with this change; `cargo tree -e features` shows `clang-sys feature "runtime"` 0 -> 1. The CI `libclang-dev` step stays: bindgen still needs libclang when it runs. Not run: the syn-3/prost-build stuck-lock scratch crate, Linux/macOS.

## 0.3.6 - 2026-11-23 (allocated by pm-assist)

### Fixed

- `onemkl-sys` build: `bindgen` is now built with `default-features = false` (feature `logging` only), which drops its `prettyplease` dependency. A lock that pairs `syn` 3 with `prettyplease` 0.2 (held there by, e.g., `prost-build` 0.14.4) made bindgen 0.73.2 fail to compile (`E0277: syn::file::File: ParseQuote` at bindgen `lib.rs:1017`). Reproduced in a scratch crate (bindgen 0.73 + prost-build =0.14.4, lock edited to bindgen->syn 3 + prettyplease 0.2.37: fails; with this change: builds); `onemkl-sys` built on Linux (WSL, oneMKL 2026.2.1). The generated bindings are now unformatted. Not run: Windows, macOS, docs.rs.

### Added

- `pde::SphericalHelmholtz` (`?_sph_p`): fast Helmholtz solver `Δ_S u - q·u = F` on the full unit sphere surface (periodic in phi, theta 0..pi). The documentation pages for `?_init_sph_p` could not be read (every release returned only the site sidebar), so the convention was measured on oneMKL 2026.2.1 (pip, Linux x86_64, LP64): argument order `(ap, bp, at, bt)`; the solver matrix measured from unit-ring inputs is `h_t²·sin(t_j)·(-Δ_h + q)` with the conservative theta stencil built from averaged node sines and weight 1 on the pole rows (exact match at 9x9), which the wrapper turns into the documented-style equation. Test `pde_trig`: a sum of Y_1^0, Y_2^1 and Y_2^2 with `q=3` converges at second order (error 2.7e-2, 6.8e-3, 1.7e-3, 4.3e-4 for np/nt = 16/8 to 128/64). The phi-mode term for m >= 1 was validated only through that convergence test, not measured separately. Not wrapped or exercised: the non-periodic `?_sph_np`, partial spheres (the domain is fixed to the full sphere), `f32`, `q = 0`, other MKL versions, Windows, macOS, OpenMP/TBB, ILP64; `dpar` size is the documented-by-search formula padded 4x plus 200, not taken from the vendor docs.
- `link-rt` linkage (single dynamic library, `mkl_rt`) for `onemkl-sys`/`onemkl`, with `service::set_interface_layer` and `service::set_threading_layer` (`MKL_Set_Interface_Layer` / `MKL_Set_Threading_Layer`). Test `service_rt` (Linux x86_64, LP64, oneMKL 2026.2.1 pip, `--no-default-features --features lp64,threading-sequential,link-rt`): setting LP64 and the sequential layer succeed before first use, `max_threads()` is 1 and `version_string()` works; enabling `link-dynamic` together with `link-rt` is rejected by the build script. `link-rt` with the MPI-dependent domains is rejected. Not exercised: Windows (`mkl_rt.lib`), macOS, ILP64, the OpenMP/TBB/GNU layers, setting a layer after MKL has initialised (returns an error by construction, not measured), and the default build on this commit beyond the existing test suite.
- `blas::compact::getrinp` (`mkl_?getrinp_compact`, `f32`/`f64`): in-place inverse of matrices factored by `getrfnp`, workspace allocated from MKL's query. Test `getrinp_inverts_lu_factored_matrices` (Linux x86_64, LP64, sequential, oneMKL 2026.2.1, AVX pack, column-major, two 2x2 matrices): both inverses match hand-computed values at 1e-12. `f32`, complex types, larger sizes, other packs and layouts not exercised.
- `lapack::larft` and `lapack::larfb` (`LAPACKE_?larft` / `LAPACKE_?larfb`) for `f32`, `f64`, `Complex32`, `Complex64`, with `ReflectorDirection` / `ReflectorStorage`. Test `larft_and_larfb_reproduce_the_qr_r_factor` (Linux x86_64, LP64, sequential, oneMKL 2026.2.1, `f64`, column-major, 3x2): `T` from `larft` has the `geqrf` taus on its diagonal, and `larfb` applying `Qᵀ` (`Side::Left`, forward, columnwise) to a copy of `A` reproduces `geqrf`'s `R` to 1e-12 with zeros below. The complex types compile but are not run; row-major, `Side::Right`, backward and rowwise storage, and `Trans` variants other than the one tested are not exercised.
- `lapack::lacgv` and `lapack::lacrm` (`LAPACKE_?lacgv` / `LAPACKE_?lacrm`) for `Complex32` / `Complex64` via the new `ComplexAuxScalar` trait. Tests (Linux x86_64, LP64, sequential, oneMKL 2026.2.1, `Complex64`): `lacgv` conjugates exactly the strided elements (`n=2`, `incx=2`); `lacrm` matches a hand-computed 2x2 product in row-major. `Complex32`, negative `incx`, column-major and non-square `m x n` not exercised.
- `lapack::sytrf`, `lapack::syconv` and `SyconvWay` (`LAPACKE_?sytrf` / `LAPACKE_?syconv`) for `f32`, `f64`, `Complex32`, `Complex64` (symmetric, not Hermitian) via the new `SymConvScalar` trait. Test `lapack_auxiliary::syconv_roundtrip_all_variants` (Linux x86_64, LP64, sequential, oneMKL 2026.2.1 pip, `f64` only, 4x4 zero-diagonal matrix so Bunch-Kaufman takes 2x2 pivots; upper/lower x row/column-major): `Convert` changes the buffer and leaves non-zero off-diagonal `D` entries in `e`, `Revert` restores the `sytrf` output exactly. Not exercised: `f32` and complex types, the numerical content of `A` and `e` after `Convert` against a reference, larger sizes, Windows, macOS, OpenMP/TBB, ILP64.
- `optim::TrnlsSession` (`TrnlsStep`, `TrnlsCheck`): step-by-step driver of `dtrnlsp_*` where the caller runs the reverse-communication loop, plus `TrnlsSession::check` (`dtrnlsp_check`, raw six-entry `info`). Test `optim_session` (Linux x86_64, LP64, sequential, oneMKL 2026.2.1 pip): a 2x2 line fit driven step by step reaches (1, 1) to 1e-6; `check` returns all-zero `info` for valid data and a non-zero `info` for a NaN Jacobian entry (the meaning of the individual `info` entries was not decoded or pinned). Measured gotcha: the `rci` state variable must persist between calls (resetting it to 0 made MKL request the residual forever). `f64` only; the bounded (`trnlspbc`) variant has no session form. Not exercised: Windows, macOS, OpenMP/TBB, ILP64.
- `optim::TrnlsBoundedSession`: the `dtrnlspbc_*` counterpart of `TrnlsSession` (owns `x` and the bounds; `step`, `check` via `dtrnlspbc_check`). Tests in `optim_session` (Linux x86_64, LP64, sequential, oneMKL 2026.2.1 pip): an active upper bound is respected, the residual drops from its start value, and the session's final `x` equals `solve_trnls_bounded`'s to 1e-8 (same loop, so this pins the session plumbing, not MKL's accuracy); on that toy problem MKL stops after 1 iteration at (1.0, 0.49975), short of the true constrained optimum (1.75, 0.5), which the tests do not claim. `check` flags a NaN Jacobian; `info` entries still undecoded. `f64` only. Not exercised: Windows, macOS, OpenMP/TBB, ILP64.
- Tests only (no API change): `vm_f32` (f32 `tanh/exp/sin/cos/ln/sqrt/add/sub/mul/div` vs std at 1e-6, `fmax`/`fmin` f32+f64, NaN operand ignored by both, NaN only if both NaN, signed zero unmeasured), `blas_gemm_f32` (sgemm, both layouts x 4 transposes vs a naive loop), `blas_fortran_gemm_map` (the Fortran `?gemm_` -> onemkl mapping: column-major `with_lda`, `'N'|'T'|'C'` -> `Transpose`; f32/f64 3x3 transpose pairs and f16 `hgemm` 2x2 pairs on 3x4x5 data with ld > rows, padding of C checked untouched; three seeded wrong mappings are rejected or give different numbers). Finding pinned by test: `hgemm` requires `cols*ld` elements per operand, stricter than Fortran's `(cols-1)*ld+rows`. Linux x86_64, LP64, sequential, oneMKL 2026.2.1 pip; Windows, macOS, OpenMP/TBB, ILP64 not run.
- `pde::SphericalPatchHelmholtz` (`?_sph_np`): Helmholtz `Δ_S u - q·u = F` on a sphere patch `[ap,bp] x [at,bt]` (`0 < at < bt < pi`, no pole) with Dirichlet data on all four sides. Measured on oneMKL 2026.2.1 (pip, Linux x86_64, LP64, sequential) by unit-input probes at np=6, nt=4 on two domains: argument order `(ap, bp, at, bt)`; the interior block of the solver matrix is the inverse of the same `hθ²·sinθ·(-Δ_h+q)` operator as `?_sph_p` with Dirichlet φ ends (dense-model agreement 1e-15); MKL's boundary rows/columns are decoupled lines (`j=0,nt` pass `f` through, `i=0,np` solved as θ-only problems; the `i=np` line also carries `4hθ²/(sinθ·hφ²)` extra on the diagonal), so the wrapper does the Dirichlet lift itself. Test `pde_trig` (same platform): an arbitrary grid function with non-zero boundary is recovered to 1e-10 after applying the discrete operator and solving; spherical-harmonic sum on [0.3,2.0]x[0.6,2.4], q=3, error 2.2e-3, 5.5e-4, 1.4e-4, 3.5e-5 for np/nt 16/8 to 128/64 (second order); `f32` at 64/32 under 2e-3; mutant (dropping the φ-side lift term) fails all three solver tests. Not exercised: the pole-including sphere (`at=0`), odd `np`, Neumann-like ends (none offered), `q=0`, the meaning of MKL's own boundary-line output (never used), other MKL versions, Windows, macOS, OpenMP/TBB, ILP64. Includes the Dirichlet theta data being lifted by the wrapper rather than by MKL: MKL's documentation pages could not be read, so every behavioural statement here is measured, not documented.
- Test `loaded_mkl_path` (Linux only): reads `/proc/self/maps` and fails with `STALE MKL: loaded libs from <dir> but expected under <dir>` when the loaded `libmkl*` directory is not the (symlink-resolved) `MKLROOT/lib`; a deliberately wrong root must fail first (control). Measured on the WSL rig: loaded `/root/mklenv/lib` (pip mkl 2026.2, build 20260915) via `/root/mklroot/lib` symlinks. Windows (`GetModuleFileNameW`) and macOS not implemented.

## 0.3.5 - 2026-11-22

### Added

- `blas::compact`: `trsm`, `potrf`, `getrfnp` and `geqrf` (`mkl_?trsm_compact`, `mkl_?potrf_compact`, `mkl_?getrfnp_compact`, `mkl_?geqrf_compact`) for `f32`/`f64`, with buffer-length checks against `get_size`. Test `blas_compact_factor` (Linux x86_64, LP64, sequential, oneMKL 2026.2.1, AVX pack, column-major, `nm` 1-2, 2x2 matrices): hand-computed lower-triangular solve, Cholesky and no-pivot LU match at 1e-12; 2x2 QR matches |R| entries at 1e-12 (tau contents unchecked). Complex types not wrapped; row-major, larger sizes, SSE/AVX-512 packs, Windows, macOS, OpenMP/TBB, ILP64 not exercised.
- `pde::Helmholtz2D` (`d_/s_helmholtz_2d`): 2-D Cartesian Helmholtz/Poisson solver `Δu - q·u = f` with Dirichlet, Neumann (outward normal derivative) and periodic sides. Test `pde_trig` (Linux x86_64, LP64, sequential, oneMKL 2026.2.1 pip): sin·sin solution on the unit square converges at second order (error ratio 3.5-4.5 from n=32 to 64, q=0 and q=3), and u = x²-y²+1 with non-zero Dirichlet data on a 2x1 rectangle, nx=16, ny=8, is reproduced to 1e-9. Measured MKL behaviour the wrapper corrects for: the library returns `(-Δ+q)⁻¹ f / hy²` (opposite sign and scaled by 1/hy² against the documented convention, fitted to 10 digits on 8 domain/q combinations at n=8) and applies no Dirichlet boundary lift (interior stayed exactly 0 for every placement of the data tried); the wrapper rescales by `-hy²` and moves boundary data to the right-hand side. Neumann and periodic (same platform): a raw 2-D probe of 8 boundary combinations x 3 domains fitted a dense reference exactly (periodic: same 1/hy² scale; Neumann: ghost-point operator with MKL weighting Neumann boundary rows by 2, so the wrapper halves the right-hand side there); MKL ignores boundary-data arrays, so the wrapper applies Dirichlet `g/h²` and Neumann `2g/h` lifts itself. Tests: exact reproduction of a quadratic with non-zero Neumann data (2-D, `q=3`, D x N, 16x8; 3-D D x N x D, 8x8x4, to 1e-9) and second-order convergence for periodic x Dirichlet, all-Neumann (2-D) and periodic x Dirichlet x Neumann (3-D). `f32` (same platform): 2-D Dirichlet x Neumann with `q=3` and 3-D all-Dirichlet reproduce polynomials to 2e-3 (single precision; periodic/Neumann 3-D `f32` untested). Not exercised: spherical solvers, 3-D raw Neumann/periodic probe (validated only through the tests above), other MKL versions, Windows, macOS, OpenMP/TBB, ILP64.
- `pde::Helmholtz3D` (`d_/s_helmholtz_3d`): 3-D counterpart of `Helmholtz2D` with the same boundary types. Test `pde_trig` (same platform as above): sin·sin·sin converges at second order (ratio 3.5-4.5 from n=16 to 32, q=0 and q=3) and u = x²-y²+z²+xyz+1 with non-zero face data on a 2x1x1 box, 16x8x4, is reproduced to 1e-9. MKL's 3-D result is `(-Δ+q)⁻¹ f / hz²` (fitted exactly on 5 box/q combinations at n=8) with no boundary lift; the wrapper corrects both. Not exercised: Neumann/periodic, spherical solvers, other MKL versions, Windows, macOS, OpenMP/TBB, ILP64; the `dpar` size is padded generously, not taken from MKL's documented formula.

## 0.3.4 - 2026-11-21

### Added

- `optim`: `numerical_jacobian_central` (`djacobix`, central differences) with a closure residual; a panic in the closure is caught in the callback and re-raised after MKL returns. Unlike the existing `numerical_jacobian` it gives MKL an owned copy of `x`. Test `optim_jacobix` (Linux x86_64, LP64, sequential, oneMKL 2026.2.1): output is column-major `m x n` and matches the analytic Jacobian within 1e-5 at `eps = 1e-8`. Single precision (`sjacobix`) not wrapped; Windows, macOS, OpenMP/TBB, ILP64 not exercised.

### Changed

- `optim::numerical_jacobian` now gives MKL an owned copy of `x`. Before, `x.as_ptr()` from a shared `&[f64]` was passed to `djacobi_init` and MKL perturbed it in place (undefined behaviour in Rust's model). No wrong result was observed before or after (oneMKL 2026.2.1, Linux x86_64, debug and release: closure sees the perturbed `x`, Jacobian matches analytic within 1e-6); this is a soundness fix, not a behaviour fix. New test `optim_numerical_jacobian`.

- `optim::djacobi_with_callback` likewise gives MKL an owned copy of `x` (same soundness reasoning, not measured separately). `numerical_jacobian` frees its RCI handle on a panicking residual (RAII guard). `m * n` is overflow-checked (`DimensionOverflow`) in the optim Jacobian routines instead of wrapping on 32-bit targets (tests use `m = usize::MAX`; 32-bit itself not run). `numerical_jacobian_central`: the residual must be `Send` and is serialised by a mutex, and MKL's output buffer is zeroed before it becomes a `&mut [f64]`; OpenMP/TBB layers not exercised, so the concurrency guard is by construction, not measured.

## 0.3.3 - 2026-11-20

### Added

- `service`: `set_xerbla_handler` (`mkl_set_xerbla`) and `set_progress_handler` (`mkl_set_progress`): safe, process-global handlers (plain `fn` pointers behind `extern "C"` trampolines; a panicking handler is caught, not unwound into MKL). `None` restores the entry MKL had before the first install. Tests `service_xerbla` and `service_progress` (one test each, own process). Measured on oneMKL 2026.2.1, Linux x86_64, LP64, sequential: `xerbla` receives the positive argument position (`2`) while the routine's `info` is `-2`; a 512x512 `dpotrf` makes 16 progress callbacks, and returning `Stop` ends it after one with `info = -1002`. Windows, macOS, OpenMP/TBB and ILP64 not exercised; routines other than `dpotrf` not checked for progress callbacks.

## 0.3.2 - 2026-11-20

### Added

- `service`: conditional numerical reproducibility (CBWR): `CbwrBranch`, `CbwrError`, `cbwr_set(branch)`, `cbwr_set_strict(branch)`, `cbwr_branch`, `cbwr_strict`, `cbwr_auto_branch` over `MKL_CBWR_Set` / `MKL_CBWR_Get` / `MKL_CBWR_Get_Auto_Branch`. Only branches present in the oneMKL 2026.2.1 headers are exposed. New test `onemkl/tests/service_cbwr.rs` (Linux x86_64, LP64, sequential; it checks the default (off) state, set, query and the `ModeLocked` late-set error (needs the `blas` feature to force a dispatch), not bit-reproducibility across CPUs). On that build `cbwr_auto_branch` returns `Auto` itself.

## 0.3.1 - 2026-11-19

### Changed

- Removed the unused `thiserror` dependency from `onemkl` (nothing in `onemkl/src` referenced it; the crate builds without it).
- New test `onemkl/tests/vml_accuracy_mode.rs` pins the VML accuracy mode: `onemkl` never calls `vmlSetMode`, so `vm` functions run in the thread default, which the test asserts is `VML_HA` (oneMKL 2026.2.1, Linux, LP64; Windows not exercised).
- ROADMAP: completeness wording and the per-routine-test criterion are marked UNVERIFIED, and each domain is mapped to its test files.

## 0.3.0 - 2026-11-18

### Breaking (pre-1.0 minor bump)

- `onemkl-sys`: the bindgen 0.72 -> 0.73 bump removed the `Copy` and `Clone` derives from two opaque handle structs, `sparse_matrix` and `DFTI_DESCRIPTOR` (they are zero-sized `_unused: [u8; 0]` types; the `*_t` / `*_HANDLE` pointer aliases are unchanged). Checked by generating `bindings.rs` with bindgen 0.72.1 (origin/main) and 0.73.2 (this branch) against oneMKL 2026.2.1 and diffing: 78 differing lines, which are the generator banner, those two derive lines, and `#[allow(unused_imports)] use super::*;` added inside 36 constant modules. No function, type name or signature changed. `onemkl` was not diffed with a tool; a search of `onemkl/src` finds no use of `sparse_matrix` or `DFTI_DESCRIPTOR` other than via the `sparse_matrix_t` / `DFTI_DESCRIPTOR_HANDLE` aliases.

### Fixed

- Linux dynamic builds: a crate that depends on `onemkl-sys` and runs its own test binary no longer dies at load with `undefined symbol: mkl_blas_dgemm` (issue #3). `onemkl-sys` now references one symbol from `libmkl_core` and one from the threading layer, so `--as-needed` keeps both in `DT_NEEDED` for every dependent; no `RUSTFLAGS` workaround is needed. Verified with a downstream crate against oneMKL 2026.2.1 (sequential, LP64, dynamic); the OpenMP and TBB layers, ILP64 and Windows were not exercised.

### Changed

- `bindgen` 0.72 -> 0.73 (resolves 0.73.2); nothing blocked it.
- CI gains a `cargo deny` job (`deny.toml`, `all-features = true`).
- README gains a claim-to-command table; the ROADMAP test count is re-measured (373 passed, 5 ignored; it said 170).

## 0.2.2 - 2026-10-31

### Changed

- Docs only, no code change. ROADMAP gains an "Ownership" section: Bonefish owns this project ("Bonefish owns oneMKL/oneAPI in our software ecosystem." (CireSnave, board 329, ruled 2026-10-30, recorded by the portfolio PM)), and the SYCL/DPC++ interface lives in `ThinkersJournal/bonefish`. The "DPC++ / SYCL APIs" out-of-scope entry now names Bonefish instead of "name and owner pending", and no longer says the sibling project depends on `onemkl-sys` (`bonefish-sys` does not). README points to Bonefish.
- The version number is a placeholder for the portfolio PM, who allocates it at the gate.

## 0.2.1 - 2026-10-30

### Changed

- ROADMAP: the "DPC++ / SYCL APIs" non-goal is withdrawn for the portfolio. It stays out of these two crates (it needs the oneAPI DPC++ compiler) and is planned as a separate sibling project for Intel GPUs. Documentation and version only; no code change.

## 0.2.0 - 2026-05-14

### Changed (minor-bump compatibility notes)

- `Error` gained a new `AllocationFailure` variant; downstream code
  matching exhaustively on `Error` needs a wildcard arm or to
  enumerate the new variant.
- `BasicRng` gained four new variants (`Sobol`, `Niederreiter`,
  `Ars5`, `Rdrand`); same caveat for exhaustive matches.
- `MKL_ENABLE_AVX`, `MKL_ENABLE_AVX512_MIC`, and
  `MKL_ENABLE_AVX512_MIC_E1` no longer appear in the `IsaLevel`
  enum — newer MKL releases (≥ 2026) drop the corresponding
  symbols, and the values they accepted are subsumed by `Sse42` /
  `Avx2` / `Avx512` / `Avx10`.

### Added — second-pass ML-focused surface expansion

- **`Stream` distribution coverage** rounded out with the
  distributions ML training and inference actually reach for:
  - `gumbel(out, displacement, scale)` (min-Gumbel — negate samples
    for the max-Gumbel used in Gumbel-softmax),
  - `laplace(out, displacement, scale)`,
  - `rayleigh(out, displacement, scale)`,
  - `chi_square(out, ndf)`,
  - `gaussian_mv(out, dimen, MatrixStorage, mean, cov_factor)` for
    multivariate normal sampling from a Cholesky factor (full /
    packed / diagonal storage),
  - `truncated_normal(out, mean, sigma, lo, hi)` (and
    `truncated_normal_f32`) via rejection sampling on top of
    `gaussian`,
  - `dirichlet(out, alpha)` derived from a sequence of `gamma`
    draws,
  - `multinomial(out, ntrial, p)` and `categorical(out, p)` for
    LLM-style token sampling,
  - `geometric(out, p)`, `hypergeometric(out, lot, marked, sample)`,
    `neg_binomial(out, a, p)`.
- **Quasi-random sequences** via `Stream::quasi_random(brng,
  dimension)` plus new `BasicRng::Sobol`, `Niederreiter`, `Ars5`,
  `Rdrand` variants. Standard primitive for Quasi-Monte Carlo
  integration, Bayesian optimization, and MC dropout uncertainty.
- **Mixed-precision iterative-refinement LAPACK** in
  `lapack`: `iter_refine_gesv_f64`, `iter_refine_posv_f64`,
  `iter_refine_gesv_c64`, `iter_refine_posv_c64` (wrap
  `LAPACKE_dsgesv` / `LAPACKE_dsposv` / `LAPACKE_zcgesv` /
  `LAPACKE_zcposv`). Returns an `IterRefineOutcome` with the
  refinement iteration count.
- **Sparse SYRK and SYPR** on `SparseMatrix<T>`: `syrk(op)` for
  `op(A)·op(A)ᵀ` (Gram matrix / normal equations) and
  `sypr(op, B, descrB)` for the symmetric triple product
  `op(A)·B·op(A)ᵀ` (GNN message passing, reduced-order models).
- **Summary-stats preprocessing extensions** on `SummaryStats<T>`:
  `quantiles(orders)`, `stream_quantiles(orders, params)`
  (Zhang-Wang streaming estimator), `robust_covariance_tbs(params)`
  (Tukey biweight S-estimator), `outliers_bacon(params)` for
  per-observation outlier weights, and `impute_missing(params)` for
  NaN-marked imputation. Sensible defaults via
  `SummaryStats::<f64>::bacon_default_params()` and
  `SummaryStats::<f64>::impute_default_params()`.
- **Service module expansion**:
  - `AlignedBuffer<T>` — owned `MKL_malloc`-backed buffer with
    user-specified alignment, `Deref<Target = [T]>` access, freed
    via `MKL_free` on drop. SIMD-aligned (e.g. 64-byte for
    AVX-512) tensor backing without copies.
  - `ThreadCountGuard` — RAII guard that sets the calling thread's
    local MKL thread count on construction and restores it on
    drop.
  - `enable_instructions(IsaLevel)` with an `IsaLevel` enum
    covering `Sse42`, `Avx`, `Avx2`, `Avx512`, `Avx10`, and all
    the enabled-N variants — wraps `MKL_Enable_Instructions`.
  - `cpu_clocks()`, `cpu_frequency_ghz()`, `max_cpu_frequency_ghz()`
    for per-call timing diagnostics.
- **New error variant** `Error::AllocationFailure` for
  `MKL_malloc`-style failures.

### Added — every major oneMKL domain has a safe Rust wrapper

#### BLAS and BLAS-like extensions (`blas`)

- Level 1 / 2 / 3 covering all universal, real-only, and complex-only
  routines, including banded and packed variants.
- Strided batched: `gemm_batch_strided`, `trsm_batch_strided`,
  `gemv_batch_strided`, `dgmm_batch_strided`, `axpy_batch_strided`,
  `copy_batch_strided`, `gemm3m_batch_strided`.
- Pointer-array batched: `gemm_batch`, `gemv_batch`, `trsm_batch`
  with matching `*BatchGroup` structs that flatten group descriptors
  into MKL's parallel parameter arrays.
- Mixed-precision GEMMs in `blas::mixed_precision`: `gemm_bf16_f32`,
  `gemm_f16_f32`, `gemm_e5m2_f32`, `gemm_e4m3_f32`, `gemm_s8u8_s32`,
  `gemm_s16_s32`, `hgemm`. Inputs are taken as raw bit-pattern
  slices so callers can integrate with whatever bf16 / fp16 / fp8
  representation they already use.
- JIT GEMM in `blas::jit`: `JitGemm<T>` for f32 / f64 / Complex32 /
  Complex64 with runtime-specialized small kernels and
  `JitStatus::{Compiled, Fallback}` introspection.
- Pack / compute API in `blas::packed`: `PackedMatrix<T>` plus
  `gemm_compute_packed_a` / `gemm_compute_packed_b` for amortized
  packing across many GEMMs.
- Compact BLAS in `blas::compact`: SIMD-packed batched GEMM with
  `gepack` / `geunpack` / `gemm` / `get_size` and three pack widths
  (Sse / Avx / Avx512).
- Out-of-place / in-place transpose, axpby, omatcopy, omatadd.

#### LAPACK (`lapack`)

- Linear solvers (general, symmetric, Hermitian, banded, packed, PD).
- Least squares and QR (`gels`, `gelsd`, `geqrf`, `gelqf`, `orgqr`,
  `ungqr`).
- Eigenvalue and SVD (`syev`, `heev`, `geev_real`, `geev_complex`,
  `gesdd`, `gesvd`).
- Generalized eigenvalue (`sygv`, `hegv`, `ggev_real`,
  `ggev_complex`).
- Divide-and-conquer + RRR (`syevd`, `heevd`, `syevr`, `heevr`).
- Auxiliary: `lacpy`, `lange`, `gecon`, `laswp`, `larfg`.

#### Sparse BLAS (`sparse`)

- `SparseMatrix<T>` with constructors for CSR / COO / CSC / BSR.
- Inspector-Executor: `mv` / `mm` / `trsv` / `optimize`, plus
  Sparse QR via `qr_factor` / `qr_solve`.
- Format conversion (`copy`, `convert_csr`, `convert_csc`,
  `convert_coo`, `convert_bsr`, `order`) and analysis-stage hint
  setters (`set_mv_hint`, `set_mm_hint`, `set_sv_hint`).
- Sparse arithmetic: `add`, `spmm`, `spmmd`.
- Iterative-solver primitives: `dot_mv`, `symgs`, `symgs_mv`.

#### Sparse direct solvers

- PARDISO (`pardiso`): factor + solve, multi-RHS, cached
  factorization, diagonal extraction (`with_diagonal_enabled` /
  `get_diagonal`), save / restore handle (`save_handle` /
  `load_handle` / `delete_handle_files`), low-level `export`,
  user-supplied `perm`, low-level `pardiso_64_raw` for the always-
  64-bit interface, `set_pardiso_pivot_callback` for custom pivot
  logic.
- DSS (`dss`): real and complex factor / solve, multi-RHS, statistics
  readout (timing / memory / determinant / inertia).

#### Iterative sparse solvers (`iss`)

- Closure-driven `solve_cg` and `solve_fgmres` returning `IssResult`
  with iterations, residual norms, and `IssStopReason`.
- User-driven step-by-step API: `CgSession` / `FgmresSession` with
  `step` returning an `RciAction` enum.

#### Preconditioners (`preconditioners`)

- ILU0 and ILUT factorization plus `apply_ilu` for the two-step
  triangular solve.

#### Eigensolvers (`feast`)

- Dense / CSR / banded standard problems and dense generalized
  problem, real and complex.

#### FFT (`fft`)

- 1-D / 2-D / 3-D / N-D complex transforms, in-place and out-of-place.
- Real-input transforms with CCE storage.
- Configurable forward / backward scaling (`*_with_scales`
  constructors).

#### Random numbers and statistics (`rng`)

- VSL `Stream` RAII type with all major BRNGs.
- Continuous and discrete distributions.
- 1-D convolution / correlation tasks (`Conv1d`, `Corr1d`).
- Multivariate summary statistics (`SummaryStats` with `mean`,
  `variance`, `min`, `max`, `sum`).

#### Vector math (`vm`)

- All major function families (transcendental, trigonometric,
  hyperbolic, special, rounding) with elementwise vector operations.

#### Optimization (`optim`)

- Trust-region nonlinear least squares (`solve_trnls`) and bound-
  constrained variant (`solve_trnls_bounded`), each returning
  `TrnlsResult` with iteration count, stopping criterion, and
  initial / final residual norms.
- RCI numerical Jacobian and low-level `djacobi_with_callback` for
  direct-FFI Jacobian computation.

#### Data fitting (`data_fitting`)

- Cubic splines: natural, Bessel, Akima, Hermite — all interpolatable
  and integrable.

#### PDE support (`pde`)

- Trigonometric transforms: Sine, Cosine, four staggered variants
  for spectral PDE solvers.

#### Service (`service`)

- Version, threading, memory, verbose mode, finalize.

### Build infrastructure

- Two-crate workspace: `onemkl-sys` (raw bindgen FFI) and `onemkl`
  (safe wrappers).
- Per-domain Cargo features all enabled by default; downstream
  callers can `default-features = false` and pick exactly what they
  need.
- Build configuration features: `lp64` / `ilp64` (interface),
  `threading-sequential` / `threading-intel-openmp` /
  `threading-tbb`, `link-dynamic` / `link-static`.
- MPI-gated domains as opt-in features: `cluster-sparse-solver`,
  `cdft`, `scalapack`, `blacs`, `pblas`.

### Documentation

- Crate-level rustdoc with a use-case-oriented orientation guide.
- Per-module rustdoc on every public domain.
- `examples/` directory with five runnable programs:
  `solve_linear_system`, `sparse_cg`, `fft_signal`,
  `quantized_inference`, `transformer_qkv_projection`.
- README with feature table and minimal example.
- `ROADMAP.md` tracking coverage tiers per domain.

### Known gaps

These are wrapped at the FFI level via `onemkl::sys` but lack a
high-level safe wrapper:

- FFT alternate real-input storage formats (CCS / PACK / PERM) —
  CCE format is fully wrapped.
- PARDISO Schur complement extraction — the `export` method is
  exposed but the exact MKL call sequence for Schur output is
  finicky enough that it currently crashes inside MKL.
- High-level `Pardiso64<T>` state-managed wrapper — only the
  low-level `pardiso_64_raw` is wrapped.
- Sparse `mkl_sparse_sp2m` (with request flags) and `_syprd`.
