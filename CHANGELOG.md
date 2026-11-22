# Changelog

All notable changes to this project are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## 0.3.5 - 2026-11-22

### Added

- `blas::compact`: `trsm`, `potrf`, `getrfnp` and `geqrf` (`mkl_?trsm_compact`, `mkl_?potrf_compact`, `mkl_?getrfnp_compact`, `mkl_?geqrf_compact`) for `f32`/`f64`, with buffer-length checks against `get_size`. Test `blas_compact_factor` (Linux x86_64, LP64, sequential, oneMKL 2026.2.1, AVX pack, column-major, `nm` 1-2, 2x2 matrices): hand-computed lower-triangular solve, Cholesky and no-pivot LU match at 1e-12; 2x2 QR matches |R| entries at 1e-12 (tau contents unchecked). Complex types not wrapped; row-major, larger sizes, SSE/AVX-512 packs, Windows, macOS, OpenMP/TBB, ILP64 not exercised.
- `pde::Helmholtz2D` (`d_/s_helmholtz_2d`): 2-D Cartesian Helmholtz/Poisson solver `Δu - q·u = f`, Dirichlet sides only (Neumann/periodic are rejected until verified). Test `pde_trig` (Linux x86_64, LP64, sequential, oneMKL 2026.2.1 pip): sin·sin solution on the unit square converges at second order (error ratio 3.5-4.5 from n=32 to 64, q=0 and q=3), and u = x²-y²+1 with non-zero Dirichlet data on a 2x1 rectangle, nx=16, ny=8, is reproduced to 1e-9. Measured MKL behaviour the wrapper corrects for: the library returns `(-Δ+q)⁻¹ f / hy²` (opposite sign and scaled by 1/hy² against the documented convention, fitted to 10 digits on 8 domain/q combinations at n=8) and applies no Dirichlet boundary lift (interior stayed exactly 0 for every placement of the data tried); the wrapper rescales by `-hy²` and moves boundary data to the right-hand side. Not exercised: `f32`, `d_helmholtz_3d`, spherical solvers, Neumann/periodic, other MKL versions, Windows, macOS, OpenMP/TBB, ILP64.

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
