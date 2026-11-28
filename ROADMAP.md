# Roadmap

This document tracks the current state of the wrappers and what's planned
next. As planned items become reality they collapse into the *Status*
section's table; the *Plans* section then shrinks accordingly.

## Ownership

Bonefish owns this project. The ruling, as relayed: "Bonefish owns oneMKL/oneAPI in our software ecosystem." (CireSnave, board 329, ruled 2026-10-30, recorded by the portfolio PM). That covers these two crates (`onemkl-sys`, `onemkl`: the CPU interface, C/LAPACKE, no DPC++ compiler needed) and the SYCL/DPC++ interface (Intel GPU and CPU through `oneapi::mkl::*`), which is built in [`ThinkersJournal/bonefish`](https://github.com/ThinkersJournal/bonefish). CPUFish consumes the crates from crates.io. Planned work for the CPU crates stays in this file; anything that needs `icpx -fsycl` belongs in Bonefish.

## Coverage tiers

Each oneMKL domain is rated against three tiers. We use these as the
acceptance criteria for the milestones below.

- **MVP** — the core most-used routines work, with a safe API and real
  + complex coverage where applicable. Enough that someone wanting "the
  obvious thing" from this domain finds it.
- **Common** — every standard routine in the domain is exposed, including
  banded / packed / generalized / expert variants where they exist.
- **Comprehensive** — every entry point oneMKL ships under the domain is
  wrapped, including obscure / specialized variants (compact, JIT,
  mixed-precision, MPI, etc.).

A tier is "done" only when (a) the public API exists, (b) it returns
`Result` and validates inputs at boundaries, (c) every wrapped routine
has at least one test, and (d) the docs name the underlying MKL
function(s).

## Status

| Domain | Tier | Notes |
| --- | --- | --- |
| BLAS L1/L2/L3 | Common | All universal + real-only + complex-only routines, including banded and packed |
| BLAS-like extensions | Common | `axpby`, `imatcopy`/`omatcopy`/`omatcopy2`/`omatadd`, batched (strided) `gemm`/`trsm`/`gemv`/`dgmm`/`axpy`/`copy`/`gemm3m`; pointer-array `gemm_batch` / `gemv_batch` / `trsm_batch` |
| LAPACK | Common | Linear solve, QR, LS, eigenvalue (incl. RRR + D&C), SVD, banded, packed, generalized |
| Sparse BLAS (IE) | Common+ | CSR / COO / CSC / BSR construction; `mv`/`mm`/`trsv`/`optimize` + Sparse QR factor / solve; copy / convert / order / mv-mm-sv hint setters; sparse arithmetic (`add`, `spmm`, `spmmd`); CG-style `dot_mv`; Gauss–Seidel sweep (`symgs`, `symgs_mv`) |
| PARDISO | Common | Factor + solve, multi-RHS, cached factorization, diagonal extraction, save/restore handle, low-level `export`, user `perm`, `pardiso_64_raw`, custom pivot callback |
| DSS | Common | Real + complex (single + double precision) factor / solve, multi-RHS, statistics (timing / memory / determinant / inertia) |
| ISS (CG, FGMRES) | Common | Closure-driven mat-vec; preconditioned CG; `IssResult` with iterations / residual norms / stop reason; full RCI surface via `CgSession` / `FgmresSession` |
| Preconditioners | Common | ILU0, ILUT, plus `apply_ilu` for two-step triangular solve. ILU0 / ILUT are MKL's only RCI preconditioners — no SSOR / Jacobi / Chebyshev exposed. |
| FEAST | Common | Dense / CSR / banded standard problems; dense generalized problem; real + complex; RCI still TODO |
| VM (Vector Math) | Common | All major function families |
| RNG (VSL) | Common | RAII `Stream` + 8 continuous + 4 discrete distributions; 1-D convolution / correlation tasks; summary statistics (mean / variance / min / max / sum) |
| FFT (DFTI) | Common | 1-D / 2-D / 3-D / N-D complex; real-input (CCE) variants; configurable forward / backward scaling |
| Data fitting | Common | Natural / Bessel / Akima / Hermite cubic splines, with interpolate / integrate |
| Optimization | Common | TRNLS, TRNLSPBC with initial / final residual norms; RCI numerical Jacobian; low-level `djacobi_with_callback` direct-FFI Jacobian |
| PDE Support | MVP+ | DCT / DST trigonometric transforms (sine, cosine, four staggered variants) for spectral PDE solvers |
| Service | Common | Version, threading, memory, verbose, finalize, CBWR |

**What the Notes column proves.** It lists what the public API exposes
(check with `cargo doc -p onemkl --no-deps`). Wording such as "All", "full" and
"All major function families" is **UNVERIFIED**: no per-routine coverage tool
has been run, so tier criterion (c) above ("every wrapped routine has at least
one test") is **UNVERIFIED** for every row. The most each row's tests show is
that the named test files pass (`cargo test -p onemkl --test <name>`):

| Domain | Test files (`--test`) |
| --- | --- |
| BLAS, extensions | `blas_level1` `blas_level2_3` `blas_level2_extended` `blas_level3_extended` `blas_packed` `blas_batched` `blas_gemm_batch` `blas_gemv_trsm_batch` `blas_extensions` `blas_compact` `blas_jit` `blas_mixed_precision` `blas_f16_typed` (needs `--features half`) |
| LAPACK | `lapack` `lapack_auxiliary` `lapack_banded` `lapack_generalized` `lapack_iter_refine` `lapack_packed` `lapack_rrr` |
| Sparse BLAS | `sparse` `sparse_arithmetic` `sparse_convert` `sparse_formats` `sparse_iterative` `sparse_qr` |
| PARDISO / DSS | `pardiso` / `dss` |
| ISS / Preconditioners | `iss` `iss_rci` / `preconditioners` |
| FEAST | `feast` `feast_extras` |
| VM | `vm` (accuracy mode: `vml_accuracy_mode`) |
| RNG, convolution, stats | `rng` `rng_convolution` `rng_summary_stats` |
| FFT | `fft` `fft_scaling` |
| Data fitting / Optimization / PDE / Service | `data_fitting` `data_fitting_splines` / `optim` `optim_jacobix` `optim_numerical_jacobian` / `pde_trig` / `service` `service_cbwr` `service_xerbla` `service_progress` |

Platforms: all of the above ran on Linux x86_64, LP64, oneMKL 2026.2.1 only.
Windows, macOS, ILP64 and the OpenMP / TBB threading layers are
**UNVERIFIED**.

Measured 2026-11-18 on Linux x86_64 with oneMKL 2026.2.1, default features:
373 tests pass, 0 fail, 5 ignored. Command: `cargo test --workspace
--no-fail-fast`, then sum the `test result:` lines (passed / failed /
ignored). The count moves with every test added; re-measure before quoting it.

## Plans

### 0.2.0 — All domains at Common

Move every domain currently at MVP up to Common. Specifically:

- **BLAS-like extensions**: ~~pointer-array `?gemm_batch`,
  `?gemv_batch`, `?trsm_batch`~~ (done — `gemm_batch` /
  `gemv_batch` / `trsm_batch` plus matching `*BatchGroup` structs);
  pack/compute API (`?gemm_pack`, `?gemm_compute`), JIT
  (`mkl_jit_create_?gemm`).
- **LAPACK**: banded variants (`?gbsv`, `?gbtrf`/`?gbtrs`,
  `?pbsv`/`?pbtrf`/`?pbtrs`), packed variants (`?spsv`/`?sptrf`,
  `?ppsv`/`?pptrf`), generalized eigenvalue (`?ggev`, `?sygv`,
  `?hegv`), expert / RRR drivers (`?gesvx`, `?syevr`, `?heevr`),
  utility routines (`?lange`, `?gecon`, `?syrfs`).
- **Sparse BLAS**: ~~COO, CSC, BSR formats~~ (done — `from_coo` /
  `from_csc` / `from_bsr` constructors); ~~`mkl_sparse_copy`,
  `mkl_sparse_convert_*`, `mkl_sparse_order`; the analysis-stage hint
  setters (`mkl_sparse_set_mv_hint`, etc.)~~ (done — `copy`,
  `convert_{csr,csc,coo,bsr}`, `order`, `set_{mv,mm,sv}_hint`).
- **PARDISO**: ~~`pardiso_getdiag`, save/restore handle pair,
  `pardiso_export`, `perm` parameter, `pardiso_64`,
  `mkl_pardiso_pivot`~~ (all done — `with_diagonal_enabled` /
  `get_diagonal`, `save_handle` / `load_handle` /
  `delete_handle_files`, low-level `export`, `set_perm` / `perm`,
  `pardiso_64_raw`, `set_pardiso_pivot_callback`). The Schur
  extraction call sequence via `export` still needs to be nailed
  down (currently crashes inside MKL). High-level `Pardiso64<T>`
  state-managed wrapper and Schur fix remain as polish.
- **DSS**: ~~complex matrices, statistics readout~~ (done — complex
  factor/solve already covered, plus `factor_time` / `peak_memory_kb`
  / `determinant` / `inertia` etc.); generalized symmetric options.
- **ISS**: ~~expose `?cg_get`/`?fgmres_get` reasoning codes; full RCI
  surface for users who want to drive their own loop~~ (done —
  `IssResult` plus `CgSession` / `FgmresSession` with `step` →
  `RciAction { Converged, NeedMatVec, NeedPrecondition,
  StoppingTest, Other }`).
- **FEAST**: ~~CSR-input drivers (`?feast_scsrev`, `?feast_hcsrev`),
  banded (`?feast_sbev`, `?feast_hbev`), dense generalized
  (`?feast_sygv` / `?feast_hegv`)~~ (done — `eigh_real_csr` /
  `eigh_complex_csr` / `eigh_real_banded` / `eigh_complex_banded` /
  `gen_eigh_real_dense` / `gen_eigh_complex_dense`); CSR
  generalized (`?feast_scsrgv` / `?feast_hcsrgv`); banded
  generalized (`?feast_sbgv` / `?feast_hbgv`); RCI variants.
- **RNG**: ~~convolution and correlation tasks (VSL_*Conv*,
  VSL_*Corr*)~~ (done — `Conv1d` / `Corr1d` plus `convolve_1d` /
  `correlate_1d` free functions); ~~summary statistics
  (mean/variance)~~ (done — `SummaryStats::{mean, variance, min, max,
  sum}`); covariance / order statistics / quantiles; remaining BRNGs
  and method codes.
- **FFT**: ~~multi-dimensional, real-input variants, configurable
  scaling~~ (done — `complex_nd` / `real_nd` plus `*_with_scales`
  constructors); alternate real-input storage formats (CCS, PACK,
  PERM); per-axis stride / distance config.
- **Data fitting**: ~~Hermite, Bessel, Akima cubic splines;
  user-supplied derivatives~~ (done — `CubicSpline1d::{bessel,
  akima, hermite}`); lookup / linear / step-function spline types;
  `?SearchCells1D` and the cell-based interpolators.

### 0.3.0 — Comprehensive coverage of three priority domains

After 0.2, push BLAS, LAPACK, and Sparse to Comprehensive — these
account for the vast majority of downstream use:

- **BLAS Comprehensive**: ~~mixed-precision GEMMs (`bf16bf16f32`,
  `f16f16f32`, `e5m2e5m2f32`, `e4m3e4m3f32`)~~ (done — `gemm_bf16_f32`
  / `gemm_f16_f32` / `gemm_e5m2_f32` / `gemm_e4m3_f32` plus
  `gemm_s8u8_s32` / `gemm_s16_s32` / `hgemm` in
  `blas::mixed_precision`); ~~JIT GEMM lifecycle~~ (done — `JitGemm<T>`
  in `blas::jit`); ~~packed compute (`cblas_gemm_*_pack`,
  `cblas_gemm_*_compute`)~~ (done — `PackedMatrix<T>` plus
  `gemm_compute_packed_{a,b}` in `blas::packed`); ~~compact BLAS
  GEMM + pack / unpack lifecycle~~ (done — `blas::compact`); ~~other
  compact kernels (`?trsm_compact`, `?potrf_compact`,
  `?getrfnp_compact`, `?geqrf_compact`, `?getrinp_compact`)~~ (done, f32/f64).
- **LAPACK Comprehensive**: ~~core auxiliary routines (`?lacpy`,
  `?lange`, `?gecon`, `?larfg`, `?laswp`)~~ (done in
  `lapack::{lacpy, lange, gecon, laswp, larfg}`); remaining
  auxiliary set (~~`?lacgv`, `?lacrm`~~ (done), ~~`?syconv`~~ (done — `lapack::{sytrf, syconv}`), ~~`?larft`, `?larfb`~~ (done — `lapack::{larft, larfb}`),
  etc.), Netlib-compatibility additions, `?tppack` / `?tpunpack`.
- **Sparse Comprehensive**: ~~matrix manipulation
  (`mkl_sparse_*_add`, `mkl_sparse_*_spmm`,
  `mkl_sparse_*_spmmd`)~~ (done — `SparseMatrix::{add, spmm,
  spmmd}`); ~~`_dotmv`, `_symgs`, `_symgs_mv`~~ (done —
  `SparseMatrix::{dot_mv, symgs, symgs_mv}`); `mkl_sparse_sp2m`
  (with request flags), `_syprd` (`A · diag · Aᵀ`),
  inspector-executor analysis routines, full export API.

### 0.4.0 — Domain-completion sweep

Push all remaining MVP domains to Common+ and start filling in the
specialized ones:

- **Optimization**: ~~`?trnlsp_get` with full return shape (initial /
  final residual norms), `?jacobi` direct caller~~ (done — residual
  norms in `TrnlsResult`, `djacobi_with_callback`); ~~`?trnlsp_check`~~ (done — `TrnlsSession::check`), ~~`?jacobix` (Jacobian
  with central differences)~~ (done — `numerical_jacobian_central`, double only), ~~session API for step-by-step driving~~ (done — `TrnlsSession`, `TrnlsBoundedSession`).
- **Service**: ~~CBWR (`mkl_cbwr_*`)~~ (done — `cbwr_set` / `cbwr_set_strict` / `cbwr_branch` / `cbwr_strict` / `cbwr_auto_branch`), ~~`mkl_set_progress`,
  `mkl_set_xerbla`~~ (done — `set_progress_handler` / `set_xerbla_handler`), ~~`mkl_set_pardiso_pivot`~~ (done — `pardiso::set_pardiso_pivot_callback`, raw unsafe form),
  ~~`mkl_enable_instructions`~~ (done — `enable_instructions`), ~~single-dynamic-library control~~ (done — `link-rt`, `set_interface_layer` / `set_threading_layer`; Linux only measured).
- **PDE Support**: ~~trigonometric transforms (sine / cosine /
  staggered)~~ (done — `TrigTransform<T>` in `pde`); ~~Cartesian
  Poisson / Helmholtz (2-D / 3-D, Dirichlet / Neumann / periodic)~~
  (done — `Helmholtz2D` / `Helmholtz3D`); ~~periodic spherical
  Helmholtz~~ (done — `SphericalHelmholtz`, full sphere); ~~non-periodic
  spherical (`?_sph_np`)~~ (done — `SphericalPatchHelmholtz`, Dirichlet patch away from the poles); polar caps and non-Dirichlet patch sides remain.
- **Compact BLAS / LAPACK**: ~~full set (`mkl_?gepack_compact`,
  `mkl_?geunpack_compact`, `mkl_?get_size_compact`, `*_potrf`,
  `*_getrfnp`, `*_geqrf`, `*_getrinp`)~~ (done, f32/f64).

### 0.5.0 — All domains Comprehensive (single-node)

Every non-MPI domain at Comprehensive:

- **FFT Comprehensive**: every storage format, transform type, and
  config flag DFTI exposes.
  Done so far: 1-D packed real storage (CCS / PACK / PERM) via `PackedRealFftPlan`; 1-D batched complex with distance / stride via `BatchedFftPlan`, 1-D batched real via `BatchedRealFftPlan`; N-D strided complex (`StridedFftPlan`), N-D strided real out-of-place (`StridedRealFftPlan`) and N-D in-place real with padded CCE (`InPlaceRealFftPlan`); N-D packed (CCS / PACK / PERM) is not available: probed 2026-11-25 on `Intel(R) oneAPI Math Kernel Library Version 2026.2-Product Build 20260915 for Intel(R) 64 architecture applications` (`MKL_Get_Version_String`), a double-precision real descriptor of dims `[4,6]` and `[3,4,5]` rejected `DftiSetValue(DFTI_PACKED_FORMAT, {PACK,PERM,CCS}_FORMAT)` and `DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_REAL)` with status 3 (`DFTI_INCONSISTENT_CONFIGURATION`), each set before and after the other, with `DFTI_INPLACE` and `DFTI_NOT_INPLACE`; the same calls on a 1-D `[6]` descriptor returned 0. This is an upstream limit of that build; retry when the pinned oneMKL changes to a newer release. Whether to report it upstream is a person's decision; strided / interleaved real batches (`StridedBatchedRealFftPlan`); per-plan thread limit (`FftPlan::complex_nd_with_thread_limit`, `RealFftPlan::real_nd_with_thread_limit`, `thread_limit()`); workspace mode (`FftWorkspace`, `FftPlan::complex_nd_with_workspace`, `RealFftPlan::real_nd_with_workspace`, `workspace()`); `DFTI_ORDERING` not exposed: probed 2026-11-26 on oneMKL 2026.2 (`tools/size-probes/fft_flags.c`), `DFTI_BACKWARD_SCRAMBLED` is accepted but gave results identical to `DFTI_ORDERED` for 1-D n=64 and 2-D 8x16 complex plans, so there is nothing observable to wrap yet.
- **Data Fitting Comprehensive**: integration call-back form, cell
  search call-backs, all spline editors.
  Hyman (`CubicSpline1d::hyman`): wrapped with `DF_BC_FREE_END`. Measured 2026-11-26 on oneMKL 2026.2 (`tools/size-probes/df_probe3.c`, 8 monotone plateau knots, worst downward step between samples): `DF_BC_FREE_END`, end first derivatives and end second derivatives 0.0 (monotone); `DF_BC_NOT_A_KNOT` 0.0140 (not monotone; natural cubic 0.0243); `DF_NO_BC` fails with -1014 although the Intel reference lists it as supported. An earlier note here called Hyman unsupported because it was probed with not-a-knot only; that conclusion is withdrawn.
  Quadratics (`QuadraticSpline1d`): default quadratic with `DF_BC_Q_VAL` (one value; `DF_NO_BC` gives -1014) and Subbotin with `DF_BC_FREE_END` plus `DF_IC_Q_KNOT` (n + 1 knots; a missing knot array gives -1012). Measured: the Subbotin spline writes `order * n` coefficients, not the documented `order * (n - 1)` (`tools/size-probes/df_coeff_extent.c`), and does not pass through the breakpoints (`docs/DATA-FITTING-FACTS.md`).
- **`InterpolateEx1D` (data fitting): wrapped for function values (`interpolate_with_callbacks`, `interpolate_with_search_callback`); derivative orders are not.** Measured in `tools/size-probes/df_interp_cb.c`: with two derivative orders the call-backs' blocks came back concatenated per region, not a plain per-site layout; it needs its own probe before a safe wrapper. Work, not declined.
- **VSL Comprehensive**: every BRNG + every method code; summary
  statistics with all weights / outliers / partial-task variants.
  Summary-statistics status (oneMKL 2026.2): weights (`SummaryStats::with_weights`, `ss7.c`) and partial covariance (`partial_covariance`, `ss9.c`) are wrapped and verified. **Pooled / group covariance is not wrapped and cannot be verified yet** (`ss8.c`): `vsl?SSEditPooledCovariance` returns 0, but the compute step then fails with -4044 (`VSL_SS_ERROR_BAD_STORAGE_ADDR`) because the pooled / group covariance storage format cannot be set: `vsl?SSEditTask` with `VSL_SS_ED_POOLED_COV_STORAGE`, `VSL_SS_ED_GROUP_COV_STORAGE` and the control `VSL_SS_ED_COV_STORAGE` all return -3 (`VSL_ERROR_BADARGS`), while the same call form works for `VSL_SS_ED_MEAN`. Retry on a newer oneMKL or if Intel documents another way to set that storage. Partial-task `indices` argument of `vsl?SSNewTask` and the MI (`impute_missing` simulation buffer) rows stay open.
- **FEAST Comprehensive**: full RCI interface.
- **f16 support** (ranked below the rows above; additive, minor version):
  `impl Scalar` and `RealScalar` for `half::f16` behind an optional `half`
  feature (`scalar.rs` is sealed, so downstream crates cannot), so
  `level3::gemm` accepts f16; optionally typed `half::f16` wrappers over the
  existing `mixed_precision::{gemm_f16_f32, hgemm}`, which already exist
  but take raw `u16` bits (`hgemm` needs `cols * ld` elements per operand:
  document it). Reason: Fuel lost its f16 hgemm when its old oneMKL code
  was removed. Source: CireSnave 2026-11-22.

### Known gaps found by probing (2026-11-27; rows, not builds)

- **ILP64 for `onemkl`: done (0.6.1 to 0.6.5).** `--features ilp64` builds the safe wrappers against bindings generated from the ILP64 headers and the full suite passes against the `mkl_intel_ilp64` libraries on the rig; CI has a row for it. Package plan and the counts: `docs/ILP64-DESIGN.md`.
- **Summary Statistics wrappers that never worked: fixed (2026-11-27).** `impute_missing` (0.5.0) and `stream_quantiles` (`VSL_SS_METHOD_FAST` refused with -4029; now ZW with one accuracy parameter) were broken as shipped; the Application Notes were not needed, the public 2026.0 C reference PDF plus probes `ss4.c`, `ss5.c`, `ss6.c` gave the layouts. Still unmeasured: `impute_missing` initial estimates and priors, draw quality, storage orders other than variable-major; BACON flagged set against a known answer; the partial-task, pooled-covariance, covariance/correlation and weights variants of Summary Statistics are not wrapped.

### CI lint gates (unscheduled; rows, not builds)

- **CI fmt / clippy jobs: done (2026-11-28).** `cargo fmt --check` is a job. Clippy is `-D warnings` over `--workspace --all-targets` in two jobs, both REQUIRED by branch protection on `main` (PM read back 2026-11-28: 4 contexts, `enforcement_level` non_admins): `cargo clippy (Linux, -D warnings)` (real oneAPI headers) and `cargo clippy (target x86_64-pc-windows-msvc, cross from Linux, -D warnings)` (a cross-TARGET lint from a Linux runner with vendored bindings: `cfg(windows)` code is compiled but nothing is linked; a real Windows host is UNMEASURED by CI). Controls (scratch PRs #50/#51, closed): a lint planted in `cfg(windows)` code failed only the cross job, one in `cfg(not(windows))` code failed only the Linux job. The old report-only job (lib only, no `-D warnings`, `continue-on-error`: its green said nothing) is deleted. macOS: cross-TARGET clippy jobs for `x86_64-apple-darwin` and `aarch64-apple-darwin` (same method, from a Linux runner; a real macOS host is UNMEASURED) were added 2026-11-28 and are NOT required yet: the PM adds their contexts after one green main run (the aarch64 command measured locally, exit 0; x86_64 first measured by CI).
- **Clippy cleanup: done (2026-11-27).** The earlier count (303 on lane/f16 at df1d17c, mostly `missing_safety_doc`) was stale: measured at `origin/main` 02b187c with `--keep-going` over `--workspace --all-targets` it was **28 findings** on Windows (11 lints: `useless_vec` 6, `manual_ignore_case_cmp` 4, `useless_conversion` 3, `unnecessary_cast` 3, `manual_is_multiple_of` 3, `neg_cmp_op_on_partial_ord` 2, `needless_range_loop` 2, `byte_char_slices` 2, `approx_constant` 1, `manual_range_contains` 1, `neg_multiply` 1). Fixed: mechanical ones fixed in place, `neg_cmp_op_on_partial_ord` rewritten with `partial_cmp` (same NaN-rejecting meaning), the platform- and interface-dependent `unnecessary_cast` / `useless_conversion` sites (bindgen enum type differs Linux/Windows; `MKL_INT` is i32 under lp64) and one `approx_constant` test literal carry an item-level `#[allow]` with a reason. Result: `cargo clippy --workspace --all-targets -- -D warnings` exits 0 on WSL Ubuntu-24.04 (rustc / clippy 1.99.1) and on Windows (the only remaining output there is cargo's `package.homepage is redundant` manifest warning, not a lint). The old CI clippy job was report-only (lib only, no `-D warnings`, `continue-on-error`), so its green said nothing; it is deleted and replaced by the two required jobs in the row above.

- **W1: pre-existing lint allows need a reason each (counted, item-level).**
  Measured 2026-11-26 on `origin/main` at 3b18f67f (rustc / clippy 1.99.1,
  WSL), by `git grep` and by `cargo clippy -p onemkl -- --force-warn
  clippy::too_many_arguments` (force-warn overrides the source allows):
  - **238 bare `#[allow(clippy::too_many_arguments)]`** (plus one combined
    `allow(missing_docs, clippy::missing_safety_doc, clippy::too_many_arguments)`),
    none with a `reason`. All are hand-written: 236 in `onemkl/src`
    (`blas/scalar.rs` 47, `lapack/scalar.rs` 44, `lapack/mod.rs` 27, `feast.rs`
    16, `blas/level2.rs` 14, `blas/compact.rs` 12, `blas/level3.rs` 11,
    `blas/extensions.rs` 11, `blas/mixed_precision.rs` 9, `sparse/scalar.rs` 8,
    `blas/packed.rs` 7, `rng/summary_stats.rs` 6, `sparse/mod.rs` 5,
    `rng/convolution.rs` 4, `data_fitting.rs` 4, `rng/mod.rs` 3, `pardiso.rs`
    2, `blas/jit.rs` 2, and 1 each in `vm/scalar.rs`, `vm/mod.rs`, `pde.rs`,
    `fft.rs`), 2 in `onemkl/tests/blas_fortran_gemm_map.rs`, 0 in `onemkl-sys`
    (no generated file carries one).
  - **Hidden warnings:** with the allows overridden, clippy flags 198
    functions (192 pre-existing plus the 6 added by `lane/allows`), so the
    "303 findings" row above never included them: they were already silenced.
    The 238 allows outnumber the 196 flagged functions (now, with the 7
    counted waivers included) by 42. The hypothesis that trait impl methods
    carry allows that do nothing was tested and NOT supported: deleting the 14
    allows after line 1000 of `blas/scalar.rs` produced 14 warnings. The
    difference is unexplained; the table in `docs/W1-TOO-MANY-ARGUMENTS.md`
    lists the flagged functions, and finding the superfluous allows is part
    of the work (an allow whose function is not in the table).
  - **First-pass classification of the 198 flagged functions** (by the text
    of the flagged line, so unreviewed; a function is only "C-mirror" once its
    doc names the C function and the argument lists match one for one):
    124 are `unsafe fn` (raw FFI trait declarations and `unsafe` helpers:
    `blas/scalar.rs` 48, `lapack/scalar.rs` 34, `blas/compact.rs` 10,
    `feast.rs` 8, `sparse/scalar.rs` 6, and 18 in 11 other files), the likely
    C mirrors; 74 are safe `pub fn`/`fn` wrappers (`blas/level2.rs` 14,
    `lapack/mod.rs` 14, `blas/extensions.rs` 11, `blas/mixed_precision.rs` 8,
    `feast.rs` 8, `blas/packed.rs` 5, and 14 elsewhere), each to be sorted
    into C-mirror, public-surface (published API, check the tag) or
    restructurable (parameter struct; a breaking change).
  - **Classification table (first pass, 2026-11-27):**
    `docs/W1-TOO-MANY-ARGUMENTS.md` lists every flagged function (file, line,
    name, argument count, kind, class), 196 in all: 124 `unsafe fn` raw FFI
    methods (likely C mirrors; 13 of them have a doc comment that does not
    name the C function and need one before a `reason` is honest), 68 safe
    `pub fn` wrappers already published (65 since v0.1.0, 3 since v0.3.8:
    public-surface candidates, or C mirrors where they take one C call's
    arguments), and 4 private functions (`blas/packed.rs` `pack`,
    `feast.rs` `finish_real_result` and `finish_complex_result`,
    `pardiso.rs` `run_phase`: restructure or reason). 7 already carry a
    counted `reason`. Mechanical, unreviewed.
  - **Chunk 1 done (2026-11-27):** the 44 `cblas_*` and 34 `lapacke_*` raw
    trait declarations (76 bare allows; 2 of the 44 already had a reason) now
    carry a `reason` naming the C function, each checked by argument count
    against every precision variant in the installed headers
    (`tools/w1/`, details in `docs/W1-TOO-MANY-ARGUMENTS.md`). Remaining bare
    allows: 238 - 76 = 162; reasoned: 83. Chunk 2 (35 more C-mirror raw
    declarations: compact, packed, data fitting, FEAST, LAPACKE extras, RNG,
    summary statistics, sparse, VM) brings that to 127 bare and 118 reasoned.
    Chunk 3 (60 published safe wrappers, "stable public surface" with the
    tag and argument count as evidence) brings it to 67 bare and 178
    reasoned (measured by `git grep`). Chunk 4 (the rest; the PR after
    the chunks 1-3 group): 9 more C mirrors, 6 public-surface, 2 test mirrors,
    4 private functions restructured (parameter structs, no API change), 50
    no-op allows removed. Result: no bare `#[allow(clippy::too_many_arguments)]`
    left in `onemkl` / `onemkl-sys` (`git grep`); every one carries a `reason`.
    Open: whether the reasoned functions should eventually take parameter
    structs (breaking for the published ones) is a separate decision; this
    row's bare-allow work is done.
  - **Wanted:** per item, either a `reason = "mirrors the oneMKL C signature
    <name> argument for argument"` (C mirrors: most of the trait declarations
    and wrappers above), or a restructure (a parameter struct) for the rest;
    delete the allows that do nothing (trait impl methods). No mass edit
    without reading what each item is.
- **W1b: crate-level allows in `onemkl-sys/src/lib.rs` (4, pre-existing).**
  `non_upper_case_globals`, `non_camel_case_types` and `non_snake_case` are
  the standard bindgen-output allows (generated file; that is the reason).
  `dead_code`: removing it on this tree produced 0 warnings from
  `cargo check -p onemkl-sys` (the bindings are `pub`, which `dead_code` does
  not flag), so it does nothing and can be deleted; this stands as a cleanup
  row, not yet done.

- **Adopt the coderipper portfolio job.** Call
  `ciresnave/coderipper/.github/workflows/portfolio.yml@467fd259d6d062cba4c5683d310f1a6f2b66e763`
  (CODERIPPER_VERSION 0.48.1, confirmed by pm-platform 2026-11-25). Reason:
  CireSnave's CI-everywhere rule (2026-09-17); this repo has no such call, so
  its PRs carry no portfolio advisory counts.

### 0.6.0 — MPI domains

Wrap the MPI-gated domains behind their existing opt-in features:

- **ScaLAPACK**: distributed dense computational + driver routines.
- **PBLAS**: parallel BLAS L1 / L2 / L3.
- **BLACS**: communication primitives.
- **Cluster FFT**: distributed DFTI.
- **Cluster PARDISO**: distributed direct sparse solve.

### 1.0.0

- All domains at Comprehensive.
- API frozen; SemVer commitments documented.
- Every `unsafe` block reviewed; safety invariants documented at the
  trait-method level.
- Tests run against multiple oneMKL versions in CI.
- Bench suite landed for the hot-path routines.

## Out of scope

These are explicit non-goals — we do not intend to wrap them.

- **DPC++ / SYCL APIs inside these two crates.** The SYCL interface of
  oneMKL (`oneapi::mkl::*`, taking a `sycl::queue`) needs the oneAPI DPC++
  compiler (`icpx -fsycl`) and a C++ shim, which these crates deliberately
  do not require. It is **not** a non-goal for the portfolio: it is the
  sibling project **Bonefish**
  ([`ThinkersJournal/bonefish`](https://github.com/ThinkersJournal/bonefish):
  `bonefish-sys`, `bonefish`, `bonefish-sycl-emit`). Its `bonefish-sys`
  wraps `oneapi::mkl::blas` and `dft` behind `extern "C"` and does **not**
  depend on `onemkl-sys` (it needs the DPC++ compiler; these crates must
  not). The ownership of both projects is settled: see "Ownership" at the
  top of this file. Status and measurements of the SYCL work live in
  Bonefish's own `ROADMAP.md`, not here.
- **OpenMP offload variants** (`*_omp_offload.h`) — need an OpenMP target
  region at the call site, which a Rust caller cannot supply; revisit only
  if the sibling project finds a workable route.
- **Fortran-style `LAPACK_*` wrappers** when LAPACKE provides the
  equivalent — we standardize on the LAPACKE interface throughout.
- **Routines users can build themselves trivially** with the existing
  primitives (e.g., wrappers that just pre-compute argument values).
