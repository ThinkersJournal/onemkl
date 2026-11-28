# onemkl safe-API memory-safety audit

**Platform: oneMKL 2026.2 measured on WSL (Linux x86_64) ONLY.** Windows/macOS behaviour is unmeasured.
**Subject of the "before" column: `origin/main` bcf7f80 (= onemkl 0.7.0 dev). For the modules listed
below, the 0.6.8 and 0.6.11 published `.crate` sources are byte-identical to bcf7f80** (md5 compared
file by file: lapack/mod, blas/{level2,extensions,packed,compact}, sparse/mod, dss, pardiso,
preconditioners, feast, iss, util), so the "before" results hold for 0.6.8 and 0.6.11.
Earlier published versions differ per file; see "Per-version groups" (static identity only, NOT yet run).

## Method

1. Static audit: three read-only reviews (lapack, blas, sparse/dss/pardiso/ilu/feast/iss) of every safe
   `pub fn` that passes a slice / pointer / size / stride / leading dimension to FFI. Counts below are
   **static** (a reading, not a verdict).
2. Measurement: `onemkl/tests/{lapack,blas,sparse}_safe_api.rs`, one probe case per call, one process per
   case (`PROBE=<line> test --exact <fn>`), 60 s timeout, run against the **unfixed** sources and the fixed ones.
   Each probe passes a short / out-of-range input to the safe fn and requires `Err` before the FFI call.
   - `FAULT`  = process died on SIGSEGV (139) or abort (134): a **measured** memory-safety hole.
   - `ACCEPTED` = the unfixed fn returned `Ok` for the invalid input (probe failed its assertion); no
     fault was observed by this probe (a short overrun may land in mapped memory). The fn accepts input
     that makes the FFI read/write outside the buffer; **not counted as a measured hole**.
   - `REJECTED` = the unfixed code already returned `Err` (checked elsewhere in MKL / the wrapper).
3. After the fix every probe case returns `Err` (full suite: 82 test binaries green on WSL).

## Static denominator (bcf7f80)

| family | safe pub fns taking buffers/sizes | static HOLE | static SUSPECT |
|---|---|---|---|
| lapack/mod.rs | 65 | 35 | 2 (+2 ilp64-only) |
| blas/* | 99 | 24 | 4 |
| sparse, dss, pardiso, preconditioners, iss, feast | 47 | ~40 | several (listed separately below) |

## Measured results on the unfixed code (one row per probe case)

| test binary | test fn | line | call | exit | verdict |
|---|---|---|---|---|---|
| lapack_safe_api | banded_buffers_are_checked | 34 | `gbsv` | 139 | FAULT |
| lapack_safe_api | banded_buffers_are_checked | 35 | `gbtrf` | 139 | FAULT |
| lapack_safe_api | banded_buffers_are_checked | 36 | `gbtrs` | 139 | FAULT |
| lapack_safe_api | banded_buffers_are_checked | 52 | `pbsv` | 139 | FAULT |
| lapack_safe_api | banded_buffers_are_checked | 53 | `pbtrf` | 139 | FAULT |
| lapack_safe_api | banded_buffers_are_checked | 54 | `pbtrs` | 139 | FAULT |
| lapack_safe_api | tridiagonal_buffers_are_checked | 59 | `gtsv` | 139 | FAULT |
| lapack_safe_api | tridiagonal_buffers_are_checked | 72 | `gtsv` | 0 | REJECTED |
| lapack_safe_api | tridiagonal_buffers_are_checked | 85 | `gttrf` | 139 | FAULT |
| lapack_safe_api | tridiagonal_buffers_are_checked | 86 | `gttrs` | 139 | FAULT |
| lapack_safe_api | tridiagonal_buffers_are_checked | 102 | `ptsv` | 139 | FAULT |
| lapack_safe_api | tridiagonal_buffers_are_checked | 103 | `pttrs_real` | 139 | FAULT |
| lapack_safe_api | packed_buffers_are_checked | 108 | `spsv` | 139 | FAULT |
| lapack_safe_api | packed_buffers_are_checked | 121 | `sptrs` | 139 | FAULT |
| lapack_safe_api | packed_buffers_are_checked | 122 | `ppsv` | 139 | FAULT |
| lapack_safe_api | packed_buffers_are_checked | 123 | `pptrs` | 139 | FAULT |
| lapack_safe_api | packed_buffers_are_checked | 124 | `hptrs` | 139 | FAULT |
| lapack_safe_api | pivot_contents_are_checked | 134 | `getrs` | 101 | ACCEPTED |
| lapack_safe_api | pivot_contents_are_checked | 138 | `getri` | 101 | ACCEPTED |
| lapack_safe_api | pivot_contents_are_checked | 139 | `laswp` | 101 | ACCEPTED |
| lapack_safe_api | pivot_contents_are_checked | 140 | `laswp` | 101 | ACCEPTED |
| lapack_safe_api | matrix_view_shapes_are_checked | 151 | `gels` | 101 | ACCEPTED |
| lapack_safe_api | matrix_view_shapes_are_checked | 157 | `geev_real` | 139 | FAULT |
| lapack_safe_api | matrix_view_shapes_are_checked | 174 | `gesdd` | 139 | FAULT |
| lapack_safe_api | larfg_stride_is_checked | 180 | `larfg` | 101 | ACCEPTED |
| blas_safe_api | banded_views_must_cover_the_band | 39 | `gbmv` | 0 | REJECTED |
| blas_safe_api | banded_views_must_cover_the_band | 56 | `gbmv` | 101 | ACCEPTED |
| blas_safe_api | banded_views_must_cover_the_band | 74 | `tbmv` | 101 | ACCEPTED |
| blas_safe_api | banded_views_must_cover_the_band | 87 | `tbsv` | 101 | ACCEPTED |
| blas_safe_api | banded_views_must_cover_the_band | 100 | `sbmv` | 101 | ACCEPTED |
| blas_safe_api | omatcopy2_buffers_are_checked | 107 | `omatcopy2` | 139 | FAULT |
| blas_safe_api | strided_batches_are_checked | 128 | `gemm_batch_strided` | 139 | FAULT |
| blas_safe_api | strided_batches_are_checked | 151 | `gemm3m_batch_strided` | 139 | FAULT |
| blas_safe_api | strided_batches_are_checked | 174 | `gemv_batch_strided` | 139 | FAULT |
| blas_safe_api | strided_batches_are_checked | 195 | `trsm_batch_strided` | 139 | FAULT |
| blas_safe_api | strided_batches_are_checked | 215 | `dgmm_batch_strided` | 139 | FAULT |
| blas_safe_api | strided_batches_are_checked | 237 | `axpy_batch_strided` | 101 | ACCEPTED |
| blas_safe_api | packed_compute_buffers_are_checked | 245 | `gemm_compute_packed_a` | 139 | FAULT |
| blas_safe_api | packed_compute_buffers_are_checked | 263 | `gemm_compute_packed_b` | 139 | FAULT |
| blas_safe_api | packed_compute_buffers_are_checked | 280 | `PackedMatrix::<f64>::pack_a(Layout::RowMajor,Tra` | 101 | ACCEPTED |
| blas_safe_api | compact_gemm_buffers_are_checked | 297 | `compact_gemm` | 139 | FAULT |
| sparse_safe_api | csr_content_is_validated | 39 | `SparseMatrix::<f64>::from_csr(1,1,IndexBase::Zer` | 101 | ACCEPTED |
| sparse_safe_api | csr_content_is_validated | 44 | `SparseMatrix::<f64>::from_csr(1,1,IndexBase::Zer` | 101 | ACCEPTED |
| sparse_safe_api | csr_content_is_validated | 49 | `SparseMatrix::<f64>::from_csr(1,1,IndexBase::One` | 101 | ACCEPTED |
| sparse_safe_api | csr_content_is_validated | 54 | `SparseMatrix::<f64>::from_csr(2,2,` | 101 | ACCEPTED |
| sparse_safe_api | csr_content_is_validated | 66 | `SparseMatrix::<f64>::from_coo(1,1,IndexBase::Zer` | 101 | ACCEPTED |
| sparse_safe_api | dense_operands_of_mm_are_checked | 84 | `a.mm(Operation::NoTrans,1.0,` | 134 | FAULT |
| sparse_safe_api | dss_buffers_are_checked | 104 | `d.define_structure(Symmetry::NonSymmetric,1,1,&[` | 101 | ACCEPTED |
| sparse_safe_api | dss_buffers_are_checked | 118 | `d.factor_real(Definite::Indefinite,&[1.0]).is_er` | 101 | ACCEPTED |
| sparse_safe_api | dss_buffers_are_checked | 121 | `d.solve_real(&[0.0;3],1000,&mut[0.0;3]).is_err()` | 139 | FAULT |
| sparse_safe_api | dss_buffers_are_checked | 122 | `d.solve_real(&[0.0;3],-1,&mut[0.0;3]).is_err());` | 0 | REJECTED |
| sparse_safe_api | pardiso_buffers_are_checked | 129 | `p.solve(3,&[1.0],&[1,2,3,6],&[1],&[0.0;3],&mut[0` | 139 | FAULT |
| sparse_safe_api | pardiso_buffers_are_checked | 136 | `p.solve(3,&[1.0;3],&ia,&ja,&[0.0;3],&mut[0.0;1])` | 101 | ACCEPTED |
| sparse_safe_api | pardiso_buffers_are_checked | 143 | `q.get_diagonal(1).is_err());}#[test]` | 101 | ACCEPTED |
| sparse_safe_api | ilu_and_feast_csr_content_is_validated | 148 | `ilu0` | 0 | REJECTED |
| sparse_safe_api | ilu_and_feast_csr_content_is_validated | 149 | `ilut` | 0 | REJECTED |
| sparse_safe_api | ilu_and_feast_csr_content_is_validated | 150 | `apply_ilu` | 139 | FAULT |
| sparse_safe_api | ilu_and_feast_csr_content_is_validated | 151 | `eigh_real_csr` | 139 | FAULT |
| sparse_safe_api | ilu_and_feast_csr_content_is_validated | 152 | `eigh_real_csr` | 0 | REJECTED |
| sparse_safe_api | fgmres_step_buffers_are_checked | 162 | `s.step(&mutb1,&mutx1).is_err());}` | 101 | ACCEPTED |

Totals: 32 FAULT, 22 ACCEPTED, 6 REJECTED (60 probe cases).

## Not measured (static SUSPECT / unprobed) — listed separately, NOT named as holes

- lapack: `larft`, `larfb` (n < k reads the k x k triangle; fixed defensively), usize wrap of `n*(n+1)/2` under ilp64 / 32-bit.
- blas: mixed-precision `validate_gemm_dims` and `JitGemm::execute` (`rows*ld` without `ld >= contiguous`; safe only if MKL rejects), `compact::get_size` i32 wrap, ilp64 `(n-1)*|inc|` wrap, `omatcopy2` leading-dimension unit semantics (checked conservatively).
- sparse: `qr_factor`/`qr_solve` ordering prerequisites, `spmmd` ldc, `convert_bsr` non-divisible dims, DSS/PARDISO pointer retention (fixed by owning copies; use-after-free not demonstrated), PARDISO `iparm` entries that change what MKL reads (Schur, iparm[35]), `Pardiso::reset` then reuse, `CgSession::finish` `x` immutability.
- Not fixed in the patch branch (need a breaking change, to ship with the *ExArgs PR): `gemm_batch` / `gemv_batch` / `trsm_batch` (raw pointer arrays in safe fns), `pardiso_64_raw`, `Pardiso::export`, `PardisoScalar`/`FeastReal`/`FeastComplex`/`BlasScalar` unsealed traits.

## Per-version groups (static, source identity only; runtime per version NOT yet run)

Published versions: 0.1.0 0.2.0 0.3.0 0.3.4 0.3.8 0.3.10 0.3.11 0.4.2 0.4.3 0.4.4 0.4.5 0.4.6 0.4.8 0.4.11
0.4.15 0.4.16 0.5.0 0.5.1 0.5.2 0.6.0 0.6.1 0.6.5 0.6.7 0.6.8 0.6.11 (25, none yanked; from the crates.io API).
Source identity of each audited file across these versions was compared by md5; the groups are in
`SAFE-API-AUDIT-versions.txt`. Per-function first/last-affected needs the probe harness built against each
group's `.crate` and is the next step.
