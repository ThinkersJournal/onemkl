# onemkl safe-API audit: per-version table (STATIC, NOT RUN)

**Platform of the runtime measurements this refers to: oneMKL 2026.2 on WSL (Linux) ONLY** (see SAFE-API-AUDIT.md, added by #56).
**Every verdict below is a static read of the published `.crate` source, not a run.** The runtime
probes were run only on bcf7f80, which is byte-identical (md5) to 0.6.8 and 0.6.11 for the audited modules.
Some probe cases were already rejected at 0.6.11 by MKL or an existing check (gtsv short dl/du, gbmv
`leading_dim`, dss `nrhs < 0`, ilu0/ilut, eigh_real_csr n=0): those fns are UNCHECKED statically only.
Versions compared: 0.1.0 0.2.0 0.3.0 0.3.4 0.3.8 0.3.10 0.3.11 0.4.2 0.4.3 0.4.4 0.4.5 0.4.6 0.4.8 0.4.11
0.4.15 0.4.16 0.5.0 0.5.1 0.5.2 0.6.0 0.6.1 0.6.5 0.6.7 0.6.8 0.6.11 (25; none yanked at the time of the audit).
Result: every probed fn is UNCHECKED in every published version (first affected 0.1.0, last affected 0.6.11);
versions differ only in type spelling, rustfmt, and the PackArgs / CsrRef / FeastStatus refactors.
`pack_b` shares `pack` with `pack_a` but was not probed.

# onemkl safe-API audit: per-version verdicts (ALL verdicts: static, not run)

Scope: 25 published versions 0.1.0 .. 0.6.11 (list in the task). Fixed reference: `C:\Projects\onemkl-wt-safe-audit\onemkl\src` (HEAD b5f351e; diff vs bcf7f80 read in full).
Method: for each probed fn, the body was extracted from every version (brace-matched, CRLF normalised, md5 per fn body) and diffed against the 0.6.11 body.
The fn-body groups are finer than the file-level groups in SAFE-API-AUDIT-versions.txt. Where a fn has several body groups, the diff between them
is listed. "Types" = `i32`/`c_int` -> `MKL_INT` spelling, rustfmt only. No semantic check was added or removed between any two groups below.
Lines are `file:line` in the group's first version | last version (grouped versions shift lines); the fixed version's missing check is named in the note.
Verdict key: UNCHECKED = body reaches the FFI call without the check the fixed tree adds. No fn was ABSENT in any version (all exist in 0.1.0). No CHECKED verdicts.
First affected = 0.1.0 and last affected = 0.6.11 (latest published) for every row unless stated. 0.6.11 = bcf7f80 for these files (per the audit).

## LAPACK (lapack/mod.rs)  groups: same body in all 25 versions except Types (0.1.0..0.6.0 vs 0.6.1..0.6.11, and getri/laswp/larfg also 0.4.2 boundary)

| fn | first | last | notes (static, not run) |
|---|---|---|---|
| gbsv | 0.1.0 | 0.6.11 | UNCHECKED all. Only `if ipiv.len() < n` (lapack/mod.rs:786 \| 0.6.11:772); no `ab`/`b` extent check before `T::lapacke_gbsv(` (:790 \| :776). Fixed adds check_band + check_rhs. Groups differ by Types only. |
| gbtrf | 0.1.0 | 0.6.11 | UNCHECKED all. Only `if ipiv.len() < m.min(n)` (:818 \| 807); `ab` unchecked before `T::lapacke_gbtrf(` (:824 \| :813). Types only. |
| gbtrs | 0.1.0 | 0.6.11 | UNCHECKED all. Only `if ipiv.len() < n` (:853 \| 845); no ab/b extent, no ipiv content check. Types only. |
| pbsv | 0.1.0 | 0.6.11 | UNCHECKED all (1 body group). No check at all before `T::lapacke_pbsv(` (:981 \| :982). |
| pbtrf | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). Straight to `T::lapacke_pbtrf(` (:1006 \| :1007). |
| pbtrs | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). Straight to `T::lapacke_pbtrs(` (:1032 \| :1036). |
| gtsv | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). Has `if dl.len() < n.saturating_sub(1) \|\| d.len() < n \|\| du.len() < ...` (:887 \| :882) = why probe line 72 (short dl/du) was REJECTED; `b`/`ldb` never checked before `T::lapacke_gtsv(` (:893 \| :888). Fixed adds check_rhs only. |
| gttrf | 0.1.0 | 0.6.11 | UNCHECKED all. Only `if du2.len() < n.saturating_sub(2) \|\| ipiv.len() < n` (:916 \| 911); dl/d/du lengths unchecked. Types only. |
| gttrs | 0.1.0 | 0.6.11 | UNCHECKED all. No check at all before `T::lapacke_gttrs(` (:950 \| :948). Types only. |
| ptsv | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). Checks d/e (`if d.len() < n \|\| e.len() < n.saturating_sub(1)` :1059 \| :1062) but `b` extent never checked before `T::lapacke_ptsv(` (:1065 \| :1068). |
| pttrs_real | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). No d/e/b check before `T::lapacke_pttrs(` (:1106 \| :1103). |
| spsv | 0.1.0 | 0.6.11 | UNCHECKED all. `ap.len() < n * (n + 1) / 2` (:1167 \| 1170, also unchecked usize wrap) and `ipiv.len() < n`; `b` extent unchecked. Types only. |
| sptrs | 0.1.0 | 0.6.11 | UNCHECKED all. No ap/ipiv/b check before `T::lapacke_sptrs(` (:1231 \| :1237). Types only. |
| ppsv | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). Only `ap.len() < n * (n + 1) / 2` (:1256 \| :1261); `b` unchecked. |
| pptrs | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). No check before `T::lapacke_pptrs(` (:1310 \| :1309). |
| hptrs | 0.1.0 | 0.6.11 | UNCHECKED all. No check before `T::lapacke_hptrs(` (:1400 \| :1405). Types only. |
| getrs | 0.1.0 | 0.6.11 | UNCHECKED all. `if ipiv.len() < n` only (:179 \| 187); pivot VALUES (1..=n) never checked, then `T::lapacke_getrs(` (:183 \| :191). Types only. |
| getri | 0.1.0 | 0.6.11 | UNCHECKED all. `if ipiv.len() < n` only (0.1.0 :204; 0.4.2 :198; 0.6.11 :209); no content check. 3 groups: Types/rustfmt only. |
| laswp | 0.1.0 | 0.6.11 | UNCHECKED all. Checks `k1 < 1 \|\| k2 < k1 \|\| (k2 as usize) > ipiv.len()` (0.1.0 :2028; 0.6.11 :2061) only; no k2 <= a.rows(), no incx != 0, no ipiv values. 3 groups: Types/rustfmt only. |
| gels | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). Straight to `T::lapacke_gels(` (:451 \| :444) after `ensure_layout`; no `b.rows() >= max(m,n)`. |
| geev_real | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). Only `wr.len() < n \|\| wi.len() < n` (:575 \| :563); vl/vr views never checked for Compute (:591 \| :579 `T::lapacke_geev(`). |
| gesdd | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). Only `s.len() < m.min(n)` (:667 \| :651); u/vt views unchecked before `T::lapacke_gesdd(`. |
| larfg | 0.1.0 | 0.6.11 | UNCHECKED all. Only `x.len() < n.saturating_sub(1)` (0.1.0 :2077; 0.4.2 :2254; 0.6.11 :2296): ignores `incx` stride/zero. 3 groups: Types/rustfmt/`where T: Default` form only. |

## BLAS (blas/level2.rs, extensions.rs, packed.rs, compact.rs)

| fn | first | last | notes (static, not run) |
|---|---|---|---|
| gbmv | 0.1.0 | 0.6.11 | UNCHECKED all (1 body group, all 25). Only `if a.leading_dim() < kl + ku + 1` (level2.rs:89 \| 96) = the REJECTED probe (line 39); view size (rows/cols vs band) never checked before `T::cblas_gbmv(` (:102 \| :109). Fixed adds check_band_view. |
| tbmv | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). Only `if a.leading_dim() < k + 1` (:293 \| 303); no view-extent check before `T::cblas_tbmv(` (:300 \| :310). |
| tbsv | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). Only `if a.leading_dim() < k + 1` (:328 \| 341); `T::cblas_tbsv(` (:335 \| :348). |
| sbmv | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). Only `if a.leading_dim() < k + 1` (:522 \| 554); `T::cblas_sbmv(` (:530 \| :562). |
| omatcopy2 | 0.1.0 | 0.6.11 | UNCHECKED all. Only `if stridea == 0 \|\| strideb == 0` (extensions.rs:156 \| 166); no extent check of a/b vs lda/ldb/stride before `T::mkl_omatcopy2(` (:162 \| :170). 2 groups (0.1.0..0.3.11, 0.4.2..): rustfmt only. |
| gemm_batch_strided | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). `let _ = (a.len(), b.len(), c.len());` (extensions.rs 0.1.0 :422; 0.6.11 :451)\| 451 area; 0.1.0 :422) discards the lengths; `T::cblas_gemm_batch_strided(` (:427 \| :456). |
| gemm3m_batch_strided | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). `let _ = (a.len(), b.len(), c.len());` (0.1.0 :903; 0.6.11 :932) then FFI. |
| gemv_batch_strided | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). `let _ = (a.len(), x.len(), y.len()); // length checks deferred to MKL.` (0.1.0 :370; 0.6.11 :396). |
| trsm_batch_strided | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). `let _ = (a.len(), b.len());` (0.1.0 :809; 0.6.11 :832). |
| dgmm_batch_strided | 0.1.0 | 0.6.11 | UNCHECKED all (1 group). `let _ = (a.len(), x.len(), c.len());` (0.1.0 :855; 0.6.11 :881). |
| axpy_batch_strided | 0.1.0 | 0.6.11 | UNCHECKED all (1 group, the probe is the negative-stride case). Calls `check_batch_buffers(x.len(), n, incx, stridex, batch_size, "x")?` (:301 \| 321) but that helper (1 group, all 25) uses `let abs_stride = stride.unsigned_abs() as usize;` (:947 \| 976) and unchecked `(batch_size - 1) * abs_stride`: negative stride passes and reaches before the slice. Fixed rejects stride < 0 + checked arithmetic. |
| gemm_compute_packed_a | 0.1.0 | 0.6.11 | UNCHECKED all. Only checks the packed handle (`"a_packed shape does not match the requested compute shape"` 0.1.0 :345; 0.6.11 :419); `b`/`c` lengths vs ldb/ldc never checked before `T::gemm_compute(` (:357 \| :431). 3 groups: Types only (c_int -> MKL_INT, rustfmt). |
| gemm_compute_packed_b | 0.1.0 | 0.6.11 | UNCHECKED all. Same: `"b_packed shape does not match..."` (0.1.0 :397; 0.6.11 :476), `a`/`c` unchecked, `T::gemm_compute(` (:407 \| :486). 3 groups: Types only. |
| PackedMatrix::pack_a | 0.1.0 | 0.6.11 | UNCHECKED all. pack_a delegates to private `pack` (0.1.0 :212 `Self::pack(layout, PackIdentifier::A, ...)`; 0.4.15+ `Self::pack(PackArgs{..})` :258). `pack` has `let needed_src = match layout { RowMajor => src_rows.saturating_mul(ld), ...}` + `if src.len() < needed_src` (packed.rs 0.1.0 :267-271; 0.4.2 :302-306; 0.4.15 :335-339; 0.6.11 :334-338) but never `ld >= contiguous` (probe: RowMajor, k=100, ld=1, len 1 passes). 4 `pack` groups differ by PackArgs refactor/Types only. Same hole in pack_b (unprobed, same helper). |
| compact::gemm (compact_gemm) | 0.1.0 | 0.6.11 | UNCHECKED all. No ap/bp/cp length check: body starts at `let m_i: c_int = ...` (blas/compact.rs 0.1.0 :351; 0.4.2 :413; 0.6.11 :436) then `T::gemm_compact(` (:361 \| :423 \| :446). Fixed adds `packed_len::<T>(...)` checks for ap/bp/cp. 3 groups: Types/rustfmt only. |

## Sparse / DSS / PARDISO / ILU / FEAST / ISS

Groups per fn (non-lapack/blas): from_csr/from_coo/define_structure/factor_real/pardiso solve/get_diagonal/ilu0/ilut share 3 groups
(0.1.0..0.3.11, 0.4.2(/0.4.15)..0.6.1, 0.6.5..0.6.11); the differences between them are Types/rustfmt/internal refactor only (details in notes).

| fn | first | last | notes (static, not run) |
|---|---|---|---|
| SparseMatrix::from_csr | 0.1.0 | 0.6.11 | UNCHECKED all. Only `row_ptr.len() != rows + 1` (sparse/mod.rs:320 \| 321) and `col_idx.len() != values.len()` (:325 \| 324); content (row_ptr[0]==base, monotone, in-range col_idx) never validated before `T::sparse_create_csr(` (:343 \| :342). Fixed: `validate_compressed`. 3 groups: Types only. |
| SparseMatrix::from_coo | 0.1.0 | 0.6.11 | UNCHECKED all. Only the 3-way length check (`row_indx.len() != col_indx.len() \|\| ...` :379 \| 378); no index range check before `T::sparse_create_coo(` (:394 \| :396). Types only. |
| SparseMatrix::mm | 0.1.0 | 0.6.11 | UNCHECKED all (1 group, all 25). First statement is `let descr_inner = descr.into().inner;` (:629 \| 643), then `T::sparse_mm(` (:631 \| :645) with x/y never sized against columns/ldx/ldy. Fixed adds check_dense_extent. |
| Dss::define_structure | 0.1.0 | 0.6.11 | UNCHECKED all. Only `row_ptr.len() != rows + 1` (dss.rs:131 \| 132); `col_idx` content/nnz/square not validated before `sys::dss_define_structure_(` (:143 \| :142). Fixed also copies the arrays (pointer retention). Types only. |
| Dss::factor_real | 0.1.0 | 0.6.11 | UNCHECKED all. `values` length vs nnz unchecked; direct `sys::dss_factor_real_(&mut self.handle, &opt, values.as_ptr().cast())` (:263 \| :264; 0.1.0 :272). Types/rustfmt only. |
| Dss::solve_real | 0.1.0 | 0.6.11 | UNCHECKED all (2 groups, Types only). No check: first line `let opt: c_int = sys::MKL_DSS_DEFAULTS as c_int;` (dss.rs 0.1.0 :284; 0.6.11 :271) then `sys::dss_solve_real_(` (:286 \| :273); b/x vs n*nrhs unchecked (nrhs=1000 faulted; nrhs=-1 was REJECTED by MKL, not by the wrapper). |
| Pardiso::solve | 0.1.0 | 0.6.11 | UNCHECKED all. `solve` is a thin wrapper: `let nrhs = (b.len() / n).max(1);` (pardiso.rs:257 \| 258; n==0 panics) then `self.run_phase(...)` (:263 \| :264). `run_phase` (3 groups: Types/CsrRef refactor only) goes straight to `sys::pardiso(` (0.1.0 :501; 0.4.15 :528; 0.6.11 :528) after `let n_i32` (:484 \| :511); no ia/ja/a/b/x length checks anywhere. Fixed adds validate_compressed + n*nrhs checks. |
| Pardiso::get_diagonal | 0.1.0 | 0.6.11 | UNCHECKED all. Checks only `self.factorized` and `if self.iparm[55] != 1` (:326 \| 346); `n` never compared to `factorized_n`: `let mut df: Vec<T> = (0..n)...` (:331 \| 351) then `sys::pardiso_getdiag(` (:336 \| :356). Types only. |
| preconditioners::ilu0 | 0.1.0 | 0.6.11 | UNCHECKED statically (all). Only `ia.len() != n + 1` (preconditioners.rs:36 \| 38) and `a.len() != ja.len()` (:41 \| 43); ia/ja content unvalidated before `sys::dcsrilu0(` (:55 \| :55). Note: the probe input (ia=[1,5,9]) returned Err at 0.6.11 (REJECTED, i.e. MKL-side), so this class is not a measured hole. Fixed adds validate_compressed. Types only. |
| preconditioners::ilut | 0.1.0 | 0.6.11 | UNCHECKED statically (all). Same two length checks (:102 \| 104, :107 \| 109) then `sys::dcsrilut(` (:137 \| :133). Probe REJECTED at 0.6.11 (MKL-side). Also `(2 * mf + 1) * n - ...` unchecked overflow (all groups). Types/rustfmt only. |
| preconditioners::apply_ilu | 0.1.0 | 0.6.11 | UNCHECKED all. Length checks only (`ia.len() != n + 1` :180/182/168/179; `alu.len() != ja.len()`), no content validation of ia/ja. 4 groups: 0.1.0 (fca292) is a different implementation: raw `sys::mkl_dcsrtrsv(` ×2 (:204, :221) on the unvalidated arrays; 0.2.0+ builds `sys::mkl_sparse_d_create_csr(` (0.2.0 :207; 0.4.2 :193; 0.6.5 :204) on them. All reach FFI unvalidated (probe line 150 FAULT at 0.6.11). |
| feast::eigh_real_csr | 0.1.0 | 0.6.11 | UNCHECKED all. Only `isa.len() != n + 1` (feast.rs:210 \| 215) and `jsa.len() != sa.len()` (:213 \| 218); no CSR content validation, no `1 <= m0 <= n`, before `T::feast_csr_real(` (:235 \| :240) (`vec![T::zero(); n * m0_estimate]` :231 \| :236). Probe n=0 was REJECTED (MKL info), content case FAULT. 4 groups: Types/FeastStatus refactor only. |
| iss::FgmresSession::step | 0.1.0 | 0.6.11 | UNCHECKED all. No b/x length check: `let mut rci` then `sys::dfgmres(` with `x.as_mut_ptr(), b.as_mut_ptr()` (iss.rs 0.1.0 :718/:720; 0.4.2 :696/:698; 0.6.11 :716/:718); `restart`/ipar[14] vs tmp also unchecked. 3 groups: Types only. (CgSession::step, iss.rs 0.1.0 :546, 0.6.11 :551, same hole, unprobed.) |

## Summary

- Every probed fn exists in every version (0.1.0 onward): 0 ABSENT. 0 CHECKED. 0 groups whose verdict differs from latest.
  The earliest affected version is 0.1.0 and the last is 0.6.11 for all 60-case fns (static, not run).
- Internal differences across groups are limited to: i32/c_int -> MKL_INT spelling; rustfmt; PackArgs/CsrRef/FeastStatus argument-struct refactors (0.4.15, 0.6.x);
  apply_ilu in 0.1.0 using the legacy `mkl_dcsrtrsv` instead of the sparse handle API (0.2.0+); Dss/Pardiso/sparse constructors otherwise unchanged.
- Part of a probe set can be REJECTED by an existing check (gtsv dl/du length, gbmv leading_dim, dss nrhs<0 [MKL], ilu0/ilut ia content [MKL], eigh_real_csr n=0 [MKL]); the fn is still UNCHECKED for the other probed class.
- Caveats: static reading of bodies only; helpers called by bodies (ensure_layout, dim_to_mkl_int, check_vec_len [2 groups, rustfmt only]) were not diffed except where cited; no runtime per version.

## MEASURED (runtime) per-version probes — oneMKL 2026.2 on WSL (Linux x86_64) ONLY

Method: a probe crate depending on `onemkl = "=<version>"` from crates.io, one binary per case, each calls the
safe fn with an undersized/empty buffer and prints whether it returned. rc 139 = SIGSEGV (FAULT, MEASURED).
Cases: lapack `gbsv`, `pbsv`, `gttrs`, `spsv`; blas `omatcopy2`, `gemm_batch_strided`; dss `solve_real` (nrhs 1000, 3-entry b/x).

| version | lapack gbsv/pbsv/gttrs/spsv | blas omatcopy2 / gemm_batch_strided | dss solve_real |
|---|---|---|---|
| 0.6.11, 0.6.7, 0.6.5, 0.6.1, 0.5.2, 0.4.15, 0.4.3, 0.4.2, 0.3.8, 0.3.0 | FAULT (139) x4 | FAULT (139) x2 | NOT MEASURED (binary exited 127: harness problem, not a result) |
| 0.1.0 | NOT MEASURED (probe did not compile against 0.1.0: API differs or build failed) | NOT MEASURED | NOT MEASURED |

Versions not run at runtime: 0.2.0 0.3.4 0.3.10 0.3.11 0.4.4 0.4.5 0.4.6 0.4.8 0.4.11 0.4.16 0.5.0 0.5.1 0.6.0 0.6.8
(their sources are covered by the static table above and by md5 identity groups; 0.6.8 == 0.6.11 byte-identical
for the audited modules). Sparse, PARDISO, ILU and FEAST probes were NOT run per version (only on bcf7f80).
So for the six lapack/blas probes the earliest runtime-measured fault is 0.3.0 and the latest 0.6.11; 0.1.0 and 0.2.0 are static only.
