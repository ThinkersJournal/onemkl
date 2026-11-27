# W1: functions flagged by `clippy::too_many_arguments` when the source allows are overridden

Generated 2026-11-27 on `origin/main` 65478879 (rustc / clippy 1.99.1, WSL) from `cargo clippy -p onemkl --lib -- --force-warn clippy::too_many_arguments`; `since` is the earliest of the release tags v0.1.0, v0.2.0, v0.3.0, v0.3.4, v0.3.8, v0.3.10, v0.3.11 in whose copy of the same file a function of that name exists. The classification is mechanical (signature text and doc comment), a FIRST PASS for review: nothing here has been confirmed against the C headers argument by argument, and the C-mirror verdicts mean "likely", not "verified". Counted waivers already carrying a `reason` are in the table too (7 as of this run: 6 from #21, 1 from #31).

| file | line | function | args | kind | first-pass class |
|---|---|---|---|---|---|
| `onemkl/src/blas/scalar.rs` | 108 | `cblas_gemv` | 12 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 130 | `cblas_gbmv` | 14 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 153 | `cblas_trmv` | 9 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 171 | `cblas_trsv` | 9 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 188 | `cblas_tbmv` | 10 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 206 | `cblas_tbsv` | 10 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 227 | `cblas_tpmv` | 8 | unsafe | C-mirror (raw FFI method): doc does not name the C function |
| `onemkl/src/blas/scalar.rs` | 246 | `cblas_tpsv` | 8 | unsafe | C-mirror (raw FFI method): doc does not name the C function |
| `onemkl/src/blas/scalar.rs` | 267 | `cblas_gemm` | 14 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 297 | `cblas_symm` | 13 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 324 | `cblas_syrk` | 11 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 350 | `cblas_syr2k` | 13 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 377 | `cblas_trmm` | 12 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 403 | `cblas_trsm` | 12 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 430 | `cblas_gemmt` | 14 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 476 | `mkl_imatcopy` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 497 | `mkl_omatcopy` | 9 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 518 | `mkl_omatcopy2` | 11 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 542 | `mkl_omatadd` | 13 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 573 | `cblas_axpy_batch_strided` | 9 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 594 | `cblas_copy_batch_strided` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 616 | `cblas_gemv_batch_strided` | 16 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 646 | `cblas_gemm_batch_strided` | 18 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 679 | `cblas_gemm_batch` | 16 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 706 | `cblas_gemv_batch` | 14 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 731 | `cblas_trsm_batch` | 14 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 758 | `cblas_trsm_batch_strided` | 15 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 789 | `cblas_dgmm_batch_strided` | 14 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 862 | `cblas_ger` | 10 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 886 | `cblas_symv` | 11 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 910 | `cblas_syr` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 933 | `cblas_syr2` | 10 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 957 | `cblas_sbmv` | 12 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 983 | `cblas_spmv` | 10 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 1026 | `cblas_spr2` | 9 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 1099 | `cblas_gerc` | 10 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 1124 | `cblas_geru` | 10 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 1148 | `cblas_hemv` | 11 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 1173 | `cblas_her` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 1196 | `cblas_her2` | 10 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 1220 | `cblas_hbmv` | 12 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 1246 | `cblas_hpmv` | 10 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 1289 | `cblas_hpr2` | 9 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 1312 | `cblas_hemm` | 13 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 1339 | `cblas_herk` | 11 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 1365 | `cblas_her2k` | 13 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 1393 | `cblas_gemm3m` | 14 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/scalar.rs` | 1421 | `cblas_gemm3m_batch_strided` | 18 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/compact.rs` | 104 | `gepack_compact` | 9 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/compact.rs` | 123 | `geunpack_compact` | 9 | unsafe | C-mirror (raw FFI method): doc does not name the C function |
| `onemkl/src/blas/compact.rs` | 142 | `gemm_compact` | 16 | unsafe | C-mirror (raw FFI method): doc does not name the C function |
| `onemkl/src/blas/compact.rs` | 322 | `gepack` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/compact.rs` | 362 | `geunpack` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/compact.rs` | 396 | `gemm` | 16 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/compact.rs` | 482 | `trsm_compact` | 14 | unsafe | C-mirror (raw FFI method): doc does not name the C function |
| `onemkl/src/blas/compact.rs` | 506 | `potrf_compact` | 8 | unsafe | C-mirror (raw FFI method): doc does not name the C function |
| `onemkl/src/blas/compact.rs` | 524 | `getrfnp_compact` | 8 | unsafe | C-mirror (raw FFI method): doc does not name the C function |
| `onemkl/src/blas/compact.rs` | 544 | `geqrf_compact` | 11 | unsafe | C-mirror (raw FFI method): doc does not name the C function |
| `onemkl/src/blas/compact.rs` | 566 | `getrinp_compact` | 9 | unsafe | C-mirror (raw FFI method): doc does not name the C function |
| `onemkl/src/blas/compact.rs` | 715 | `trsm` | 14 | pub | public surface (published since v0.3.8); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/compact.rs` | 855 | `geqrf` | 8 | pub | public surface (published since v0.3.8); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/extensions.rs` | 77 | `imatcopy` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/extensions.rs` | 107 | `omatcopy` | 9 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/extensions.rs` | 144 | `omatcopy2` | 11 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/extensions.rs` | 181 | `omatadd` | 13 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/extensions.rs` | 295 | `axpy_batch_strided` | 9 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/extensions.rs` | 326 | `copy_batch_strided` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/extensions.rs` | 357 | `gemv_batch_strided` | 16 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/extensions.rs` | 407 | `gemm_batch_strided` | 18 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/extensions.rs` | 788 | `trsm_batch_strided` | 15 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/extensions.rs` | 835 | `dgmm_batch_strided` | 14 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/extensions.rs` | 879 | `gemm3m_batch_strided` | 18 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/jit.rs` | 80 | `jit_create` | 12 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/jit.rs` | 345 | `new` | 11 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/level2.rs` | 32 | `gemv` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/level2.rs` | 76 | `gbmv` | 12 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/level2.rs` | 284 | `tbmv` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/level2.rs` | 319 | `tbsv` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/level2.rs` | 363 | `tpmv` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/level2.rs` | 398 | `tpsv` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/level2.rs` | 431 | `symv` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/level2.rs` | 525 | `sbmv` | 10 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/level2.rs` | 565 | `spmv` | 10 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/level2.rs` | 625 | `spr2` | 9 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/level2.rs` | 661 | `hemv` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/level2.rs` | 755 | `hbmv` | 10 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/level2.rs` | 795 | `hpmv` | 10 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/level2.rs` | 855 | `hpr2` | 9 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/level3.rs` | 78 | `gemmt` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/mixed_precision.rs` | 122 | `gemm_bf16_f32` | 14 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/mixed_precision.rs` | 170 | `gemm_f16_f32` | 14 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/mixed_precision.rs` | 222 | `gemm_e5m2_f32` | 14 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/mixed_precision.rs` | 274 | `gemm_e4m3_f32` | 14 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/mixed_precision.rs` | 328 | `gemm_s8u8_s32` | 18 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/mixed_precision.rs` | 394 | `gemm_s16_s32` | 18 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/mixed_precision.rs` | 464 | `hgemm` | 14 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/packed.rs` | 86 | `gemm_pack` | 10 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/packed.rs` | 107 | `gemm_compute` | 13 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/blas/packed.rs` | 239 | `pack_a` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/packed.rs` | 254 | `pack_b` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/packed.rs` | 268 | `pack` | 9 | fn | private fn: restructure or reason |
| `onemkl/src/blas/packed.rs` | 365 | `gemm_compute_packed_a` | 11 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/blas/packed.rs` | 419 | `gemm_compute_packed_b` | 11 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/data_fitting.rs` | 58 | `df_edit_pp_spline_1d` | 9 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/data_fitting.rs` | 87 | `df_interpolate_1d` | 12 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/data_fitting.rs` | 110 | `df_integrate_1d` | 11 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/feast.rs` | 197 | `eigh_real_csr` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/feast.rs` | 265 | `eigh_complex_csr` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/feast.rs` | 337 | `eigh_real_banded` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/feast.rs` | 402 | `eigh_complex_banded` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/feast.rs` | 469 | `gen_eigh_real_dense` | 9 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/feast.rs` | 540 | `gen_eigh_complex_dense` | 9 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/feast.rs` | 608 | `finish_real_result` | 8 | fn | private fn: restructure or reason |
| `onemkl/src/feast.rs` | 636 | `finish_complex_result` | 8 | fn | private fn: restructure or reason |
| `onemkl/src/feast.rs` | 681 | `feast_dense_real` | 15 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/feast.rs` | 710 | `feast_csr_real` | 16 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/feast.rs` | 740 | `feast_banded_real` | 16 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/feast.rs` | 770 | `feast_dense_gen_real` | 17 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/feast.rs` | 935 | `feast_dense_complex` | 15 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/feast.rs` | 963 | `feast_csr_complex` | 16 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/feast.rs` | 992 | `feast_banded_complex` | 16 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/feast.rs` | 1021 | `feast_dense_gen_complex` | 17 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/fft.rs` | 1833 | `with_scales` | 8 | pub | public surface (published since v0.3.8); C-mirror if it takes one C call's arguments |
| `onemkl/src/lapack/scalar.rs` | 26 | `lapacke_gesv` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 64 | `lapacke_getrs` | 9 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 101 | `lapacke_posv` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 136 | `lapacke_potrs` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 212 | `lapacke_gels` | 9 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 235 | `lapacke_gelsd` | 11 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 262 | `lapacke_gesdd` | 11 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 288 | `lapacke_gesvd` | 13 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 316 | `lapacke_gbsv` | 10 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 338 | `lapacke_gbtrf` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 359 | `lapacke_gbtrs` | 11 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 384 | `lapacke_gtsv` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 429 | `lapacke_gttrs` | 11 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 453 | `lapacke_pbsv` | 9 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 491 | `lapacke_pbtrs` | 9 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 546 | `lapacke_spsv` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 583 | `lapacke_sptrs` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 652 | `lapacke_lacpy` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 727 | `lapacke_larft` | 10 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 751 | `lapacke_larfb` | 14 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 778 | `lapacke_laswp` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 803 | `lapacke_sysv` | 9 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 846 | `lapacke_geev` | 12 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 894 | `lapacke_sygv` | 10 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 922 | `lapacke_ggev` | 15 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 973 | `lapacke_syevr` | 17 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 1007 | `lapacke_hesv` | 9 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 1049 | `lapacke_geev_complex` | 11 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 1074 | `lapacke_pttrs_complex` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 1095 | `lapacke_hpsv` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 1132 | `lapacke_hptrs` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 1154 | `lapacke_hegv` | 10 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 1182 | `lapacke_ggev_complex` | 14 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/scalar.rs` | 1230 | `lapacke_heevr` | 17 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/mod.rs` | 751 | `gbsv` | 10 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/lapack/mod.rs` | 785 | `gbtrf` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/lapack/mod.rs` | 817 | `gbtrs` | 11 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/lapack/mod.rs` | 854 | `gtsv` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/lapack/mod.rs` | 913 | `gttrs` | 11 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/lapack/mod.rs` | 946 | `pbsv` | 9 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/lapack/mod.rs` | 997 | `pbtrs` | 9 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/lapack/mod.rs` | 1094 | `pttrs_complex` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/lapack/mod.rs` | 1128 | `spsv` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/lapack/mod.rs` | 1191 | `sptrs` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/lapack/mod.rs` | 1292 | `hpsv` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/lapack/mod.rs` | 1355 | `hptrs` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/lapack/mod.rs` | 1489 | `ggev_real` | 9 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/lapack/mod.rs` | 1815 | `ggev_complex` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/lapack/mod.rs` | 2427 | `lapacke_lacrm` | 9 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/lapack/mod.rs` | 2567 | `lapacke_syconv` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/pardiso.rs` | 277 | `solve_multi` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/pardiso.rs` | 480 | `run_phase` | 9 | fn | private fn: restructure or reason |
| `onemkl/src/pardiso.rs` | 594 | `pardiso_64_raw` | 15 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/pde.rs` | 393 | `init_2d` | 11 | unsafe | C-mirror (raw FFI method): doc does not name the C function |
| `onemkl/src/pde.rs` | 406 | `commit_2d` | 9 | unsafe | C-mirror (raw FFI method): doc does not name the C function |
| `onemkl/src/pde.rs` | 417 | `solve_2d` | 9 | unsafe | C-mirror (raw FFI method): doc does not name the C function |
| `onemkl/src/pde.rs` | 1022 | `new` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/rng/summary_stats.rs` | 73 | `ss_edit_moments` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/rng/summary_stats.rs` | 180 | `ss_edit_missing_values` | 11 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/rng/mod.rs` | 884 | `rng_lognormal` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/rng/mod.rs` | 957 | `rng_beta` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/rng/mod.rs` | 1044 | `rng_gaussian_mv` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/sparse/scalar.rs` | 31 | `sparse_create_csr` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/sparse/scalar.rs` | 52 | `sparse_create_coo` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/sparse/scalar.rs` | 73 | `sparse_create_csc` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/sparse/scalar.rs` | 94 | `sparse_create_bsr` | 10 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/sparse/scalar.rs` | 132 | `sparse_mm` | 11 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/sparse/scalar.rs` | 185 | `sparse_dotmv` | 8 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/sparse/mod.rs` | 483 | `from_bsr` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/sparse/mod.rs` | 623 | `mm` | 11 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/sparse/mod.rs` | 1111 | `sparse_qr_solve` | 9 | unsafe | C-mirror (raw FFI method) |
| `onemkl/src/sparse/mod.rs` | 1183 | `qr_solve` | 8 | pub | public surface (published since v0.1.0); C-mirror if it takes one C call's arguments |
| `onemkl/src/vm/scalar.rs` | 278 | `vm_linear_frac` | 8 | unsafe | C-mirror (raw FFI method): doc does not name the C function |

Summary by class: C-mirror 124, private fn 4, public surface 68; total 196.

## Chunk 1: C mirrors verified by argument count (blas/scalar.rs, lapack/scalar.rs)

For the raw trait declarations in `onemkl/src/blas/scalar.rs` (`cblas_*`) and `onemkl/src/lapack/scalar.rs` (`lapacke_*`), `tools/w1/cblas_check.py` and `tools/w1/lap_check.py` compare the Rust argument count with the argument count of every precision variant of the named C function in the installed oneMKL 2026.2 headers (`mkl_cblas.h`, `mkl_lapacke.h`; complex-only and real-only methods are compared with their own precisions only). Result: **44 of 44** flagged `cblas_*` declarations and **34 of 34** flagged `lapacke_*` declarations have the same argument count as every compared variant (2 of the 44 already carried a reason from #21). The 76 bare allows on these declarations now carry `reason = "mirrors the oneMKL C signature <name> argument for argument"`. The check compares counts, not parameter types or order; the declarations keep the C parameter order by construction (they were written from those prototypes), which was not re-verified here. The four flagged lines in `blas/scalar.rs` that are not `cblas_*` methods and the remaining files are still open.

## Chunk 2: C mirrors in the other raw trait declarations (35)

`tools/w1/gen_check.py` does the same count comparison with an explicit Rust-name to C-name map for the remaining `unsafe fn` declarations whose C function is in the installed headers (`mkl_spblas.h`, `mkl_vsl_functions.h`, `mkl_vml_functions.h`, `mkl_solvers_ee.h`, `mkl_df_functions.h`, `mkl_compact.h`, `mkl_cblas.h`, `mkl_lapacke.h`; prototypes parsed from both the plain and the `_Mkl_Api(...)` macro forms). **35 of 35 mapped declarations match** and now carry a `reason` naming the C function: `blas/compact.rs` 8 (`mkl_?gepack_compact`, `geunpack`, `gemm`, `trsm`, `potrf`, `getrfnp`, `geqrf`, `getrinp`), `blas/packed.rs` 2 (`cblas_?gemm_pack`, `_compute`), `data_fitting.rs` 3 (`df?EditPPSpline1D`, `Interpolate1D`, `Integrate1D`), `feast.rs` 8 (`?feast_syev`, `scsrev`, `sbev`, `sygv` and the complex `heev`, `hcsrev`, `hbev`, `hegv`), `lapack/mod.rs` 2 (`LAPACKE_?lacrm`, `?syconv`), `rng` 5 (`v?RngLognormal`, `v?RngBeta`, `v?RngGaussianMV`, `vsl?SSEditMoments`, `vsl?SSEditMissingValues`), `sparse/scalar.rs` 6 (`mkl_sparse_?_create_csr/coo/csc/bsr`, `_mm`, `_dotmv`), `vm/scalar.rs` 1 (`v?LinearFrac`). Count only, as above. Not matched and left as they were: `sparse/mod.rs` `sparse_qr_solve` (no `mkl_sparse_?_qr_solve` prototype in the installed `mkl_spblas.h`), `pde.rs` `init_2d`/`commit_2d`/`solve_2d`, `blas/jit.rs` `jit_create`, `blas/compact.rs` `gepack`/`geunpack`, and the four non-`cblas_*` flagged lines of `blas/scalar.rs`.

## Chunk 3: published safe wrappers (60)

`tools/w1/pub_check.py` takes each flagged safe `pub fn` and looks for a function of the same name with the same argument count in the earliest of the release tags v0.1.0, v0.2.0, v0.3.0, v0.3.4, v0.3.8, v0.3.10, v0.3.11 (same file). All 68 qualify (64 since v0.1.0, 3 since v0.3.8, 1 since v0.3.10). 4 of them share their name with another `pub fn` in the same file, so the evidence is ambiguous and they were left alone; 4 already had a reason from #21; the other 60 now carry `reason = "stable public surface: published since <tag> with <N> arguments; changing the signature breaks consumers"`. Evidence is name plus argument count at the tag, not an identical signature; a published function whose parameter types changed would still pass, which review of the diff against the tag would catch.

## Chunk 4: the rest (this PR closes W1's bare allows)

- **C mirrors (9):** `sparse_qr_solve` (`mkl_sparse_?_qr_solve`, in `mkl_sparse_qr.h`), `jit_create` (`mkl_cblas_jit_create_?gemm`), the three Helmholtz declarations `init_2d` / `commit_2d` / `solve_2d` (`?_init_Helmholtz_2D`, `?_commit_Helmholtz_2D`, `?_Helmholtz_2D`, `mkl_poisson.h`; the trait-level `allow(..., clippy::too_many_arguments)` on `HelmholtzScalar` is replaced by one reasoned allow per method) and the four `mkl_imatcopy` / `omatcopy` / `omatcopy2` / `omatadd` declarations (`mkl_trans.h`). Argument counts equal for every variant, as above.
- **Public surface (6):** `compact::gepack` and `geunpack` (8 arguments since v0.1.0), `mixed_precision::gemm_f16_f32` and `hgemm` (14 since v0.1.0), `fft` `with_scales` (8, since v0.3.10) and `pde` `new` (8, since v0.3.8): argument-count lists compared per release tag.
- **Test mirrors (2):** the Fortran `?gemm_` wrappers in `onemkl/tests/blas_fortran_gemm_map.rs` carry a `reason` (the reference one mirrors the Fortran argument list; the other adds a test-only mutation selector).
- **Restructured (4, private, no API change):** `PackedMatrix::pack` takes a `PackArgs`, `finish_real_result` / `finish_complex_result` take a `FeastStatus`, and `Pardiso::run_phase` takes a `CsrRef`; their bare allows are gone and the whole suite passes.
- **Removed (50):** allows on functions that `clippy` does not flag with the allow overridden (they were no-ops: those functions have 7 arguments or fewer). The two test allows were kept because the test targets are not covered by the `--lib` scan. `cargo clippy -p onemkl --all-targets` reports no `too_many_arguments` after this change.
- **Result (measured by `git grep` on this branch):** no bare `#[allow(clippy::too_many_arguments)]` is left in `onemkl` or `onemkl-sys`; every remaining one carries a `reason`. Of the 245 allow sites found at the start of this chunk, 52 had no flagged function (50 in the library, 2 in the tests); that accounts for most of the earlier gap between the 238 bare allows and the 196 flagged functions, but the arithmetic is not exact (some sites are the second line of a multi-line attribute, some flagged functions are covered by one trait-level allow), so no exact reconciliation is claimed.
