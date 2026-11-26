# Provenance of the `# Safety` sections

Every `unsafe fn` in `onemkl` carries a `# Safety` section. They were not all written the same way; this lists which were read per function and which were derived from a rule. A reviewer who wants to spot-check should start with the formula-derived ones.

| Module | How written | Per-function review |
|---|---|---|
| `vm/scalar.rs` (90) | Formula: one template per signature shape (`n` plus read/write buffers of `n` elements, no partial overlap), applied by script | No. The in-place aliasing sentence is generic; it was not checked per routine against the MKL VM reference |
| `lapack/scalar.rs` (57), `lapack/mod.rs` (4) | Formula: leading-dimension and `stored_rows x stored_cols` rule per pointer argument role, `info` convention; per-routine lengths such as `ipiv` (`n` or `min(m, n)`) taken from the routine's argument names | No. Convention-based, not checked per routine against the LAPACK reference |
| `blas/scalar.rs` | 43 blocks formula-derived (extent / `ld` / layout rule); the rest were already hand-written and kept | Partly |
| `blas/compact.rs`, `blas/packed.rs`, `blas/jit.rs` | Written per function by hand | Yes |
| `sparse/*` | Written per function by hand (pointer retention, extents, layout) | Yes |
| `rng/*`, `rng/summary_stats.rs`, `rng/convolution.rs` | Written per function by hand | Yes |
| `feast.rs`, `pde.rs`, `data_fitting.rs` | Written per function by hand | Yes |

Array extents stated in these docs fall in two classes. **Verified** means checked against the installed oneMKL 2026.2 headers or by a guard-page probe (`tools/size-probes`): FEAST dense `fpm` = 128, trig-transform `dpar` = `5n/2 + 2` (tight), compact-format size and `gemm_pack_get_size` are in bytes, Data Fitting and the listed Summary Statistics extents. **Unverified** extents are marked as such in the docs; they are FEAST csr/banded/general, Summary Statistics robust-covariance / outliers / missing-values / stream quantiles, `corr_exec`, and the trig `ipar` = 128 (the documentation says 128; the measured minimum is 15).

None of the sections was compiled against a sanitiser; rustdoc and the test suites only prove they are well-formed and that the calls behave. Waivers: none (no `#[allow]` added; the six `too_many_arguments` allows are board 509).
