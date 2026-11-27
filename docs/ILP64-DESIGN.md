<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# ILP64 for the safe `onemkl` wrappers: design note and schedule

Status: **implemented** (2026-11-27, `lane/ilp64-int`); this note keeps the design, the counts and the
package plan. `onemkl-sys` supported `ilp64` from the start; `onemkl` refused it in `onemkl/build.rs`
until this change.

## What ILP64 changes

`MKL_INT` is `int` (32 bit) under LP64 and `long long` (64 bit) under ILP64, so every extent,
stride, leading dimension, index array and count in the MKL C API doubles in width. Pointers to
index arrays (`ipiv`, CSR `row_ptr` / `col_idx`, PARDISO `perm`) change element type, so this is
not a cast away: a `&[i32]` can no longer be handed to MKL.

## Measured error counts (corrected)

An earlier version of this note, and the `ROADMAP` row it was written from, said "about 590
`E0308`" (342 mismatched types, 197 wrong arguments, 52 `?` conversions). That figure is withdrawn:
it was a hand-run count whose feature set and commit were not recorded, and it did not reproduce.
Re-measured on 2026-11-27:

| what | command | errors |
|---|---|---|
| main e3c72cc, library only | `cargo check -p onemkl --no-default-features --features ilp64,threading-sequential,link-dynamic,<all 13 domains>`, real bindgen (clang on the ILP64 headers, oneMKL 2026.2, WSL Ubuntu 24.04), `build.rs` guard bypassed | **412** (174 mismatched `iN`/`iN`, 138 wrong arguments, 52 `?` conversions, 48 pointer mismatches) |
| same, per file | | `lapack/scalar.rs` 95, `optim.rs` 56, `lapack/mod.rs` 50, `blas/mixed_precision.rs` 42, `pde.rs` 25, `dss.rs` 21, `rng/mod.rs` 18, `iss.rs` 16, `feast.rs` 16, `blas/compact.rs` 15, `sparse/scalar.rs` 12, then 8 or fewer in nine files |
| after package 1 (`blas/`, `lapack/`) | same | 200 (0 in `blas/`, `lapack/`) |
| after packages 2 and 3, main 845b59e merged | same, plus `--all-targets` (tests, examples) | **0** |

Most extents already spelled `MKL_INT` in `blas/` and `lapack/` (about 1,260 uses); the rest used
`c_int` or `i32` for values the C API declares `MKL_INT`. Not every `c_int` is an `MKL_INT`: statuses,
flags, enum values and `layout` arguments stay `int`, and the compiler told them apart.

## Options considered

### A. Keep refusing
Cost zero; consumers needing ILP64 lose every safe wrapper, and matrices with more than `i32::MAX`
rows, columns or nonzeros are out of reach. Rejected: demand orders work, it does not decide
whether (CireSnave's standing rule, relayed by pm-kernels 2026-11-27).

### B1. Alias `MKL_INT`, selected by the feature (chosen)
Spell every `MKL_INT` value as `onemkl_sys::MKL_INT` (an alias of `i32` or `i64` chosen by the
`lp64`/`ilp64` feature). One code path; a build has one width, which matches MKL itself (one
interface layer per process). Under `lp64` nothing changes for callers.

### B2. True generics over a sealed width trait
Both widths cannot coexist in one build (a single `mkl_intel_*` interface layer), so generics buy
nothing at run time and worsen every signature. Not done.

### C. Separate `onemkl-ilp64` crate
A second copy of the crate or a macro fork; drift between copies. Not done.

## What was done (packages, in the order they landed)

1. `blas/`, `lapack/`: extents, `ipiv`, `info`, `isuppz`, `itype`, found-eigenvalue counts, compact sizes.
2. `sparse/`, `pardiso`, `dss`, `iss`, `preconditioners`: index arrays, handles, RCI codes.
3. `feast`, `pde`, `optim`, `data_fitting`, `rng/` (including summary statistics and convolution), `service`.
4. `build.rs` refusal removed; tests, examples and doctests moved to `MKL_INT`; a CI step runs the whole
   suite against the `mkl_intel_ilp64` libraries.

Public items that took `i32` for such values now take `MKL_INT`. That is the same type under `lp64`, so it
is not a break for existing callers; `Error` variants keep their `i32` payloads and saturate.

## Which consumers would use ILP64

Read at `origin/main` of each repo on 2026-11-27 (`git grep -i -E "ilp64|MKL_INT"`; only these three
were checked, not every repo under `C:\Projects`).
- `cpufish-onemkl`: depends on `onemkl` with `features = ["lp64", ...]`, uses only
  `blas::level3::gemm`, `Layout`, `MatrixRef`, `MatrixMut`, `Transpose`. Explicitly LP64.
- `fuel-mkl-cpu-backend` (fuel): `onemkl = "0.4"` through the workspace; uses `blas::level*`,
  `matrix`, `enums`, `service::{enable_instructions, AlignedBuffer}`. No `ilp64` feature set.
- `lightbulb`: no direct `onemkl` dependency; it reaches it only through fuel.

No consumer needs ILP64 today, which only ranks the work below consumer-needed rows.
