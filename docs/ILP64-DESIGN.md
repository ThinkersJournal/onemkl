<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# ILP64 for the safe `onemkl` wrappers: design note

Status: decision wanted, no code. Measured 2026-11-27 at `origin/main` e3c72cc, oneMKL 2026.2.
`onemkl-sys` supports `ilp64`; `onemkl` refuses it in `onemkl/build.rs`.

## What ILP64 changes

`MKL_INT` is `int` (32 bit) under LP64 and `long long` (64 bit) under ILP64, so every extent,
stride, leading dimension, index array and count in the MKL C API doubles in width. Pointers to
index arrays (`ipiv`, CSR `row_ptr` / `col_idx`, PARDISO `perm`) change element type, so this is
not a cast away: a `&[i32]` can no longer be handed to MKL.

## Measured error classes (real bindgen, `--features ilp64`)

About 590 `E0308`: 342 mismatched types, 197 wrong argument types, 52 `?` conversions.

Where the width appears in the sources (word counts by `grep -w`, so approximate and including
comments):

| area | `c_int` | `MKL_INT` | `i32` |
|---|---|---|---|
| `blas/` | 247 | 654 | 10 |
| `lapack/` | 185 | 610 | 199 |
| `sparse/` | 91 | 0 | 21 |
| `rng/` | 216 | 26 | 16 |
| `feast.rs` | 210 | 0 | 5 |
| `data_fitting.rs` | 177 | 0 | 3 |
| `pde.rs` | 141 | 0 | 0 |
| `optim.rs` | 54 | 0 | 5 |
| `iss.rs` | 39 | 0 | 2 |
| `pardiso.rs` | 32 | 0 | 20 |
| `service.rs` | 27 | 9 | 13 |
| `dss.rs` | 22 | 0 | 5 |

Not every `c_int` is an `MKL_INT`: many are status codes, option flags and enum values, which stay
`int` under ILP64. Telling them apart is the real work. `fft.rs` already uses a width alias, so it
is the template. `blas/` and `lapack/` already spell `MKL_INT` (about 1,260 uses), so they are the
cheapest; `feast`, `data_fitting`, `pde`, `optim`, `iss`, `pardiso`, `dss` and `sparse` hard-code
`c_int` or `i32`, and `lapack/` has 199 `i32` that must be audited one by one (`ipiv` and `info`
are the likely suspects; not yet audited).

## Options

### A. Keep refusing (status quo)
Cost: zero. Consumers needing ILP64 use `onemkl-sys` directly and lose every safe wrapper.
Matrices with more than `i32::MAX` rows, columns or nonzeros, or a dense matrix with a leading
dimension above 2^31 - 1, are out of reach.

### B. Generic over the width via a sealed trait
Introduce `trait MklInt: Copy + sealed` implemented for `i32` and `i64`, and make every wrapper
that takes or returns index data generic over it (or alias it to `sys::MKL_INT` per build).
- B1, alias only (`pub type Int = sys::MKL_INT`, feature-selected): one code path, mechanical,
  about the size of the 590 errors plus the audit above. Public signatures change with the
  feature, so the two builds are different APIs behind one name. Feature unification means one
  crate graph can only have one width: a library depending on `onemkl` with `lp64` and another
  with `ilp64` cannot both link. That is already true of MKL itself (one interface layer per
  process), so it matches the library.
- B2, true generics (`fn getrf<I: MklInt>(...)`): both widths in one build is not possible at
  link time anyway (a single `mkl_intel_*` interface layer), so the generics buy nothing at
  runtime and make every signature and error message worse. Not recommended.

Cost: B1 is the 590 errors plus the audit above (not estimated in days; no attempt made), plus an ILP64 CI row, one rig job, and a
breaking release note (the alias is the same `i32` under `lp64`, so lp64 callers see no change
if `Int` is introduced as an alias to `i32`; it is breaking only where a public type is replaced
by the alias, so audit `pub` items first).

### C. Separate `onemkl-ilp64` crate
Cost: a second copy of the whole crate, or a macro-generated fork; two crates to version, test and
document, and drift between them. Worst maintenance cost; only worth it if two widths had to
coexist in one build, which MKL does not allow.

## Recommendation and schedule

Option B1 behind the existing `ilp64` feature (CireSnave's standing rule, relayed by pm-kernels
2026-11-27: demand orders work, it never decides whether). An earlier draft of this note said
"keep refusing until a consumer asks"; that is withdrawn. ILP64 in the safe wrappers is an
unblocked row, ranked below consumer-needed work and above idle.

Work packages, in order, one PR per group (PR sizing: each package under ~150k diff characters):

| # | Package | Error class it removes (of the ~590 `E0308`) | Breaking? |
|---|---|---|---|
| 1 | `onemkl::Int = sys::MKL_INT`; `blas/` and `lapack/` use it wherever they mean `MKL_INT`; audit `lapack/`'s 199 `i32` | mismatched types (342 total) in the two largest areas | No under `lp64` (`Int` is `i32`); public items that name `i32` for an extent change type under `ilp64` only |
| 2 | `sparse/`, `pardiso.rs`, `dss.rs`, `iss.rs`, `preconditioners.rs` (index arrays, handles) | wrong argument types (197 total) | Public index-array types become `Int`-typed: flag for pm-assist |
| 3 | `feast.rs`, `pde.rs`, `data_fitting.rs`, `optim.rs`, `rng/`, `service.rs` | `?` conversions (52 total) and the rest | As package 2 |
| 4 | Drop the `build.rs` refusal; CI row | n/a | No |

Each package is measured the same way: build with real bindgen and the ILP64 libraries on the rig
(`--features ilp64`), count `E0308` before and after, and run the tests under both interface
layers (lp64 and ilp64 library paths). The CI row (package 4, started earlier if the rig allows)
runs the existing tests with `--features ilp64`.

## Which consumers would use ILP64

Read at `origin/main` of each repo on 2026-11-27 (`git grep -i -E "ilp64|MKL_INT"`).
- `cpufish-onemkl`: depends on `onemkl` with `features = ["lp64", ...]`, uses only
  `blas::level3::gemm`, `Layout`, `MatrixRef`, `MatrixMut`, `Transpose`. Explicitly LP64. 0 hits
  for ILP64. A GEMM with a dimension above 2^31 - 1 elements per side is not a CPU-inference
  shape.
- `fuel-mkl-cpu-backend` (fuel): `onemkl = "0.4"` through the workspace; uses `blas::level*`,
  `matrix`, `enums`, `service::{enable_instructions, AlignedBuffer}`. No `ilp64` feature set. 0 hits.
- `lightbulb`: 0 Cargo.toml hits for `onemkl` itself; it reaches it only through fuel.

(Only the three named consumers were checked, not every repo under `C:\Projects`.)

Result: no consumer needs ILP64 today, which only ranks this work below consumer-needed rows. The
wrappers a consumer would hit first (`blas`, `lapack`) are also the cheapest to convert, so
package 1 alone would serve GEMM-class use.
