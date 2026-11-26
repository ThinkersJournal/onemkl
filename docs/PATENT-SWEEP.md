# Patent sweep: onemkl 0.3.11 (self-serve; not legal advice; no attorney)

Version line covered: 0.3 (workspace `Cargo.toml` version 0.3.11 at `origin/main` 5c3422d; before 1.0 the second number is the major). Method: grepping `onemkl/src` to characterise what the crate does. **No patent was read for onemkl and no patent-database search was run**; nothing here says the project is clear of any patent.

## Scope

`onemkl` is a Rust binding to Intel oneMKL: its public functions validate arguments and call the vendor library (`onemkl-sys` bindings generated from the oneMKL headers). The algorithms (BLAS, LAPACK, FFT, sparse solvers, RNG, summary statistics, data fitting and the rest) run inside Intel's library, which the user installs separately under Intel's licence; this repository does not ship it.

Measured at 5c3422d: `grep -rn 'sys::' onemkl/src` finds 1413 call sites in 27 files; the same query is the positive control for "the crate reaches the library through `sys::`" (it finds `sys::DftiSetValue` in `fft.rs`). That count says where the numeric work happens; it does not prove the crate has no own algorithm: small pure-Rust glue exists (dimension and stride checks, buffer-length helpers such as `cce_complex_len` in `fft.rs`, a reference evaluator in tests). No per-function audit was done.

## Independent claims read

None. No patent claim was read for this sweep.

## Verdicts

| Idea | Verdict |
|---|---|
| Numerical algorithms implemented by oneMKL and called through this crate | NOT CLEARED, unread |
| Pure-Rust glue in the crate (argument validation, buffer-length helpers) | NOT CLEARED, unread |

Allowed verdicts here are "outside as read" (needs a claim actually read), "needs more" (with the limb quoted) and "NOT CLEARED, unread". Neither of the two rows above has a claim read against it, so neither can be better than "NOT CLEARED, unread".

## What would change this

A named patent from CireSnave's rulings, or a search result, read against the code: record its independent claim 1 verbatim, the limbs the crate does or does not perform, the query used and a positive control for each absence, then give the verdict above.
