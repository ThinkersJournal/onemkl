# Size probes

C programs that measure the buffer extents quoted in the `# Safety` docs. Each allocates the buffer under test so it ends at a
`PROT_NONE` guard page, runs the MKL call in a forked child and reports the smallest length for which the call does not fault
(reads and writes both fault). `harness control` in `trig_conv_feast.c` shows the method returns 8 for a routine that reads 8 doubles.

Build and run (Linux, oneMKL 2026.2 installed under `$MKLROOT`):

    gcc -O0 -I$MKLROOT/include trig_conv_feast.c -o probe -L$MKLROOT/lib -lmkl_rt -Wl,--no-as-needed -lm
    MKL_THREADING_LAYER=SEQUENTIAL ./probe

Results at the time of writing are quoted in the doc comments, with the oneMKL version. A guard-page probe gives the extent MKL
touches for the arguments used, not a documented guarantee; see each doc comment for what was and was not covered.

`feast2.c` (FEAST csr / banded / generalised dense) and `ss2.c` (Summary Statistics robust covariance, outlier weights, stream quantiles) use the same method (`gcc ... feast2.c`, same flags). `ss3.c` (2026-11-27): `corr_exec` extents verified (same `1 + (shape - 1) * stride` rule as convolution, strides 1 and 2); trig `ipar` was already measured by `trig_conv_feast.c` (smallest 15 entries for n = 64, six transform types; the reference's 128 is kept as the requirement). `ss_edit_missing_values` could not be verified: no MI parameter set tried (including a `k * m` simulated-values buffer) made `VSL_SS_METHOD_MI` compute, and `SummaryStats::impute_missing` as shipped passes a null `simul_vals_out`, for which the compute step returns -4054 (`VSL_SS_ERROR_BAD_MI_SIMUL_VALS_ADDR`).

`fft_flags.c` records which `DFTI_WORKSPACE` / `DFTI_ORDERING` settings oneMKL 2026.2 accepts. `df_probe3.c` records the Data Fitting call-back, Hyman and quadratic behaviour (see `docs/DATA-FITTING-FACTS.md`, section "Measured"). An earlier version of this line cited a `df_types.c` that was never committed and a Hyman non-monotone result from it; that result is withdrawn here (the Hyman data point is re-measured in `df_probe3.c`).
