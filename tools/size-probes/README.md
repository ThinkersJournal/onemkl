# Size probes

C programs that measure the buffer extents quoted in the `# Safety` docs. Each allocates the buffer under test so it ends at a
`PROT_NONE` guard page, runs the MKL call in a forked child and reports the smallest length for which the call does not fault
(reads and writes both fault). `harness control` in `trig_conv_feast.c` shows the method returns 8 for a routine that reads 8 doubles.

Build and run (Linux, oneMKL 2026.2 installed under `$MKLROOT`):

    gcc -O0 -I$MKLROOT/include trig_conv_feast.c -o probe -L$MKLROOT/lib -lmkl_rt -Wl,--no-as-needed -lm
    MKL_THREADING_LAYER=SEQUENTIAL ./probe

Results at the time of writing are quoted in the doc comments, with the oneMKL version. A guard-page probe gives the extent MKL
touches for the arguments used, not a documented guarantee; see each doc comment for what was and was not covered.

`feast2.c` (FEAST csr / banded / generalised dense) and `ss2.c` (Summary Statistics robust covariance, outlier weights, stream quantiles) use the same method (`gcc ... feast2.c`, same flags). Not probed: `ss_edit_missing_values` (needs NaN data and an MI parameter set), `corr_exec`, trig `ipar` = 128.

`df_types.c` and `fft_flags.c` record which Data Fitting spline types and which `DFTI_WORKSPACE` / `DFTI_ORDERING` settings oneMKL 2026.2 accepts. Result worth keeping: `DF_PP_HYMAN` (cubic, `DF_BC_NOT_A_KNOT`) constructs, but on 8 knots of monotone plateau data its worst downward step between knots was 0.0082 against 0.0162 for a natural spline, so it does not keep the data monotone and `onemkl` does not wrap it; linear (`DF_PP_LINEAR`, `DF_PP_DEFAULT`, no BC) constructs and quadratic variants tried here returned -1012 / -1014.
