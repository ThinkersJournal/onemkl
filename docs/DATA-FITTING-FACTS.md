# Data Fitting facts read from the Intel reference

Source: Intel oneMKL Developer Reference (C), 2026.0, the PDF at
<https://cdrdv2.intel.com/v1/dl/getContent/916344?fileName=onemkl_developer-reference-c_2026.0-766684-916344.pdf>
(HTML pages are under
<https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/>;
`df-integrate1d-df-integrateex1d.html`, `df-integrcallback.html`,
`df-searchcellscallback.html`, `df-interpcallback.html`, the Data Fitting task
editor pages). Fetched 2026-11-26. The wording below is ours; credit: "Intel
oneMKL Developer Reference, <URL above>". Facts only; the wrappers that use
them (Hyman, integration call-backs) are separate changes. Where the reference and a header or a measurement disagree, the header or
the measurement is stated as such.

Positive control (a fact we measured before reading): the Hermite cubic takes
its interior slopes as an internal-condition array of `n - 2` first
derivatives, and the two end-point slopes as boundary conditions
`DF_BC_1ST_LEFT_DER | DF_BC_1ST_RIGHT_DER` in a 2-element array (left, then
right). The reference says exactly that. It matches.

## IntegrateEx1D / InterpolateEx1D call-backs

- Argument tail of `df?IntegrateEx1D`: `le_cb, le_params, re_cb, re_params,
  i_cb, i_params, search_cb, search_params`. Any of them may be NULL (the
  library then uses its default behaviour for that region).
- Integration call-backs by region, with `[a, b]` the interpolation interval:
  `le_cb` serves `[llim, min(rlim, a))` when `llim < a`; `re_cb` serves
  `[max(llim, b), rlim)` when `rlim` is at or beyond `b`; `i_cb` serves
  `[max(a, llim), min(rlim, b))`. The library splits each limit pair into those
  pieces and sums the pieces. A pair that lies wholly inside `[a, b]` reaches
  `i_cb` unchanged. A reversed pair (`rlim < llim`) gives the negated integral.
- The reference's argument table types `re_cb` as an interpolation call-back
  type, while the three call-backs are all integration call-backs everywhere
  else and in `mkl_df_functions.h` (all three are `df?IntegrCallBack`). Treat
  that table row as a typo; the header is authoritative.
- Integration call-back arguments: `n` (pairs, `long long*`), `lcell`/`rcell`
  (cell indices containing each limit), `llim`/`rlim` (limits), `r` (results,
  packed as the interpolation page describes), `user_params`, `library_params`
  (currently carries nothing).
- Interpolation call-backs (`df?InterpolateEx1D`): `le_cb` left of `[a, b]`,
  `re_cb` right of it, `i_cb` inside it; arguments `n, cell, site, r,
  user_params, library_params`; `r` is row-major.
- **Measured (oneMKL 2026.2, `tools/size-probes/df_interp_cb.c`, linear spline y = x on 0..5, sites -1, 0.5, 2.5, 5, 7, `dfdInterpolateEx1D`, one derivative order):** `le_cb` is called once with the sites left of the first breakpoint (`n` 1, site -1, cell 0), `i_cb` once with the sites from the first to the last breakpoint inclusive (`n` 3: 0.5, 2.5, 5; cells **0-based** 0, 2, 4; the end point 5 gets the last cell 4), `re_cb` once with the sites right of the last (`n` 1, site 7, cell 4 = `nx - 2`); order of calls le, i, re; `library_params` NULL. A call-back that is not given leaves its region to the library (control: no call-backs gives the plain values -1, 0.5, 2.5, 5, 7). With unsorted sites 2.5, -1, 7, 0.5, 5 the call-backs receive their sites in the order passed (2.5, 0.5, 5) and the values land at the original positions. A negative status (-5) from `le_cb` made the routine return -5 and skip the later call-backs (r partly written, the rest 0); a positive status (3) made all call-backs run and the routine return 3 with the results kept (not swallowed, unlike the search call-back). With two derivative orders (`dorder` {1, 1}) the result came back as the le block, then the i block, then the re block, each as the call-back wrote it: not a plain per-site layout, so derivative orders are not wrapped.
- Return contract for every call-back: 0 success, negative error, positive
  warning. The reference does not say how the routine's own status reflects a
  non-zero call-back status; that needs a probe before it is documented.
- Search call-back: arguments `n, site, cell, flag, user_params,
  library_params`. For each site it writes `cell[i]`; if that cell is exact it
  sets `flag[i] = 1`, otherwise 0 and the library treats the index as a start
  for its own search. Returning `DF_STATUS_EXACT_RESULT` declares all indices
  exact, so `flag` need not be filled. In `IntegrateEx1D` the library passes
  `library_params` with a `limit_type_flag` of `DF_INTEGR_SEARCH_CB_LLIM_FLAG`
  or `DF_INTEGR_SEARCH_CB_RLIM_FLAG` saying which limits it is searching; the
  other two routines pass NULL.
- **Measured (oneMKL 2026.2, Linux x86_64, `tools/size-probes/df_search.c`, 6 breakpoints 0..5, sites 0.5, 1.5, 2.5, 4.5, `dfdSearchCellsEx1D` with a call-back):** the call-back is called once with `n` = number of sites, `library_params` NULL (and `user_params` as passed). The library's own search returns cells **1-based** (1, 2, 3, 5), and a call-back that sets `flag[i] = 1` (or returns `DF_STATUS_EXACT_RESULT`) has its cells copied through unchanged, so an exact call-back must return **1-based** cells (0-based ones came back as 0, 1, 2, 4). With `flag[i] = 0`, or a nonsense flag value 7, the library corrected both 0-based and 1-based starting guesses to 1, 2, 3, 5. A call-back returning -5 made the routine return -5 and left `cell` untouched; returning 3 made it return 3 with the cells filled. This contradicts the 0-based numbering in the next bullet, which describes the reference's wording and is not what this build does for the search routine. `IntegrateEx1D` / `InterpolateEx1D` with a `search_cb` (`tools/size-probes/df_search2.c`, linear spline on 0..5, limits (0.5, 2.5), (1.5, 4.5), (-1, 7), 3 sites 0.5, 2.5, 4.5): `IntegrateEx1D` calls it twice, each time with `n` = 3 and `library_params` set, first with `limit_type_flag` = 0 (`DF_INTEGR_SEARCH_CB_LLIM_FLAG`, left limits; site 0.5 first) then 1 (`..._RLIM_FLAG`, right limits; site 2.5 first); `InterpolateEx1D` calls it once with `n` = 3 and `library_params` NULL. With 1-based exact cells (`flag` 1) and with `flag` 0 the results were the same and correct (integrals 3, 9, 24; values 0.5, 2.5, 4.5). Not measured: a search call-back that returns wrong cells inside those two routines.
- Cell indexing for a partition of `nx` breakpoints, **as measured on oneMKL 2026.2 (corrects an earlier 0-based statement here, 2026-11-27; `tools/size-probes/df_search.c`, `df_search2.c`)**: 1-based. The site in `[x_i, x_{i+1})` is cell `i + 1`, `i = 0..nx-2`; sites in the closed last interval give `nx - 1`; the integration call-backs pass 0 left of the interpolation interval and `nx` at or right of its end. `find_cells` converts to 0-based. The reference's wording is not what this build does for the search routine.
- The task keeps raw pointers to `bc`, `ic` and `scoeff`; they must outlive
  every computation (already honoured by our wrappers).

## Spline types, boundary and internal conditions

- Minimum breakpoints: default linear/quadratic 2, default cubic 3, Subbotin
  2, natural 2, Hermite 3, Bessel 4, Hyman 4, Akima 5.
- Boundary conditions: the reference lists `DF_NO_BC` for linear and all cubic
  types. Not-a-knot is listed for Akima, Bessel, Hermite and natural cubic;
  free-end and first/second end derivatives for those plus Subbotin quadratic;
  periodic for linear and all cubic; `DF_BC_Q_VAL` (function value at
  `(x_0 + x_1) / 2`) for the default quadratic. Derivative conditions combine
  with bitwise OR and travel as a 2-element array (left, right). Priority when
  combined: not-a-knot, then free-end, then periodic, then first derivative
  over second.
- Hyman: the reference lists no required boundary condition and periodic
  support; it is not in the not-a-knot row. What this build does, measured, is
  in the section "Measured on oneMKL 2026.2" below (`DF_NO_BC` fails, free-end
  and end-derivative conditions keep plateau data monotone, not-a-knot does
  not).
- Quadratics: default quadratic is built with `DF_PP_QUADRATIC`, type
  `DF_PP_DEFAULT` and the `DF_BC_Q_VAL` condition (one value). Subbotin uses
  free-end or end derivatives and an optional knot array through
  `DF_IC_Q_KNOT`: `n + 1` knots with `t_0 = x_0`, `t_n = x_{n-1}` and each
  interior knot strictly inside its cell, or 4 values with
  `DF_UNIFORM_PARTITION` OR-ed into the IC type. Without knots, midpoints are
  used. Neither quadratic supports periodic.
- Internal conditions: Hermite `DF_IC_1ST_DER` (`n - 2` values); default cubic
  `DF_IC_2ND_DER` (`n - 2`); Subbotin `DF_IC_Q_KNOT` as above.
- The two earlier quadratic failures (-1012, -1014, "wrong arguments") are
  consistent with a missing `DF_BC_Q_VAL` value or missing end conditions:
  probe those before concluding anything.

## Measured on oneMKL 2026.2 (tools/size-probes/df_probe3.c)

One partition note first: when `x` is declared with `DF_UNIFORM_PARTITION`, MKL
reads only the first two entries of `x` as the end points. Declare a general
partition as `DF_NON_UNIFORM_PARTITION` (an early run of the probe tripped on
this and misreported the call-back regions).

Integration call-backs, linear spline on `[0, 7]`, call-backs that return
`100 + (rlim - llim)`:

- Regions match the reference: `[1,3]` gives one `i_cb` call with `(1,3)`;
  `[-2,9]` gives `le_cb(-2,0)`, `i_cb(0,7)`, `re_cb(7,9)`; `[8,9]` gives only
  `re_cb(8,9)`. Each call has `n = 1`; the result is the sum of the pieces.
- Cell arguments are 1-based (`[1,3]` arrives as `lcell = 2, rcell = 4`), `0`
  left of `a`, and `nx` (8) for a limit at or right of `b`; a call-back must
  not treat them as 0-based indices.
- A reversed pair `[3,1]` reaches the call-back as `(1,3)` and the library
  negates the sum (`-102`).
- A positive call-back status is returned as the routine's status (5) and the
  result is still produced. A negative status is returned as the routine's
  status (-3) and `r` is left untouched.

Spline construction (8 knots of monotone plateau data):

- Hyman, worst downward step between samples on the plateau data (natural
  cubic control 0.0243): `DF_BC_FREE_END`, end first derivatives 0, and end
  second derivatives 0 all 0.0 (monotone); `DF_BC_NOT_A_KNOT` 0.0140 (not
  monotone); `DF_NO_BC` fails with -1014 (the reference lists it as
  supported, so this build diverges). An earlier version of this section
  claimed the derivative stayed non-negative; that came from a probe that
  printed the minimum value, not the downward step, and is withdrawn. The
  reference spells the second-derivative flags `DF_BC_2ST_LEFT_DER`; the
  header has `DF_BC_2ND_LEFT_DER`.
- Default quadratic: needs `DF_BC_Q_VAL` with one value (`DF_NO_BC` gives
  -1014).
- Subbotin: constructs only with `DF_BC_FREE_END` plus a `DF_IC_Q_KNOT` array
  of `n + 1` knots; without the knots, or with `DF_NO_BC`, `DF_BC_NOT_A_KNOT`,
  free-end or end derivatives alone, it returns -1012; `DF_NO_BC` with the
  knots returns -1014.

Coefficient extent (`tools/size-probes/df_coeff_extent.c`, 8 knots, sentinel
buffer): linear 14, every cubic type 28, default quadratic 21, all equal to
`order * (nx - 1)`; the Subbotin quadratic wrote 24 = `order * nx`, three more
than the reference's `order * (nx - 1)` = 21. A buffer sized by the reference
is overrun (stack smashing seen in the probe). Also: the Subbotin spline did
not pass through the breakpoints on `y = x^2` (value 0.29 at `x = 0`).

- **Measured (oneMKL 2026.2, `tools/size-probes/df_search3.c`, same setup as `df_search2.c`): a search call-back returning wrong cells inside `IntegrateEx1D` / `InterpolateEx1D`.** All with `flag` 1: correct 1-based cells reproduce integrals 3, 9, 24 and values 0.5, 2.5, 4.5; 0-based cells and cell 0 gave the same correct results (the routines tolerate them); cell 99 gave 0, 0, 0 with status 0 in both routines (silent, no error). A call-back returning -5 made both routines return -5 and left the output untouched; returning 3 made both return 0 with correct results (a positive status is swallowed). `IntegrateEx1D` is wrapped (`IntegrateCallbacks::search`) with the exact cells checked and a positive status turned into an error; `InterpolateEx1D` is wrapped for function values (`interpolate_with_callbacks`, `interpolate_with_search_callback`; same search checks).
