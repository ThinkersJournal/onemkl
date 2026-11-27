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
- Cell indexing for a partition of `nx` breakpoints: `cell = i` for the site in `[x_i, x_{i+1})`, `i = 0..nx-2`; `nx - 1` for the closed last interval; `nx` for sites right of the last breakpoint. (The reference numbers cells from 0 here; `find_cells` converts MKL's own results to 0-based, so compare before relying on this.)
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
