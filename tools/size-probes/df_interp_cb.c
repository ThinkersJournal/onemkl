// SPDX-License-Identifier: MIT OR Apache-2.0
/* Data Fitting interpolation call-backs of dfdInterpolateEx1D (le_cb, re_cb, i_cb) on oneMKL 2026.2: which sites reach which call-back, n, the cells passed in,
 * how r is laid out for two derivative orders, and what a negative / positive call-back status does to the routine's status and result.
 * Linear spline y = x on knots 0..5. Sites -1, 0.5, 2.5, 5, 7 (then, last, unsorted 2.5 -1 7 0.5 5) (left of, inside, inside, at the end, right of). Each call-back writes 100*tag + site (tag 1 le, 2 re, 3 i) so the result shows who served which site.
 * Build: gcc -O1 df_interp_cb.c -I$MKLROOT/include -L$MKLROOT/lib -lmkl_rt -lm -o df_interp_cb */
#include <stdio.h>
#include "mkl_df.h"
#define NX 6
#define NS 5
static const double X[NX] = {0, 1, 2, 3, 4, 5};
static int g_status;
static int g_nord;
static double g_site[NS] = {-1, 0.5, 2.5, 5, 7};
static int cbk(int tag, const char *name, long long *n, long long *cell, double *site, double *r, dfInterpCallBackLibraryParams *lib) {
  printf("  %s n=%lld lib=%s\n   sites:", name, *n, lib ? "set" : "NULL");
  for (long long i = 0; i < *n; i++) printf(" %g", site[i]);
  printf("\n   cells:");
  for (long long i = 0; i < *n; i++) printf(" %lld", cell[i]);
  printf("\n");
  for (int k = 0; k < g_nord; k++)
    for (long long i = 0; i < *n; i++) r[i * g_nord + k] = 100.0 * tag + site[i] + 1000.0 * k;
  return g_status;
}
static int le(long long *n, long long *c, double *s, double *r, void *u, dfInterpCallBackLibraryParams *l) { return cbk(1, "le", n, c, s, r, l); }
static int re(long long *n, long long *c, double *s, double *r, void *u, dfInterpCallBackLibraryParams *l) { return cbk(2, "re", n, c, s, r, l); }
static int ic(long long *n, long long *c, double *s, double *r, void *u, dfInterpCallBackLibraryParams *l) { return cbk(3, "i", n, c, s, r, l); }
static DFTaskPtr mk(void) {
  DFTaskPtr t = 0; static double sc[2 * (NX - 1)], y[NX] = {0, 1, 2, 3, 4, 5};
  dfdNewTask1D(&t, NX, X, DF_NON_UNIFORM_PARTITION, 1, y, DF_NO_HINT);
  dfdEditPPSpline1D(t, DF_PP_LINEAR, DF_PP_DEFAULT, DF_NO_BC, NULL, DF_NO_IC, NULL, sc, DF_NO_HINT);
  dfdConstruct1D(t, DF_PP_SPLINE, DF_METHOD_STD);
  return t;
}
static void run(const char *what, int use_le, int use_re, int use_i, int nord, int st) {
  g_status = st; g_nord = nord;
  printf("%s (le=%d re=%d i=%d dorders=%d cb status=%d)\n", what, use_le, use_re, use_i, nord, st);
  DFTaskPtr t = mk();
  double site[NS], v[2 * NS], mark = -777;
  for (int i = 0; i < NS; i++) site[i] = g_site[i];
  for (int i = 0; i < 2 * NS; i++) v[i] = mark;
  MKL_INT dorder[2] = {1, 1}; long long cell[NS] = {-9, -9, -9, -9, -9};
  int rc = dfdInterpolateEx1D(t, DF_INTERP, DF_METHOD_PP, NS, site, DF_NON_UNIFORM_PARTITION, nord, dorder, NULL, v, DF_NO_HINT, (MKL_INT *)NULL,
                              use_le ? le : NULL, NULL, use_re ? re : NULL, NULL, use_i ? ic : NULL, NULL, NULL, NULL);
  printf(" rc=%d r:", rc);
  for (int i = 0; i < nord * NS; i++) printf(" %g", v[i]);
  printf("\n");
  (void)cell;
  dfDeleteTask(&t);
}
int main(void) {
  run("control: no call-backs", 0, 0, 0, 1, 0);
  run("le only", 1, 0, 0, 1, 0);
  run("re only", 0, 1, 0, 1, 0);
  run("i only", 0, 0, 1, 1, 0);
  run("all three", 1, 1, 1, 1, 0);
  run("control: no call-backs, two derivative orders", 0, 0, 0, 2, 0);
  run("all three, two derivative orders (value, 1st derivative)", 1, 1, 1, 2, 0);
  run("le only, two derivative orders (library serves the rest)", 1, 0, 0, 2, 0);
  run("re only, two derivative orders (library serves the rest)", 0, 1, 0, 2, 0);
  run("i only, two derivative orders (library serves the rest)", 0, 0, 1, 2, 0);
  run("all three, negative status -5", 1, 1, 1, 1, -5);
  run("all three, positive status 3", 1, 1, 1, 1, 3);
  { double u[NS] = {2.5, -1, 7, 0.5, 5}; for (int i = 0; i < NS; i++) g_site[i] = u[i]; }
  run("UNSORTED sites 2.5 -1 7 0.5 5, all three, no hint", 1, 1, 1, 1, 0);
  return 0;
}
