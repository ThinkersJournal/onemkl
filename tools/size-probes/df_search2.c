// SPDX-License-Identifier: MIT OR Apache-2.0
/* search_cb inside dfdIntegrateEx1D and dfdInterpolateEx1D (oneMKL 2026.2): how often it is called, n, library_params / limit_type_flag, and whether it changes the result. 
 * Build like df_search.c. */
#include <stdio.h>
#include "mkl_df.h"
#define NX 6
static const double X[NX] = {0, 1, 2, 3, 4, 5};
static const double Y[NX] = {0, 1, 2, 3, 4, 5};
static int g_exact;
static int cb(long long *n, double *site, long long *cell, int *flag, void *user, dfSearchCallBackLibraryParams *lib) {
  printf("  search cb n=%lld lib=%s", *n, lib ? "set" : "NULL");
  if (lib) printf(" limit_type_flag=%d (LLIM=%d RLIM=%d)", lib->limit_type_flag, DF_INTEGR_SEARCH_CB_LLIM_FLAG, DF_INTEGR_SEARCH_CB_RLIM_FLAG);
  printf(" site0=%g\n", site[0]);
  for (long long i = 0; i < *n; i++) { long long c = (long long)site[i] + 1; if (c > NX - 1) c = NX - 1; if (c < 1) c = 1; cell[i] = c; flag[i] = g_exact; }
  return 0;
}
static DFTaskPtr mk(void) {
  DFTaskPtr t = 0; static double sc[2 * (NX - 1)];
  dfdNewTask1D(&t, NX, X, DF_NON_UNIFORM_PARTITION, 1, Y, DF_NO_HINT);
  dfdEditPPSpline1D(t, DF_PP_LINEAR, DF_PP_DEFAULT, DF_NO_BC, NULL, DF_NO_IC, NULL, sc, DF_NO_HINT);
  dfdConstruct1D(t, DF_PP_SPLINE, DF_METHOD_STD);
  return t;
}
int main(void) {
  for (g_exact = 0; g_exact <= 1; g_exact++) {
    printf("flag=%d\n", g_exact);
    DFTaskPtr t = mk();
    double l[3] = {0.5, 1.5, -1.0}, r[3] = {2.5, 4.5, 7.0}, out[3] = {0};
    int rc = dfdIntegrateEx1D(t, DF_METHOD_PP, 3, l, DF_NO_HINT, r, DF_NO_HINT, NULL, NULL, out, DF_NO_HINT,
                              NULL, NULL, NULL, NULL, NULL, NULL, cb, NULL);
    printf(" integrate rc=%d out=%g %g %g (exact 2, 3, ?)\n", rc, out[0], out[1], out[2]);
    double site[3] = {0.5, 2.5, 4.5}, v[3] = {0}; MKL_INT dorder[1] = {1};
    rc = dfdInterpolateEx1D(t, DF_INTERP, DF_METHOD_PP, 3, site, DF_NON_UNIFORM_PARTITION, 1, dorder, NULL, v, DF_NO_HINT, NULL,
                            NULL, NULL, NULL, NULL, NULL, NULL, cb, NULL);
    printf(" interpolate rc=%d v=%g %g %g\n", rc, v[0], v[1], v[2]);
    dfDeleteTask(&t);
  }
  return 0;
}
