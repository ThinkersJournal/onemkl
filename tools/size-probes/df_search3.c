// SPDX-License-Identifier: MIT OR Apache-2.0
/* search_cb inside dfdIntegrateEx1D / dfdInterpolateEx1D (oneMKL 2026.2): what happens when the call-back returns WRONG exact cells (0-based, i.e. shifted down by one; far out of range), or a status (-5, 3). Controls: the correct 1-based exact cells (must reproduce integrals 3, 9, 24 and values 0.5, 2.5, 4.5 from df_search2.c).
 * Build: gcc -O1 df_search3.c -I$MKLROOT/include -L$MKLROOT/lib -lmkl_rt -lm -o df_search3 */
#include <stdio.h>
#include "mkl_df.h"
#define NX 6
static const double X[NX] = {0, 1, 2, 3, 4, 5};
static const double Y[NX] = {0, 1, 2, 3, 4, 5};
static int g_mode, g_rc; /* mode: 0 correct 1-based exact, 1 0-based exact, 2 cell 99 exact, 3 cell 0 exact */
static int cb(long long *n, double *site, long long *cell, int *flag, void *user, dfSearchCallBackLibraryParams *lib) {
  for (long long i = 0; i < *n; i++) {
    long long c = (long long)site[i]; if (c > NX - 2) c = NX - 2; if (c < 0) c = 0;
    if (g_mode == 0) c += 1; else if (g_mode == 2) c = 99; else if (g_mode == 3) c = 0;
    cell[i] = c; flag[i] = 1;
  }
  return g_rc;
}
static DFTaskPtr mk(void) {
  DFTaskPtr t = 0; static double sc[2 * (NX - 1)];
  dfdNewTask1D(&t, NX, X, DF_NON_UNIFORM_PARTITION, 1, Y, DF_NO_HINT);
  dfdEditPPSpline1D(t, DF_PP_LINEAR, DF_PP_DEFAULT, DF_NO_BC, NULL, DF_NO_IC, NULL, sc, DF_NO_HINT);
  dfdConstruct1D(t, DF_PP_SPLINE, DF_METHOD_STD);
  return t;
}
int main(void) {
  int modes[] = {0, 1, 2, 3, 0, 0}, rcs[] = {0, 0, 0, 0, -5, 3};
  for (int k = 0; k < 6; k++) {
    g_mode = modes[k]; g_rc = rcs[k];
    printf("mode=%d cb_rc=%d\n", g_mode, g_rc);
    DFTaskPtr t = mk();
    double l[3] = {0.5, 1.5, -1.0}, r[3] = {2.5, 4.5, 7.0}, out[3] = {-77, -77, -77};
    int rc = dfdIntegrateEx1D(t, DF_METHOD_PP, 3, l, DF_NO_HINT, r, DF_NO_HINT, NULL, NULL, out, DF_NO_HINT,
                              NULL, NULL, NULL, NULL, NULL, NULL, cb, NULL);
    printf(" integrate rc=%d out=%g %g %g\n", rc, out[0], out[1], out[2]);
    double site[3] = {0.5, 2.5, 4.5}, v[3] = {-77, -77, -77}; MKL_INT dorder[1] = {1};
    rc = dfdInterpolateEx1D(t, DF_INTERP, DF_METHOD_PP, 3, site, DF_NON_UNIFORM_PARTITION, 1, dorder, NULL, v, DF_NO_HINT, NULL,
                            NULL, NULL, NULL, NULL, NULL, NULL, cb, NULL);
    printf(" interpolate rc=%d v=%g %g %g\n", rc, v[0], v[1], v[2]);
    dfDeleteTask(&t);
  }
  return 0;
}
