// SPDX-License-Identifier: MIT OR Apache-2.0
/* Summary Statistics with observation weights (VSL_SS_ED_WEIGHTS), oneMKL 2026.2: what the weighted mean / 2nd central moment are,
 * what ACCUM_WEIGHT holds, and error codes (negative weight, zero weights, NULL). Build:
 * gcc -O1 ss7.c -I$MKLROOT/include -L$MKLROOT/lib -lmkl_rt -lm -o ss7 */
#include <stdio.h>
#include <math.h>
#include "mkl_vsl.h"
#define P 2
#define N 6
static double X[P * N] = {1, 2, 3, 4, 5, 10, 2, 2, 4, 4, 8, 0};
int main(void) {
  double w[N] = {1, 2, 3, 1, 1, 2}, acc[2] = {0, 0};
  double W = 0, W2 = 0; for (int i = 0; i < N; i++) { W += w[i]; W2 += w[i] * w[i]; }
  double mean[P], r2[P], c2[P];
  VSLSSTaskPtr t; MKL_INT p = P, n = N, st = VSL_SS_MATRIX_STORAGE_ROWS;
  int rc = vsldSSNewTask(&t, &p, &n, &st, X, w, NULL); printf("newtask(weights arg) rc=%d\n", rc);
  rc = vsldSSEditTask(t, VSL_SS_ED_ACCUM_WEIGHT, acc); printf("edit accum rc=%d\n", rc);
  vsldSSEditTask(t, VSL_SS_ED_MEAN, mean); vsldSSEditTask(t, VSL_SS_ED_2R_MOM, r2); vsldSSEditTask(t, VSL_SS_ED_2C_MOM, c2);
  rc = vsldSSCompute(t, VSL_SS_MEAN | VSL_SS_2R_MOM | VSL_SS_2C_MOM, VSL_SS_METHOD_FAST); printf("compute rc=%d\n", rc);
  printf("accum = %g %g (W=%g sum w^2=%g)\n", acc[0], acc[1], W, W2);
  for (int v = 0; v < P; v++) {
    double sw = 0, m = 0, s = 0;
    for (int i = 0; i < N; i++) { sw += w[i] * X[v * N + i]; } m = sw / W;
    for (int i = 0; i < N; i++) s += w[i] * (X[v * N + i] - m) * (X[v * N + i] - m);
    printf("var %d: mean=%.10g (analytic %.10g) 2c=%.10g | s/W=%.10g s/(W-W2/W)=%.10g s/(W-1)=%.10g | 2r=%.10g\n", v, mean[v], m, c2[v], s / W, s / (W - W2 / W), s / (W - 1), r2[v]);
  }
  vslSSDeleteTask(&t);
  /* error controls: edit weights after creation with bad values */
  double bad[N] = {1, -2, 3, 1, 1, 2}, zero[N] = {0, 0, 0, 0, 0, 0};
  double *tests[2] = {bad, zero}; const char *nm[2] = {"negative weight", "all-zero weights"};
  for (int k = 0; k < 2; k++) {
    VSLSSTaskPtr u; vsldSSNewTask(&u, &p, &n, &st, X, tests[k], NULL);
    vsldSSEditTask(u, VSL_SS_ED_MEAN, mean);
    rc = vsldSSCompute(u, VSL_SS_MEAN, VSL_SS_METHOD_FAST);
    printf("%s: compute rc=%d mean=%g %g\n", nm[k], rc, mean[0], mean[1]); vslSSDeleteTask(&u);
  }
  VSLSSTaskPtr u; vsldSSNewTask(&u, &p, &n, &st, X, NULL, NULL);
  rc = vsldSSEditTask(u, VSL_SS_ED_WEIGHTS, w); printf("edit WEIGHTS after new rc=%d\n", rc);
  vsldSSEditTask(u, VSL_SS_ED_MEAN, mean); rc = vsldSSCompute(u, VSL_SS_MEAN, VSL_SS_METHOD_FAST); printf("weighted via edit: rc=%d mean=%.10g %.10g\n", rc, mean[0], mean[1]);
  vslSSDeleteTask(&u);
  return 0;
}
