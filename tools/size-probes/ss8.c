// SPDX-License-Identifier: MIT OR Apache-2.0
/* Summary Statistics group / pooled covariance (vsldSSEditPooledCovariance), oneMKL 2026.2: req_grp_indices layout, the group index base, output shapes, pooled-cov formula, error codes.
 * Build: gcc -O1 ss8.c -I$MKLROOT/include -L$MKLROOT/lib -lmkl_rt -lm -o ss8 */
#include <stdio.h>
#include <string.h>
#include "mkl_vsl.h"
#define P 2
#define N 8
/* variable-major: row v holds N observations */
static double X[P * N] = {1, 2, 4, 3, 10, 12, 11, 15,   2, 1, 3, 5, 7, 4, 9, 8};
static MKL_INT G[N] = {0, 0, 0, 0, 1, 1, 1, 1};
static void run(const char *name, MKL_INT *grp, MKL_INT *req) {
  double pm[P], pc[P * P], gm[2 * P], gc[2 * P * P];
  for (int i = 0; i < P; i++) pm[i] = -9; for (int i = 0; i < P * P; i++) pc[i] = -9;
  for (int i = 0; i < 2 * P; i++) gm[i] = -9; for (int i = 0; i < 2 * P * P; i++) gc[i] = -9;
  VSLSSTaskPtr t; MKL_INT p = P, n = N, st = VSL_SS_MATRIX_STORAGE_ROWS, full = VSL_SS_MATRIX_STORAGE_FULL;
  vsldSSNewTask(&t, &p, &n, &st, X, NULL, NULL);
  MKL_INT ps = VSL_SS_MATRIX_STORAGE_FULL, gs = VSL_SS_MATRIX_STORAGE_FULL;
  int s1 = vsldSSEditTask(t, VSL_SS_ED_POOLED_COV_STORAGE, (double *)&ps);
  int s2 = vsldSSEditTask(t, VSL_SS_ED_GROUP_COV_STORAGE, (double *)&gs);
  int rc = vsldSSEditPooledCovariance(t, grp, pm, pc, req, gm, gc);
  printf("  storage edits before pooled editor: %d %d\n", s1, s2);
  int c = vsldSSCompute(t, VSL_SS_POOLED_MEAN | VSL_SS_POOLED_COV | VSL_SS_GROUP_MEAN | VSL_SS_GROUP_COV, VSL_SS_METHOD_FAST);
  printf("%s: edit rc=%d compute rc=%d\n", name, rc, c);
  printf("  pooled mean %g %g | pooled cov %g %g %g %g\n", pm[0], pm[1], pc[0], pc[1], pc[2], pc[3]);
  printf("  group mean %g %g %g %g | group cov", gm[0], gm[1], gm[2], gm[3]); for (int i = 0; i < 8; i++) printf(" %g", gc[i]); printf("\n");
  vslSSDeleteTask(&t);
}
int main(void) {
  /* analytic: group 0 var0 {1,2,4,3} mean 2.5, var1 {2,1,3,5} mean 2.75 ; group 1 var0 {10,12,11,15} mean 12, var1 {7,4,9,8} mean 7 */
  MKL_INT r_a[2] = {1, 1};   /* flags per group (reference) */
  MKL_INT r_b[2] = {1, 0};   /* only group 0 */
  MKL_INT r_c[2] = {0, 1};   /* only group 1 */
  run("req flags {1,1}", G, r_a);
  run("req flags {1,0}", G, r_b);
  run("req flags {0,1}", G, r_c);
  MKL_INT g1[N] = {1, 1, 1, 1, 2, 2, 2, 2};  /* 1-based groups */
  run("1-based groups, req={2,1,2}", g1, (MKL_INT[]){1, 1});
  MKL_INT gbad[N] = {0, 0, 0, 0, 1, 1, 1, 5};
  run("group index 5 of 2", gbad, r_a);
  run("NULL req", G, NULL);
  return 0;
}
