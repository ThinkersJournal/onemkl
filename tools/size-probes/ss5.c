// SPDX-License-Identifier: MIT OR Apache-2.0
/* Summary Statistics: TBS parameter counts and the n > 2p rule; BACON stopping-criterion slot (oneMKL 2026.0 C reference layouts). */
#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <mkl_vsl.h>
static int tbs(int p, int n, int np, double *par) {
  double *x = malloc(8 * p * n), m[8], c[64];
  srand(2);
  for (int i = 0; i < p * n; i++) x[i] = (double)rand() / RAND_MAX;
  MKL_INT P = p, N = n, st = VSL_SS_MATRIX_STORAGE_ROWS, cs = VSL_SS_MATRIX_STORAGE_FULL, k = np;
  VSLSSTaskPtr t;
  vsldSSNewTask(&t, &P, &N, &st, x, NULL, NULL);
  int e = vsldSSEditRobustCovariance(t, &cs, &k, np ? par : NULL, m, c);
  int rc = vsldSSCompute(t, VSL_SS_ROBUST_COV, VSL_SS_METHOD_TBS);
  vslSSDeleteTask(&t);
  printf("tbs p=%d n=%d nparams=%d edit=%d compute=%d\n", p, n, np, e, rc);
  return rc;
}
static void bacon(double stop) {
  int p = 2, n = 50; double x[100], w[50];
  srand(3);
  for (int i = 0; i < 100; i++) x[i] = (double)rand() / RAND_MAX;
  x[0] = 50; x[n] = 50; /* one gross outlier */
  MKL_INT P = p, N = n, st = VSL_SS_MATRIX_STORAGE_ROWS, k = 3;
  double par[3] = {VSL_SS_METHOD_BACON_MEDIAN_INIT, 0.05, stop};
  VSLSSTaskPtr t;
  vsldSSNewTask(&t, &P, &N, &st, x, NULL, NULL);
  vsldSSEditOutliersDetection(t, &k, par, w);
  int rc = vsldSSCompute(t, VSL_SS_OUTLIERS, VSL_SS_METHOD_BACON);
  int out = 0; for (int i = 0; i < n; i++) out += w[i] == 0;
  printf("bacon stop=%g compute=%d outliers=%d w0=%g\n", stop, rc, out, w[0]);
  vslSSDeleteTask(&t);
}
int main(void) {
  double a[6] = {0.25, 0.001, 0.001, 10, 1, 1};
  for (int np = 0; np <= 6; np++) tbs(3, 40, np, a);
  tbs(3, 7, 4, a); tbs(3, 6, 4, a); tbs(3, 5, 4, a);
  bacon(0.005); bacon(0.05);
  return 0;
}
