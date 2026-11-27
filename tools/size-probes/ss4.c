// SPDX-License-Identifier: MIT OR Apache-2.0
/* vsldSSEditMissingValues with the parameter layout from the oneMKL 2026.0 C reference (params = {em_iter, da_iter, eps, m, k}). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mkl.h>
#include <mkl_vsl.h>
#define P 3
#define N 40
static int run(int with_est, int kk) {
  double x[P * N];
  srand(1);
  for (int i = 0; i < P * N; i++) x[i] = (double)rand() / RAND_MAX;
  for (int j = 0; j < P; j++) { unsigned long long b = VSL_SS_DNAN; memcpy(&x[j * N + 3 + j], &b, 8); }
  MKL_INT p = P, n = N, xst = VSL_SS_MATRIX_STORAGE_COLS, np = 5, zero = 0;
  MKL_INT m = 2, k = kk, da = 30;
  double par[5] = {50.0, (double)da, 1e-3, (double)m, (double)k};
  MKL_INT sn = m * k, en = m * da * (P + (P * P + P) / 2);
  double sim[64] = {0}; double *est = calloc(en, 8);
  VSLSSTaskPtr t;
  int rc = vsldSSNewTask(&t, &p, &n, &xst, x, NULL, NULL);
  printf("new rc=%d\n", rc);
  rc = vsldSSEditMissingValues(t, &np, par, &zero, NULL, &zero, NULL, &sn, sim, with_est ? &en : &zero, with_est ? est : NULL);
  printf("edit rc=%d\n", rc);
  rc = vsldSSCompute(t, VSL_SS_MISSING_VALS, VSL_SS_METHOD_MI);
  printf("compute(with_est=%d) rc=%d sim=%g %g %g %g %g %g est0=%g\n", with_est, rc, sim[0], sim[1], sim[2], sim[3], sim[4], sim[5], est[0]);
  vslSSDeleteTask(&t);
  return rc;
}
int main(void) { run(0,3); run(1,3); return 0; }
