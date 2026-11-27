// SPDX-License-Identifier: MIT OR Apache-2.0
/* Summary Statistics stream quantiles: which compute method and parameter counts oneMKL 2026.2 accepts, and accuracy against exact quantiles. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include <mkl_vsl.h>
static void run(const char *name, int method, int np, double eps) {
  int p = 1, n = 1000; double x[1000], q[3] = {0.25, 0.5, 0.9}, out[3] = {0}, par[1] = {eps};
  for (int i = 0; i < n; i++) x[i] = (double)((i * 7919) % n) / n; /* permutation of 0..n-1, exact quantile ~ order */
  MKL_INT P = p, N = n, st = VSL_SS_MATRIX_STORAGE_ROWS, qn = 3, k = np;
  VSLSSTaskPtr t;
  vsldSSNewTask(&t, &P, &N, &st, x, NULL, NULL);
  int e = vsldSSEditStreamQuantiles(t, &qn, q, out, np ? &k : NULL, np ? par : NULL);
  int rc = vsldSSCompute(t, VSL_SS_STREAM_QUANTS, method);
  printf("%s np=%d eps=%g edit=%d compute=%d out=%g %g %g\n", name, np, eps, e, rc, out[0], out[1], out[2]);
  vslSSDeleteTask(&t);
}
int main(void) {
  run("ZW", VSL_SS_METHOD_SQUANTS_ZW, 1, 0.01);
  run("ZW", VSL_SS_METHOD_SQUANTS_ZW, 1, 0.05);
  run("ZW", VSL_SS_METHOD_SQUANTS_ZW, 0, 0);
  run("ZW", VSL_SS_METHOD_SQUANTS_ZW, 2, 0.01);
  run("ZW", VSL_SS_METHOD_SQUANTS_ZW, 1, 0);
  run("ZW", VSL_SS_METHOD_SQUANTS_ZW, 1, 2.0);
  run("FAST", VSL_SS_METHOD_FAST, 1, 0.01);
  return 0;
}
