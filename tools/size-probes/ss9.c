// SPDX-License-Identifier: MIT OR Apache-2.0
/* Summary Statistics partial covariance / correlation (vsldSSEditPartialCovCor, VSL_SS_PARTIAL_COV | VSL_SS_PARTIAL_COR), oneMKL 2026.2:
 * output shape, the formula (checked against S_YY - S_YZ S_ZZ^-1 S_ZY computed here), p_idx_array coding, and error codes.
 * Build: gcc -O1 ss9.c -I$MKLROOT/include -L$MKLROOT/lib -lmkl_rt -lm -o ss9 */
#include <stdio.h>
#include <math.h>
#include "mkl_vsl.h"
#define P 3
#define N 10
static double X[P * N] = {
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  2, 1, 4, 3, 7, 5, 8, 9, 12, 11,
  5, 3, 6, 8, 2, 9, 4, 12, 7, 10};

static void run(const char *name, MKL_INT *idx) {
  double mean[P], cov[P * P], cor[P * P], pcov[P * P], pcor[P * P];
  for (int i = 0; i < P * P; i++) pcov[i] = pcor[i] = -9;
  VSLSSTaskPtr t; MKL_INT p = P, n = N, st = VSL_SS_MATRIX_STORAGE_ROWS, full = VSL_SS_MATRIX_STORAGE_FULL;
  vsldSSNewTask(&t, &p, &n, &st, X, NULL, NULL);
  int e0 = vsldSSEditTask(t, VSL_SS_ED_MEAN, mean);
  int e1 = vsldSSEditPartialCovCor(t, idx, cov, &full, cor, &full, pcov, &full, pcor, &full);
  int c = vsldSSCompute(t, VSL_SS_MEAN | VSL_SS_COV | VSL_SS_COR | VSL_SS_PARTIAL_COV | VSL_SS_PARTIAL_COR, VSL_SS_METHOD_FAST);
  printf("%s: edit rc=%d,%d compute rc=%d\n", name, e0, e1, c);
  printf("  cov  %g %g %g | %g %g %g | %g %g %g\n", cov[0], cov[1], cov[2], cov[3], cov[4], cov[5], cov[6], cov[7], cov[8]);
  printf("  pcov %g %g %g | %g %g %g | %g %g %g\n", pcov[0], pcov[1], pcov[2], pcov[3], pcov[4], pcov[5], pcov[6], pcov[7], pcov[8]);
  printf("  pcor %g %g %g | %g %g %g | %g %g %g\n", pcor[0], pcor[1], pcor[2], pcor[3], pcor[4], pcor[5], pcor[6], pcor[7], pcor[8]);
  vslSSDeleteTask(&t);
}
int main(void) {
  /* Z = variable 0, Y = variables 1, 2. Analytic: S_YY - S_YZ S_ZZ^-1 S_ZY with the n - 1 covariance. */
  double m[P] = {0}, S[P][P];
  for (int v = 0; v < P; v++) for (int i = 0; i < N; i++) m[v] += X[v * N + i] / N;
  for (int a = 0; a < P; a++) for (int b = 0; b < P; b++) {
    double s = 0; for (int i = 0; i < N; i++) s += (X[a * N + i] - m[a]) * (X[b * N + i] - m[b]);
    S[a][b] = s / (N - 1);
  }
  double pc11 = S[1][1] - S[1][0] * S[0][1] / S[0][0], pc12 = S[1][2] - S[1][0] * S[0][2] / S[0][0], pc22 = S[2][2] - S[2][0] * S[0][2] / S[0][0];
  printf("analytic partial cov of (1,2) given 0 (n-1): %g %g %g ; pcor12 = %g\n", pc11, pc12, pc22, pc12 / sqrt(pc11 * pc22));
  MKL_INT a[P] = {-1, 1, 1};
  run("idx {-1,1,1}", a);
  MKL_INT b[P] = {1, -1, 1};
  run("idx {1,-1,1}", b);
  MKL_INT bad[P] = {0, 1, 1};
  run("idx {0,1,1} (bad code)", bad);
  MKL_INT allz[P] = {-1, -1, -1};
  run("idx all -1", allz);
  run("idx NULL", NULL);
  return 0;
}
