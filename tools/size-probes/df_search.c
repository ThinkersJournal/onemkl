// SPDX-License-Identifier: MIT OR Apache-2.0
/* Data Fitting user search call-back (dfdSearchCellsEx1D): is the call-back invoked, which cell numbering it must return,
 * what flag / DF_STATUS_EXACT_RESULT do, and how a negative or positive call-back status shows in the routine's status.
 * Build (oneMKL 2026.2): gcc -O1 df_search.c -I$MKLROOT/include -L$MKLROOT/lib -lmkl_rt -lm -o df_search */
#include <stdio.h>
#include <string.h>
#include "mkl_df.h"
#define NX 6
static const double X[NX] = {0, 1, 2, 3, 4, 5};
static const double Y[NX] = {0, 1, 4, 9, 16, 25};
static int g_calls, g_mode, g_base; /* g_base: value added to the 0-based cell */
static int cb(long long *n, double *site, long long *cell, int *flag, void *user, dfSearchCallBackLibraryParams *lib) {
  g_calls++;
  printf("  cb n=%lld lib=%s user=%s\n", *n, lib ? "set" : "NULL", user ? "set" : "NULL");
  for (long long i = 0; i < *n; i++) {
    long long c = (long long)site[i]; if (c > NX - 2) c = NX - 2; if (c < 0) c = 0; /* 0-based interval index */
    cell[i] = c + g_base;
    if (g_mode == 0) flag[i] = 1;      /* exact */
    else if (g_mode == 1) flag[i] = 0; /* approximate start */
    else if (g_mode == 2) flag[i] = 7; /* nonsense flag */
  }
  if (g_mode == 3) return DF_STATUS_EXACT_RESULT;
  if (g_mode == 4) return -5;
  if (g_mode == 5) return 3;
  return 0;
}
static void run(const char *name, int mode, int base) {
  g_calls = 0; g_mode = mode; g_base = base;
  DFTaskPtr t = 0;
  int s = dfdNewTask1D(&t, NX, X, DF_NON_UNIFORM_PARTITION, 1, Y, DF_NO_HINT);
  double site[4] = {0.5, 1.5, 2.5, 4.5}; MKL_INT cell[4] = {-9, -9, -9, -9};
  int rc = dfdSearchCellsEx1D(t, DF_METHOD_STD, 4, site, DF_NON_UNIFORM_PARTITION, NULL, cell, cb, (void *)&g_calls);
  printf("%s (base %d): new=%d rc=%d calls=%d cells=%lld %lld %lld %lld\n", name, base, s, rc, g_calls, (long long)cell[0], (long long)cell[1], (long long)cell[2], (long long)cell[3]);
  dfDeleteTask(&t);
}
int main(void) {
  { DFTaskPtr t = 0; double site[4] = {0.5, 1.5, 2.5, 4.5}; MKL_INT cell[4];
    dfdNewTask1D(&t, NX, X, DF_NON_UNIFORM_PARTITION, 1, Y, DF_NO_HINT);
    int rc = dfdSearchCells1D(t, DF_METHOD_STD, 4, site, DF_NON_UNIFORM_PARTITION, NULL, cell);
    printf("builtin search: rc=%d cells=%lld %lld %lld %lld\n", rc, (long long)cell[0], (long long)cell[1], (long long)cell[2], (long long)cell[3]);
    dfDeleteTask(&t); }
  run("exact flags 0-based", 0, 0);
  run("exact flags 1-based", 0, 1);
  run("flag=0 approx 0-based", 1, 0);
  run("flag=0 approx 1-based", 1, 1);
  run("flag=7", 2, 1);
  run("EXACT_RESULT return", 3, 1);
  run("negative return -5", 4, 1);
  run("positive return 3", 5, 1);
  return 0;
}
