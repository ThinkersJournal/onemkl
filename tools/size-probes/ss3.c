#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <mkl.h>
#include <mkl_vsl.h>

/* Guard-page probe (method as in trig_conv_feast.c) for the last three extents that were
 * unverified: Summary Statistics missing values (estimates buffer), correlation exec (x, y, z
 * extents, like conv), and the trig `ipar` array (also printed by trig_conv_feast.c: 15, not 128).
 * Build like df_probe3.c. */
static void *guarded(size_t bytes) {
  size_t pg = 4096, total = ((bytes + pg - 1) / pg) * pg + pg;
  char *base = mmap(NULL, total + pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  mprotect(base + total, pg, PROT_NONE);
  return base + total - bytes;
}
static double *GD(size_t n) { return guarded(n * 8); }
typedef int (*fn_t)(long);
static int survives(fn_t fn, long a) {
  fflush(stdout);
  pid_t p = fork();
  if (p == 0) _exit(fn(a) ? 0 : 3);
  int st; waitpid(p, &st, 0);
  return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}
static long minimal(fn_t fn, long lo, long hi) {
  for (long a = lo; a <= hi; a++) if (survives(fn, a)) return a;
  return -1;
}

/* ---- missing values (MI), as onemkl::SummaryStats::impute_missing calls it ---- */
#define P 3
#define N 40
static int g_verbose;
static int mi_run(long len) {
  double *x = malloc(8 * P * N);
  srand(1);
  for (int i = 0; i < P * N; i++) x[i] = (double)rand() / RAND_MAX;
  for (int j = 0; j < P; j++) x[j * N + 3 + j] = VSL_SS_DNAN; /* three missing values */
  double *est = GD(len);
  for (long i = 0; i < len; i++) est[i] = 0.0;
  MKL_INT p = P, n = N, xst = VSL_SS_MATRIX_STORAGE_ROWS, np = 5, zero = 0, en = P + P * P;
  double par[5] = {VSL_SS_METHOD_MI, 25.0, 1.0e-4, 1.0, 0.0};
  VSLSSTaskPtr t;
  int rc = vsldSSNewTask(&t, &p, &n, &xst, x, NULL, NULL);
  if (rc) return 0;
  rc = vsldSSEditMissingValues(t, &np, par, &zero, NULL, &zero, NULL, &zero, NULL, &en, est);
  if (rc) { if (g_verbose) printf("  edit rc=%d\n", rc); return 0; }
  rc = vsldSSCompute(t, VSL_SS_MISSING_VALS, VSL_SS_METHOD_MI);
  if (g_verbose) printf("  compute rc=%d est[0]=%g\n", rc, est[0]);
  vslSSDeleteTask(&t);
  return rc == 0;
}

/* ---- correlation exec: which 0 x, 1 y, 2 z; direct mode ---- */
static int g_xs = 8, g_ys = 5, g_zs = 12, g_str, g_which;
static int corr_run(long len) {
  VSLCorrTaskPtr t; MKL_INT mode = VSL_CORR_MODE_DIRECT;
  if (vsldCorrNewTask1D(&t, mode, g_xs, g_ys, g_zs)) return 0;
  double *x = g_which == 0 ? GD(len) : calloc(1 + (g_xs - 1) * g_str, 8);
  double *y = g_which == 1 ? GD(len) : calloc(1 + (g_ys - 1) * g_str, 8);
  double *z = g_which == 2 ? GD(len) : calloc(1 + (g_zs - 1) * g_str, 8);
  double *g = g_which == 0 ? x : g_which == 1 ? y : z;
  for (long i = 0; i < len; i++) g[i] = 1.0;
  int rc = vsldCorrExec1D(t, x, g_str, y, g_str, z, g_str);
  vslCorrDeleteTask(&t);
  return rc == 0;
}

int main(void) {
  g_verbose = 1; printf("mi single run (control that the call itself succeeds with a big buffer):\n");
  printf("  big buffer ok=%d\n", mi_run(100)); g_verbose = 0;
  printf("missing values: min estimates buffer = %ld doubles (p + p*p = %d)\n", minimal(mi_run, 1, 40), P + P * P);
  for (g_str = 1; g_str <= 2; g_str++) for (g_which = 0; g_which < 3; g_which++) {
    int shp = g_which == 0 ? g_xs : g_which == 1 ? g_ys : g_zs;
    printf("corr direct stride %d %c: min len=%ld (formula 1+(shape-1)*stride=%d)\n", g_str, "xyz"[g_which],
           minimal(corr_run, 1, 40), 1 + (shp - 1) * g_str);
  }
  return 0;
}
