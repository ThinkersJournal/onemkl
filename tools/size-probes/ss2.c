#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <mkl.h>
#include <mkl_vsl.h>

/* Guard-page probe (method as in trig_conv_feast.c) for the Summary Statistics editors whose
 * extents were unverified: robust covariance (TBS), outlier detection (BACON), missing values (MI),
 * stream quantiles (ZW). */
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
#define P 3
#define N 40
static int g_w, g_verbose;
static void fill(double *x) {
  srand(1);
  for (int i = 0; i < P * N; i++) x[i] = (double)rand() / RAND_MAX + (i % N == 7 ? 20.0 : 0.0);
}
#define CK(call) do { int rc_ = (call); if (rc_) { if (g_verbose) printf("  fail %d at %s\n", rc_, #call); return 0; } } while (0)

/* robust cov: which 0 mean(p), 1 cov(p*p) ; params n_params 4 */
static int rc_run(long len) {
  double *x = malloc(8 * P * N); fill(x);
  double *mean = g_w == 0 ? GD(len) : malloc(8 * P);
  double *cov = g_w == 1 ? GD(len) : malloc(8 * P * P);
  MKL_INT p = P, n = N, xst = VSL_SS_MATRIX_STORAGE_ROWS, cst = VSL_SS_MATRIX_STORAGE_FULL, np = 4;
  double par[4] = {0.1, 0.05, 0.001, 20};
  VSLSSTaskPtr t;
  CK(vsldSSNewTask(&t, &p, &n, &xst, x, NULL, NULL));
  CK(vsldSSEditRobustCovariance(t, &cst, &np, par, mean, cov));
  CK(vsldSSCompute(t, VSL_SS_ROBUST_COV, VSL_SS_METHOD_TBS));
  vslSSDeleteTask(&t);
  return 1;
}
/* outliers: weight(n) ; 3 params */
static int ol_run(long len) {
  double *x = malloc(8 * P * N); fill(x);
  double *wt = GD(len);
  MKL_INT p = P, n = N, xst = VSL_SS_MATRIX_STORAGE_ROWS, np = 3;
  double par[3] = {0.1, 0.01, 0.05};
  VSLSSTaskPtr t;
  CK(vsldSSNewTask(&t, &p, &n, &xst, x, NULL, NULL));
  CK(vsldSSEditOutliersDetection(t, &np, par, wt));
  CK(vsldSSCompute(t, VSL_SS_OUTLIERS, VSL_SS_METHOD_BACON));
  vslSSDeleteTask(&t);
  return 1;
}
/* stream quantiles: quants_out(p*m) ; params 1 (accuracy) */
static int sq_run(long len) {
  double *x = malloc(8 * P * N); fill(x);
  double *qo = GD(len);
  MKL_INT p = P, n = N, xst = VSL_SS_MATRIX_STORAGE_ROWS, m = 3, np = 1;
  double ord[3] = {0.25, 0.5, 0.75}, par[1] = {0.01};
  VSLSSTaskPtr t;
  CK(vsldSSNewTask(&t, &p, &n, &xst, x, NULL, NULL));
  CK(vsldSSEditStreamQuantiles(t, &m, ord, qo, &np, par));
  CK(vsldSSCompute(t, VSL_SS_STREAM_QUANTS, VSL_SS_METHOD_SQUANTS_ZW));
  vslSSDeleteTask(&t);
  return 1;
}

int main(void) {
  g_verbose = 1;
  g_w = 0; printf("robust cov full-size run ok=%d\n", rc_run(P));
  printf("outliers full-size run ok=%d\n", ol_run(N));
  printf("stream quantiles full-size run ok=%d\n", sq_run(P * 3));
  g_verbose = 0;
  for (g_w = 0; g_w < 2; g_w++)
    printf("robust cov (TBS, full storage) %s: min len=%ld (expect %d)\n", g_w ? "cov" : "mean", minimal(rc_run, 1, 100), g_w ? P * P : P);
  printf("outliers (BACON) weights: min len=%ld (expect n=%d)\n", minimal(ol_run, 1, 100), N);
  printf("stream quantiles (ZW) quants_out: min len=%ld (expect p*m=%d)\n", minimal(sq_run, 1, 100), P * 3);
  return 0;
}
