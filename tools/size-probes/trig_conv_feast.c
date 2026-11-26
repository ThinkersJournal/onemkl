#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <mkl.h>
#include <mkl_trig_transforms.h>
#include <mkl_solvers_ee.h>
#include <mkl_vsl.h>
#include <mkl_df.h>

/* Allocate `bytes` so the buffer ends exactly at a PROT_NONE guard page: any
 * access past the end faults. Contents zero (or filled by caller). */
static void *guarded(size_t bytes) {
  size_t pg = 4096, total = ((bytes + pg - 1) / pg) * pg + pg;
  char *base = mmap(NULL, total + pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  mprotect(base + total, pg, PROT_NONE);
  return base + total - bytes; /* ends at guard page; may be unaligned to 8 -> pad */
}
static void *guardedD(size_t n) { /* n doubles, 8-aligned end */
  return guarded(n * sizeof(double));
}
static void *guardedI(size_t n) { return guarded(n * sizeof(MKL_INT)); }

/* run fn(arg) in a forked child; return 1 if it exited 0, 0 if it faulted */
typedef int (*fn_t)(long);
static int survives(fn_t fn, long arg) {
  fflush(stdout);
  pid_t p = fork();
  if (p == 0) { _exit(fn(arg) ? 0 : 3); }
  int st; waitpid(p, &st, 0);
  return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}
/* smallest arg in [lo,hi] with survives(), assuming monotone; -1 if none */
static long minimal(fn_t fn, long lo, long hi) {
  for (long a = lo; a <= hi; a++) if (survives(fn, a)) return a;
  return -1;
}

/* ---------- trig: f extent, n=64, type from global ---------- */
static int g_type, g_n;
static int trig_f(long fl) {
  MKL_INT n = g_n, tt = g_type, stat = 0, ipar[128];
  double *dpar = calloc(5 * n + 64, sizeof(double));
  double *f = guardedD(fl);
  for (long i = 0; i < fl; i++) f[i] = 0.0;
  d_init_trig_transform(&n, &tt, ipar, dpar, &stat);
  if (stat) return 0;
  DFTI_DESCRIPTOR_HANDLE h = 0;
  d_commit_trig_transform(f, &h, ipar, dpar, &stat);
  if (stat) return 0;
  d_forward_trig_transform(f, &h, ipar, dpar, &stat);
  d_backward_trig_transform(f, &h, ipar, dpar, &stat);
  free_trig_transform(&h, ipar, &stat);
  return 1;
}
/* ipar: smallest length such that init+commit survive */
static int trig_ipar(long il) {
  MKL_INT n = g_n, tt = g_type, stat = 0;
  MKL_INT *ipar = guardedI(il);
  double *dpar = calloc(5 * n + 64, sizeof(double));
  double *f = calloc(n + 2, sizeof(double));
  d_init_trig_transform(&n, &tt, ipar, dpar, &stat);
  DFTI_DESCRIPTOR_HANDLE h = 0;
  d_commit_trig_transform(f, &h, ipar, dpar, &stat);
  d_forward_trig_transform(f, &h, ipar, dpar, &stat);
  free_trig_transform(&h, ipar, &stat);
  return 1;
}
static int trig_dpar(long dl) {
  MKL_INT n = g_n, tt = g_type, stat = 0, ipar[128];
  double *dpar = guardedD(dl);
  double *f = calloc(n + 2, sizeof(double));
  d_init_trig_transform(&n, &tt, ipar, dpar, &stat);
  DFTI_DESCRIPTOR_HANDLE h = 0;
  d_commit_trig_transform(f, &h, ipar, dpar, &stat);
  d_forward_trig_transform(f, &h, ipar, dpar, &stat);
  free_trig_transform(&h, ipar, &stat);
  return 1;
}

/* ---------- convolution: exec extents ---------- */
static int g_xs, g_ys, g_zs, g_str; /* shapes, stride */
static int g_which;                 /* 0 x, 1 y, 2 z */
static int conv_run(long len) {
  VSLConvTaskPtr t; MKL_INT mode = VSL_CONV_MODE_DIRECT;
  if (vsldConvNewTask1D(&t, mode, g_xs, g_ys, g_zs)) return 0;
  double *x = g_which == 0 ? guardedD(len) : calloc(1 + (g_xs - 1) * g_str, 8);
  double *y = g_which == 1 ? guardedD(len) : calloc(1 + (g_ys - 1) * g_str, 8);
  double *z = g_which == 2 ? guardedD(len) : calloc(1 + (g_zs - 1) * g_str, 8);
  /* zero the guarded one */
  double *g = g_which == 0 ? x : g_which == 1 ? y : z;
  for (long i = 0; i < len; i++) g[i] = 1.0;
  int rc = vsldConvExec1D(t, x, g_str, y, g_str, z, g_str);
  vslConvDeleteTask(&t);
  return rc == 0;
}

/* ---------- feast dense (double, symmetric tridiagonal-ish) ---------- */
#define FN 20
#define FM0 10
static int g_fw; /* which buffer: 0 a,1 e,2 x,3 res */
static int feast_run(long len) {
  char uplo = 'F'; MKL_INT n = FN, lda = FN, fpm[128], loop, m0 = FM0, m, info;
  double epsout, emin = 0.5, emax = 4.5;
  feastinit(fpm);
  double *a = g_fw == 0 ? guardedD(len) : malloc(sizeof(double) * FN * FN);
  double *e = g_fw == 1 ? guardedD(len) : malloc(sizeof(double) * FM0 * 2);
  double *x = g_fw == 2 ? guardedD(len) : malloc(sizeof(double) * FN * FM0 * 2);
  double *res = g_fw == 3 ? guardedD(len) : malloc(sizeof(double) * FM0 * 2);
  /* fill all (full matrix: diagonal 2, off-diagonal -1 -> eigenvalues in (0,4)) */
  double *ag = g_fw == 0 ? a : NULL;
  double *abuf = a;
  for (long i = 0; i < (g_fw == 0 ? len : FN * FN); i++) abuf[i] = 0.0;
  long cnt = g_fw == 0 ? len : FN * FN;
  for (int j = 0; j < FN; j++) for (int i = 0; i < FN; i++) {
    long idx = (long)j * FN + i;
    if (idx >= cnt) continue;
    abuf[idx] = (i == j) ? 2.0 : (abs(i - j) == 1 ? -1.0 : 0.0);
  }
  (void)ag;
  dfeast_syev(&uplo, &n, a, &lda, fpm, &epsout, &loop, &emin, &emax, &m0, e, x, &m, res, &info);
  return 1;
}

int main(void) {
  /* positive control for the harness itself: a call that needs exactly 1+ doubles */
  printf("harness control: copy of 8 doubles needs 8 -> ");
  {
    long found = -1;
    /* trivial fn: reads 8 doubles */
    extern int ctl(long);
    found = minimal(ctl, 1, 12);
    printf("minimal=%ld (expect 8)\n", found);
  }
  for (g_type = 0; g_type < 6; g_type++) {
    g_n = 64;
    long f = minimal(trig_f, 1, 70), ip = minimal(trig_ipar, 1, 140), dp = minimal(trig_dpar, 1, 400);
    printf("trig type %d n=64: min f=%ld  min ipar=%ld  min dpar=%ld (5n/2+2=%d)\n", g_type, f, ip, dp, 5 * 64 / 2 + 2);
  }
  /* conv direct: x=8 y=5 z=12 (full), strides 1 and 2 */
  g_xs = 8; g_ys = 5; g_zs = 12;
  for (g_str = 1; g_str <= 2; g_str++) for (g_which = 0; g_which < 3; g_which++) {
    int shp = g_which == 0 ? g_xs : g_which == 1 ? g_ys : g_zs;
    printf("conv direct stride %d %c: min len=%ld (formula 1+(shape-1)*stride=%d)\n", g_str, "xyz"[g_which],
           minimal(conv_run, 1, 40), 1 + (shp - 1) * g_str);
  }
  for (g_fw = 0; g_fw < 4; g_fw++)
    printf("feast_syev n=%d m0=%d buffer %s: min len=%ld\n", FN, FM0, (const char *[]){"a", "e", "x", "res"}[g_fw],
           minimal(feast_run, 1, 600));
  return 0;
}
int ctl(long len) {
  double *p = guardedD(len);
  volatile double s = 0;
  for (int i = 0; i < 8; i++) s += p[i]; /* reads 8 */
  (void)s;
  return 1;
}
