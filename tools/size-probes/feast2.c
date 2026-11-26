#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <mkl.h>
#include <mkl_solvers_ee.h>

/* Same guard-page method as trig_conv_feast.c: the buffer under test ends at a PROT_NONE page,
 * the call runs in a forked child, the smallest surviving length is reported. */
static void *guarded(size_t bytes) {
  size_t pg = 4096, total = ((bytes + pg - 1) / pg) * pg + pg;
  char *base = mmap(NULL, total + pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  mprotect(base + total, pg, PROT_NONE);
  return base + total - bytes;
}
static void *gD(size_t n) { return guarded(n * sizeof(double)); }
static void *gI(size_t n) { return guarded(n * sizeof(MKL_INT)); }
typedef int (*fn_t)(long);
static int survives(fn_t fn, long arg) {
  fflush(stdout);
  pid_t p = fork();
  if (p == 0) _exit(fn(arg) ? 0 : 3);
  int st; waitpid(p, &st, 0);
  return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}
static long minimal(fn_t fn, long lo, long hi) {
  for (long a = lo; a <= hi; a++) if (survives(fn, a)) return a;
  return -1;
}

#define N 20
#define M0 10
static int g_w;
static char g_uplo;

/* csr, symmetric tridiagonal 2/-1. which: 0 sa, 1 isa, 2 jsa, 3 e, 4 x, 5 res */
static int csr_run(long len) {
  char uplo = g_uplo; MKL_INT n = N, fpm[128], loop, m0 = M0, m, info;
  double epsout, emin = 0.5, emax = 4.5;
  feastinit(fpm);
  long nnz = 0; for (int i = 0; i < N; i++) nnz += (uplo == 'F') ? ((i > 0) + 1 + (i < N - 1)) : (1 + (i < N - 1));
  double *sa = g_w == 0 ? gD(len) : malloc(8 * (nnz + 8));
  MKL_INT *isa = g_w == 1 ? gI(len) : malloc(sizeof(MKL_INT) * (N + 2));
  MKL_INT *jsa = g_w == 2 ? gI(len) : malloc(sizeof(MKL_INT) * (nnz + 8));
  double *e = g_w == 3 ? gD(len) : malloc(8 * M0 * 2);
  double *x = g_w == 4 ? gD(len) : malloc(8 * N * M0 * 2);
  double *res = g_w == 5 ? gD(len) : malloc(8 * M0 * 2);
  /* build in malloc'd scratch, then copy the first `len` entries into the guarded buffer */
  double *sa0 = malloc(8 * (nnz + 8)); MKL_INT *isa0 = malloc(sizeof(MKL_INT) * (N + 2)), *jsa0 = malloc(sizeof(MKL_INT) * (nnz + 8));
  long k = 0;
  for (int i = 0; i < N; i++) {
    isa0[i] = k + 1;
    if (uplo == 'F' && i > 0) { sa0[k] = -1; jsa0[k++] = i; }
    sa0[k] = 2; jsa0[k++] = i + 1;
    if (i < N - 1) { sa0[k] = -1; jsa0[k++] = i + 2; }
  }
  isa0[N] = k + 1;
  if (g_w == 0) memcpy(sa, sa0, 8 * (len < k ? len : k)); else memcpy(sa, sa0, 8 * k);
  if (g_w == 1) memcpy(isa, isa0, sizeof(MKL_INT) * (len < N + 1 ? len : N + 1)); else memcpy(isa, isa0, sizeof(MKL_INT) * (N + 1));
  if (g_w == 2) memcpy(jsa, jsa0, sizeof(MKL_INT) * (len < k ? len : k)); else memcpy(jsa, jsa0, sizeof(MKL_INT) * k);
  for (long i = 0; i < (g_w == 4 ? len : 0); i++) x[i] = 0;
  dfeast_scsrev(&uplo, &n, sa, isa, jsa, fpm, &epsout, &loop, &emin, &emax, &m0, e, x, &m, res, &info);
  return 1;
}

/* banded sbev, kla=1. which: 0 a, 1 e, 2 x, 3 res */
static int g_lda;
static int band_run(long len) {
  char uplo = g_uplo; MKL_INT n = N, kla = 1, lda = g_lda, fpm[128], loop, m0 = M0, m, info;
  double epsout, emin = 0.5, emax = 4.5;
  feastinit(fpm);
  double *a = g_w == 0 ? gD(len) : malloc(8 * lda * N);
  double *e = g_w == 1 ? gD(len) : malloc(8 * M0 * 2);
  double *x = g_w == 2 ? gD(len) : malloc(8 * N * M0 * 2);
  double *res = g_w == 3 ? gD(len) : malloc(8 * M0 * 2);
  double *a0 = calloc(lda * N, 8);
  for (int j = 0; j < N; j++) {
    if (uplo == 'U') { a0[(long)j * lda + 1] = 2; if (j > 0) a0[(long)j * lda] = -1; }
    else if (uplo == 'L') { a0[(long)j * lda] = 2; if (j < N - 1) a0[(long)j * lda + 1] = -1; }
    else { a0[(long)j * lda + 1] = 2; if (j > 0) a0[(long)j * lda] = -1; if (j < N - 1) a0[(long)j * lda + 2] = -1; }
  }
  long cnt = lda * N;
  memcpy(a, a0, 8 * (g_w == 0 && len < cnt ? len : cnt));
  dfeast_sbev(&uplo, &n, &kla, a, &lda, fpm, &epsout, &loop, &emin, &emax, &m0, e, x, &m, res, &info);
  return 1;
}

/* generalised dense sygv, B = I (SPD), uplo F. which: 0 a, 1 b */
static int gen_run(long len) {
  char uplo = 'F'; MKL_INT n = N, lda = N, ldb = N, fpm[128], loop, m0 = M0, m, info;
  double epsout, emin = 0.5, emax = 4.5;
  feastinit(fpm);
  double *a = g_w == 0 ? gD(len) : malloc(8 * N * N);
  double *b = g_w == 1 ? gD(len) : malloc(8 * N * N);
  double *e = malloc(8 * M0 * 2), *x = malloc(8 * N * M0 * 2), *res = malloc(8 * M0 * 2);
  double *a0 = calloc(N * N, 8), *b0 = calloc(N * N, 8);
  for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) {
    a0[(long)j * N + i] = i == j ? 2 : (abs(i - j) == 1 ? -1 : 0);
    b0[(long)j * N + i] = i == j ? 1 : 0;
  }
  memcpy(a, a0, 8 * (g_w == 0 && len < N * N ? len : N * N));
  memcpy(b, b0, 8 * (g_w == 1 && len < N * N ? len : N * N));
  dfeast_sygv(&uplo, &n, a, &lda, b, &ldb, fpm, &epsout, &loop, &emin, &emax, &m0, e, x, &m, res, &info);
  return 1;
}

int main(void) {
  const char *cn[] = {"sa", "isa", "jsa", "e", "x", "res"};
  for (int u = 0; u < 2; u++) {
    g_uplo = "FU"[u];
    for (g_w = 0; g_w < 6; g_w++)
      printf("feast_scsrev n=%d m0=%d uplo=%c %s: min len=%ld\n", N, M0, g_uplo, cn[g_w], minimal(csr_run, 1, 800));
  }
  const char *bn[] = {"a", "e", "x", "res"};
  for (int u = 0; u < 3; u++) {
    g_uplo = "ULF"[u]; g_lda = g_uplo == 'F' ? 3 : 2;
    for (g_w = 0; g_w < 4; g_w++)
      printf("feast_sbev n=%d kla=1 lda=%d m0=%d uplo=%c %s: min len=%ld\n", N, g_lda, M0, g_uplo, bn[g_w], minimal(band_run, 1, 800));
  }
  for (g_w = 0; g_w < 2; g_w++)
    printf("feast_sygv n=%d m0=%d uplo=F %s: min len=%ld (n*n=%d)\n", N, M0, g_w ? "b" : "a", minimal(gen_run, 1, 800), N * N);
  return 0;
}
