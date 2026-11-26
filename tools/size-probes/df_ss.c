#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <mkl.h>
#include <mkl_vsl.h>
#include <mkl_df.h>

static void *guarded(size_t bytes) {
  size_t pg = 4096, total = ((bytes + pg - 1) / pg) * pg + pg;
  char *base = mmap(NULL, total + pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  mprotect(base + total, pg, PROT_NONE);
  return base + total - bytes;
}
static double *GD(size_t n) { return guarded(n * 8); }
static int *GI(size_t n) { return guarded(n * sizeof(int)); }
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

/* ---------------- data fitting ---------------- */
#define NX 10
#define NY 2
#define NS 7
#define NL 3
static int g_w;
/* g_w: 0 x, 1 y, 2 scoeff, 3 site, 4 r (interp, dorder {1,1}), 5 llim, 6 rint, 7 dorder */
static int df_run(long len) {
  double xs[NX]; for (int i = 0; i < NX; i++) xs[i] = i;
  double *x = g_w == 0 ? GD(len) : malloc(NX * 8);
  double *y = g_w == 1 ? GD(len) : malloc(NX * NY * 8);
  double *sc = g_w == 2 ? GD(len) : malloc(NY * (NX - 1) * 4 * 8);
  double *site = g_w == 3 ? GD(len) : malloc(NS * 8);
  double *r = g_w == 4 ? GD(len) : malloc(NS * NY * 2 * 8);
  double *llim = g_w == 5 ? GD(len) : malloc(NL * 8);
  double *rlim = malloc(NL * 8);
  double *rint = g_w == 6 ? GD(len) : malloc(NL * NY * 8);
  int *dorder = g_w == 7 ? GI(len) : malloc(2 * sizeof(int));
  for (int i = 0; i < NX; i++) { if (g_w == 0 && i >= len) break; x[i] = i; }
  for (long i = 0; i < (g_w == 1 ? len : NX * NY); i++) y[i] = (double)(i % 7);
  for (long i = 0; i < (g_w == 2 ? len : NY * (NX - 1) * 4); i++) sc[i] = 0;
  for (long i = 0; i < (g_w == 3 ? len : NS); i++) site[i] = 0.5 + i;
  for (long i = 0; i < (g_w == 5 ? len : NL); i++) llim[i] = 0.5 + i;
  for (int i = 0; i < NL; i++) rlim[i] = 1.5 + i;
  for (long i = 0; i < (g_w == 7 ? len : 2); i++) dorder[i] = 1;
  DFTaskPtr t;
  if (dfdNewTask1D(&t, NX, x, DF_NON_UNIFORM_PARTITION, NY, y, DF_NO_HINT)) return 0;
  if (dfdEditPPSpline1D(t, DF_PP_CUBIC, DF_PP_NATURAL, DF_BC_FREE_END, NULL, DF_NO_IC, NULL, sc, DF_NO_HINT)) return 0;
  if (dfdConstruct1D(t, DF_PP_SPLINE, DF_METHOD_STD)) return 0;
  if (dfdInterpolate1D(t, DF_INTERP, DF_METHOD_PP, NS, site, DF_NON_UNIFORM_PARTITION, 2, dorder, DF_NO_APRIORI_INFO, r, DF_NO_HINT, NULL)) return 0;
  if (dfdIntegrate1D(t, DF_METHOD_PP, NL, llim, DF_NON_UNIFORM_PARTITION, rlim, DF_NON_UNIFORM_PARTITION, NULL, NULL, rint, DF_NO_HINT)) return 0;
  dfDeleteTask(&t);
  (void)xs;
  return 1;
}

/* ---------------- summary stats ---------------- */
#define SP 3
#define SN 20
#define SQ 4
static int g_s; /* 0 x(p*n), 1 w(n), 2 mean(p), 3 c2m(p), 4 quants(p*q) */
static int ss_run(long len) {
  double *x = g_s == 0 ? GD(len) : malloc(SP * SN * 8);
  double *w = g_s == 1 ? GD(len) : malloc(SN * 8);
  double *mean = g_s == 2 ? GD(len) : malloc(SP * 8);
  double *c2m = g_s == 3 ? GD(len) : malloc(SP * 8);
  double *qo = malloc(SQ * 8); qo[0] = 0.1; qo[1] = 0.5; qo[2] = 0.9; qo[3] = 0.99;
  double *qs = g_s == 4 ? GD(len) : malloc(SP * SQ * 8);
  for (long i = 0; i < (g_s == 0 ? len : SP * SN); i++) x[i] = (double)((i * 7) % 13);
  for (long i = 0; i < (g_s == 1 ? len : SN); i++) w[i] = 1.0;
  MKL_INT p = SP, n = SN, xst = VSL_SS_MATRIX_STORAGE_ROWS, qn = SQ;
  VSLSSTaskPtr t;
  {int rc=(vsldSSNewTask(&t, &p, &n, &xst, x, w, NULL)); if (rc) { if (getenv("DBG")) printf("fail %d: %s\n", rc, "vsldSSNewTask"); return 0; }}
  {int rc=(vsldSSEditTask(t, VSL_SS_ED_MEAN, mean)); if (rc) { if (getenv("DBG")) printf("fail %d: %s\n", rc, "vsldSSEditTask"); return 0; }}
  {int rc=(vsldSSEditTask(t, VSL_SS_ED_2R_MOM, c2m)); if (rc) { if (getenv("DBG")) printf("fail %d: %s\n", rc, "vsldSSEditTask"); return 0; }}
  {int rc=(vsldSSEditQuantiles(t, &qn, qo, qs, NULL, NULL)); if (rc) { if (getenv("DBG")) printf("fail %d: %s\n", rc, "vsldSSEditQuantiles"); return 0; }}
  {int rc=(vsldSSCompute(t, VSL_SS_MEAN | VSL_SS_2R_MOM | VSL_SS_QUANTS, VSL_SS_METHOD_FAST)); if (rc) { if (getenv("DBG")) printf("fail %d: %s\n", rc, "vsldSSCompute"); return 0; }}
  vslSSDeleteTask(&t);
  return 1;
}

int main(void) {
  if (getenv("SSONLY")) { g_s = 0; printf("direct ss_run(200)=%d\n", ss_run(200)); return 0; }
  const char *dn[] = {"x(nx)", "y(nx*ny)", "scoeff(ny*(nx-1)*4)", "site(nsite)", "r(nsite*ny*2 derivs)", "llim(nlim)", "rint(nlim*ny)", "dorder(ndorder=2)"};
  long ex[] = {NX, NX * NY, NY * (NX - 1) * 4, NS, NS * NY * 2, NL, NL * NY, 2};
  for (g_w = 0; g_w < 8; g_w++) printf("df %s: min len=%ld (expected %ld)\n", dn[g_w], minimal(df_run, 1, 200), ex[g_w]);
  const char *sn[] = {"x(p*n)", "w(n)", "mean(p)", "c2m(p)", "quants(p*q)"};
  long se[] = {SP * SN, SN, SP, SP, SP * SQ};
  for (g_s = 0; g_s < 5; g_s++) printf("ss %s: min len=%ld (expected %ld)\n", sn[g_s], minimal(ss_run, 1, 200), se[g_s]);
  return 0;
}
