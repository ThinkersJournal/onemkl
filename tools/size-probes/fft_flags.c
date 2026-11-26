#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include <mkl_dfti.h>
/* Which DFTI_WORKSPACE / DFTI_ORDERING settings oneMKL accepts, and do they change the result? */
static int run(int nd, long *dims, int ws, int ord, int inplace, double *out_err) {
  DFTI_DESCRIPTOR_HANDLE h; long tot = 1; for (int i = 0; i < nd; i++) tot *= dims[i];
  long st = DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, nd, nd == 1 ? (void *)dims[0] : (void *)dims);
  (void)st;
  return 0;
}
int main(void) {
  struct { const char *n; MKL_LONG v; } wsv[] = {{"default", -1}, {"ALLOW", DFTI_ALLOW}, {"AVOID", DFTI_AVOID}};
  struct { const char *n; MKL_LONG v; } ordv[] = {{"default", -1}, {"ORDERED", DFTI_ORDERED}, {"BACKWARD_SCRAMBLED", DFTI_BACKWARD_SCRAMBLED}};
  for (int nd = 1; nd <= 2; nd++) for (int ip = 0; ip < 2; ip++)
    for (int w = 0; w < 3; w++) for (int o = 0; o < 3; o++) {
      MKL_LONG dims1 = 64, dims2[2] = {8, 16};
      DFTI_DESCRIPTOR_HANDLE h;
      long tot = nd == 1 ? 64 : 128;
      MKL_LONG s0 = nd == 1 ? DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, dims1)
                            : DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 2, dims2);
      MKL_LONG s1 = DftiSetValue(h, DFTI_PLACEMENT, ip ? DFTI_INPLACE : DFTI_NOT_INPLACE);
      MKL_LONG s2 = wsv[w].v < 0 ? 0 : DftiSetValue(h, DFTI_WORKSPACE, wsv[w].v);
      MKL_LONG s3 = ordv[o].v < 0 ? 0 : DftiSetValue(h, DFTI_ORDERING, ordv[o].v);
      MKL_LONG s4 = DftiCommitDescriptor(h);
      double err = -1;
      if (!s4) {
        double *x = malloc(16 * tot), *y = malloc(16 * tot), *z = malloc(16 * tot);
        for (long i = 0; i < 2 * tot; i++) x[i] = sin(0.37 * i) + 0.1 * i;
        for (long i = 0; i < 2 * tot; i++) y[i] = x[i];
        MKL_LONG f = ip ? DftiComputeForward(h, y) : DftiComputeForward(h, y, z);
        double *res = ip ? y : z;
        /* reference: a separate default descriptor */
        DFTI_DESCRIPTOR_HANDLE r; double *rr = malloc(16 * tot);
        if (nd == 1) DftiCreateDescriptor(&r, DFTI_DOUBLE, DFTI_COMPLEX, 1, dims1); else DftiCreateDescriptor(&r, DFTI_DOUBLE, DFTI_COMPLEX, 2, dims2);
        DftiSetValue(r, DFTI_PLACEMENT, DFTI_NOT_INPLACE); DftiCommitDescriptor(r);
        DftiComputeForward(r, x, rr);
        err = 0; for (long i = 0; i < 2 * tot; i++) err = fmax(err, fabs(res[i] - rr[i]));
        printf("nd=%d %s ws=%s ord=%s: set=%ld,%ld,%ld commit=%ld fwd=%ld maxdiff-vs-default=%g\n", nd, ip ? "inplace" : "outofplace", wsv[w].n, ordv[o].n, (long)s1, (long)s2, (long)s3, (long)s4, (long)f, err);
      } else {
        printf("nd=%d %s ws=%s ord=%s: set=%ld,%ld,%ld commit=%ld (%s)\n", nd, ip ? "inplace" : "outofplace", wsv[w].n, ordv[o].n, (long)s1, (long)s2, (long)s3, (long)s4, DftiErrorMessage(s4));
      }
      DftiFreeDescriptor(&h);
    }
  return 0;
}
