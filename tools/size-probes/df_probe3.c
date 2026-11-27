/* Data Fitting probes for the call-back, Hyman and quadratic questions in
 * docs/DATA-FITTING-FACTS.md. Build (oneMKL 2026.2, sequential):
 *   gcc -O1 df_probe3.c -I$MKLROOT/include -L$MKLROOT/lib -lmkl_intel_lp64 \
 *       -lmkl_sequential -lmkl_core -lm -o df_probe3
 * Prints one line per case: what was tried and the status / numbers. */
#include <stdio.h>
#include <string.h>
#include "mkl_df.h"

#define NX 8
static const double X[NX] = {0, 1, 2, 3, 4, 5, 6, 7};
static const double Y[NX] = {0, 0, 0.1, 1.0, 1.0, 1.1, 2.0, 2.0}; /* monotone, plateaus */

static DFTaskPtr mk(void) {
    DFTaskPtr t = 0;
    int s = dfdNewTask1D(&t, NX, X, DF_NON_UNIFORM_PARTITION, 1, Y, DF_NO_HINT);
    if (s != DF_STATUS_OK) { printf("newtask %d\n", s); return 0; }
    return t;
}

static void spline_ic(const char *name, int order, int type, int bctype, const double *bc, int ictype, const double *ic) {
    DFTaskPtr t = mk();
    double sc[1 * 4 * (NX - 1)];
    memset(sc, 0, sizeof sc);
    int s = dfdEditPPSpline1D(t, order, type, bctype, (double *)bc, ictype, (double *)ic, sc, DF_NO_HINT);
    int c = DF_STATUS_OK;
    double worst = 0;
    int nonmono = 0;
    if (s == DF_STATUS_OK) {
        c = dfdConstruct1D(t, DF_PP_SPLINE, DF_METHOD_STD);
        if (c == DF_STATUS_OK) {
            double site[64], r[64];
            int n = 0;
            for (double v = 0; v <= 7.0001 && n < 64; v += 0.1) site[n++] = v;
            MKL_INT dorder[1] = {1}; /* value only (dorder[0] selects the function value) */
            int ev = dfdInterpolate1D(t, DF_INTERP, DF_METHOD_PP, n, site, DF_NON_UNIFORM_PARTITION,
                                      1, dorder, 0, r, DF_NO_HINT, 0);
            if (ev == DF_STATUS_OK) {
                for (int i = 1; i < n; i++) {
                    double drop = r[i - 1] - r[i]; /* > 0 means the curve went down */
                    if (drop > worst) worst = drop;
                }
                nonmono = worst > 1e-12;
            } else c = ev;
        }
    }
    printf("%-34s edit=%d construct/eval=%d worst_drop=%.6f nonmonotone=%d\n", name, s, c, worst, nonmono);
    dfDeleteTask(&t);
}

/* call-backs: record the call and write a recognisable result */
static void spline(const char *n, int o, int t, int b, const double *bc) { spline_ic(n, o, t, b, bc, DF_NO_IC, 0); }

static int calls_le, calls_re, calls_i;
static int ret_code = 0;
static int cb(MKL_INT64 *n, MKL_INT64 *lc, double *ll, MKL_INT64 *rc, double *rl, double *r, void *u, dfIntegrCallBackLibraryParams *lp) {
    (void)lc; (void)rc; (void)lp;
    int *which = (int *)u;
    (*which)++;
    for (MKL_INT64 i = 0; i < *n; i++) { r[i] = 100.0 + (rl[i] - ll[i]); }
    printf("  cb(#%d) n=%lld first llim=%g rlim=%g lcell=%lld rcell=%lld\n", *which, (long long)*n, ll[0], rl[0], (long long)lc[0], (long long)rc[0]);
    return ret_code;
}

static void integ(const char *name, double lo, double hi, int ret) {
    DFTaskPtr t = mk();
    double sc[1 * 2 * (NX - 1)];
    int s = dfdEditPPSpline1D(t, DF_PP_LINEAR, DF_PP_DEFAULT, DF_NO_BC, 0, DF_NO_IC, 0, sc, DF_NO_HINT);
    int c = dfdConstruct1D(t, DF_PP_SPLINE, DF_METHOD_STD);
    double r[1] = {-1};
    calls_le = calls_re = calls_i = 0;
    ret_code = ret;
    int st = dfdIntegrateEx1D(t, DF_METHOD_PP, 1, &lo, DF_NO_HINT, &hi, DF_NO_HINT, 0, 0, r, DF_NO_HINT,
                              cb, &calls_le, cb, &calls_re, cb, &calls_i, 0, 0);
    printf("%-34s edit=%d construct=%d integrate_status=%d r=%g calls le/re/i=%d/%d/%d\n", name, s, c, st, r[0], calls_le, calls_re, calls_i);
    dfDeleteTask(&t);
}

int main(void) {
    double nobc = 0, q = 0.05, ends[2] = {0, 0};
    (void)nobc;
    spline("hyman NO_BC", DF_PP_CUBIC, DF_PP_HYMAN, DF_NO_BC, 0);
    spline("hyman NOT_A_KNOT", DF_PP_CUBIC, DF_PP_HYMAN, DF_BC_NOT_A_KNOT, 0);
    spline("hyman FREE_END", DF_PP_CUBIC, DF_PP_HYMAN, DF_BC_FREE_END, 0);
    spline("hyman 1ST_L|1ST_R = 0,0", DF_PP_CUBIC, DF_PP_HYMAN, DF_BC_1ST_LEFT_DER | DF_BC_1ST_RIGHT_DER, ends);
    spline("hyman 2ND_L|2ND_R = 0,0", DF_PP_CUBIC, DF_PP_HYMAN, DF_BC_2ND_LEFT_DER | DF_BC_2ND_RIGHT_DER, ends);
    spline("natural FREE_END (control)", DF_PP_CUBIC, DF_PP_NATURAL, DF_BC_FREE_END, 0);
    spline("default quadratic Q_VAL", DF_PP_QUADRATIC, DF_PP_DEFAULT, DF_BC_Q_VAL, &q);
    spline("default quadratic NO_BC", DF_PP_QUADRATIC, DF_PP_DEFAULT, DF_NO_BC, 0);
    spline("subbotin FREE_END", DF_PP_QUADRATIC, DF_PP_SUBBOTIN, DF_BC_FREE_END, 0);
    spline("subbotin 1ST_L|1ST_R", DF_PP_QUADRATIC, DF_PP_SUBBOTIN, DF_BC_1ST_LEFT_DER | DF_BC_1ST_RIGHT_DER, ends);
    {
        double kn[NX + 1];
        kn[0] = X[0]; kn[NX] = X[NX - 1];
        for (int i = 1; i < NX; i++) kn[i] = 0.5 * (X[i - 1] + X[i]) + 0.1;
        spline_ic("subbotin NO_BC", DF_PP_QUADRATIC, DF_PP_SUBBOTIN, DF_NO_BC, 0, DF_NO_IC, 0);
        spline_ic("subbotin NO_BC + Q_KNOT(n+1)", DF_PP_QUADRATIC, DF_PP_SUBBOTIN, DF_NO_BC, 0, DF_IC_Q_KNOT, kn);
        spline_ic("subbotin FREE_END + Q_KNOT", DF_PP_QUADRATIC, DF_PP_SUBBOTIN, DF_BC_FREE_END, 0, DF_IC_Q_KNOT, kn);
        spline_ic("subbotin NOT_A_KNOT", DF_PP_QUADRATIC, DF_PP_SUBBOTIN, DF_BC_NOT_A_KNOT, 0, DF_NO_IC, 0);
    }
    puts("-- IntegrateEx1D, linear spline on [0,7]; each cb returns 100 + width");
    integ("inside [1,3], cb ret 0", 1, 3, 0);
    integ("left+inside+right [-2,9], ret 0", -2, 9, 0);
    integ("outside only [8,9], ret 0", 8, 9, 0);
    integ("inside [1,3], cb ret +5", 1, 3, 5);
    integ("inside [1,3], cb ret -3", 1, 3, -3);
    integ("reversed [3,1], ret 0", 3, 1, 0);
    return 0;
}
