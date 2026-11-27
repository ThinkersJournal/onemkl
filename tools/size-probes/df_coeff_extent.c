/* How many spline coefficients oneMKL 2026.2 writes for each spline type
 * (sentinel-filled buffer, count the entries that changed). Result worth
 * keeping: every type writes order * (nx - 1) except the Subbotin quadratic,
 * which writes order * nx (24 for nx = 8); the Intel reference says
 * order * (nx - 1). Build like df_probe3.c. */
#include <stdio.h>
#include <string.h>
#include "mkl_df.h"
#define NX 8
static double X[NX]={0,1,2,3,4,5,6,7};
static void run(const char*name,int order,int stype,int bctype,double*bc,int ictype,double*ic){
  double Y[NX]; for(int i=0;i<NX;i++)Y[i]=X[i]*X[i]*0.1+i;
  DFTaskPtr t=0; dfdNewTask1D(&t,NX,X,DF_NON_UNIFORM_PARTITION,1,Y,DF_NO_HINT);
  static double sc[400]; for(int i=0;i<400;i++) sc[i]=-12345.0;
  int s=dfdEditPPSpline1D(t,order,stype,bctype,bc,ictype,ic,sc,DF_NO_HINT);
  int c=dfdConstruct1D(t,DF_PP_SPLINE,DF_METHOD_STD);
  int last=-1; for(int i=0;i<400;i++) if(sc[i]!=-12345.0) last=i;
  printf("%-28s edit=%d construct=%d coefficients written=%d  (order*(nx-1)=%d)\n",name,s,c,last+1,order*(NX-1));
  dfDeleteTask(&t);
}
int main(void){
  double q=0.1, ends[2]={0,0}, kn[NX+1], ic1[NX-2]={0,0,0,0,0,0};
  kn[0]=X[0]; kn[NX]=X[NX-1]; for(int i=1;i<NX;i++)kn[i]=0.5*(X[i-1]+X[i]);
  run("linear",DF_PP_LINEAR,DF_PP_DEFAULT,DF_NO_BC,0,DF_NO_IC,0);
  run("natural cubic",DF_PP_CUBIC,DF_PP_NATURAL,DF_BC_FREE_END,0,DF_NO_IC,0);
  run("bessel",DF_PP_CUBIC,DF_PP_BESSEL,DF_BC_NOT_A_KNOT,0,DF_NO_IC,0);
  run("akima",DF_PP_CUBIC,DF_PP_AKIMA,DF_BC_NOT_A_KNOT,0,DF_NO_IC,0);
  run("hermite",DF_PP_CUBIC,DF_PP_HERMITE,DF_BC_1ST_LEFT_DER|DF_BC_1ST_RIGHT_DER,ends,DF_IC_1ST_DER,ic1);
  run("hyman free-end",DF_PP_CUBIC,DF_PP_HYMAN,DF_BC_FREE_END,0,DF_NO_IC,0);
  run("default quadratic Q_VAL",DF_PP_QUADRATIC,DF_PP_DEFAULT,DF_BC_Q_VAL,&q,DF_NO_IC,0);
  run("subbotin free-end+Q_KNOT",DF_PP_QUADRATIC,DF_PP_SUBBOTIN,DF_BC_FREE_END,0,DF_IC_Q_KNOT,kn);
  return 0;}
