# Usage (from the repo root): python tools/w1/gen_check.py <dir with the copied oneMKL headers> <flagged.txt>; the name map (Rust method -> C functions) is MAP below. MIRROR = same argument count as every listed C function that the headers declare.
import re,sys,glob
ROOT='.'
SPD=sys.argv[1]
text=''
for h in ('mkl_spblas.h','mkl_vsl_functions.h','mkl_vml_functions.h','mkl_solvers_ee.h','mkl_df_functions.h','mkl_blas.h','mkl_cblas.h','mkl_lapacke.h','mkl_blas_compact.h','mkl_sparse_qr.h','mkl_poisson.h','mkl_trans.h'):
    try: text+=open(f'{SPD}/{h}',encoding='utf-8',errors='replace').read()+'\n'
    except FileNotFoundError: pass
def split_args(a):
    a=a.strip()
    if a in('','void'): return 0
    d=0;n=1
    for ch in a:
        if ch in '([': d+=1
        elif ch in ')]': d-=1
        elif ch==',' and d==0: n+=1
    return n
protos={}
for m in re.finditer(r'\b(?:_Mkl_Api|_Mkl_Vml_Api|_mkl_api)\s*\(\s*[\w\s\*]+?,\s*(\w+)\s*,\s*\((.*?)\)\s*(?:NOTHROW)?\s*\)\s*;',text,re.S):
    protos.setdefault(m.group(1),split_args(m.group(2)))
for m in re.finditer(r'\b(?:void|int|float|double|lapack_int|sparse_status_t|mkl_jit_status_t|MKL_INT|CBLAS_INDEX|size_t)\s+\*?\s*(\w+)\s*\(([^;{}]*?)\)\s*(?:NOTHROW|MKL_DEPRECATED[^;]*)?\s*;',text,re.S):
    protos.setdefault(m.group(1),split_args(m.group(2)))
print(len(protos),'prototypes',file=sys.stderr)
def f(pfx,mid,suffix=''): return [pfx.replace('?',p)+mid+suffix for p in ('s','d','c','z')]
MAP={
 'sparse_create_csr':[f('mkl_sparse_?_create_csr','')[i] for i in range(4)],
}
def sp(n): return [f'mkl_sparse_{p}_{n}' for p in 'sdcz']
MAP={'mkl_imatcopy':[f'mkl_{p}imatcopy' for p in 'sdcz'],'mkl_omatcopy':[f'mkl_{p}omatcopy' for p in 'sdcz'],'mkl_omatcopy2':[f'mkl_{p}omatcopy2' for p in 'sdcz'],'mkl_omatadd':[f'mkl_{p}omatadd' for p in 'sdcz'],'sparse_qr_solve':[f'mkl_sparse_{p}_qr_solve' for p in 'sdcz'],'jit_create':[f'mkl_cblas_jit_create_{p}gemm' for p in 'sdcz'],'init_2d':['s_init_Helmholtz_2D','d_init_Helmholtz_2D'],'commit_2d':['s_commit_Helmholtz_2D','d_commit_Helmholtz_2D'],'solve_2d':['s_Helmholtz_2D','d_Helmholtz_2D'],'sparse_create_csr':sp('create_csr'),'sparse_create_coo':sp('create_coo'),'sparse_create_csc':sp('create_csc'),'sparse_create_bsr':sp('create_bsr'),'sparse_mm':sp('mm'),'sparse_dotmv':sp('dotmv'),'sparse_qr_solve':sp('qr_solve'),
 'rng_lognormal':['vsRngLognormal','vdRngLognormal'],'rng_beta':['vsRngBeta','vdRngBeta'],'rng_gaussian_mv':['vsRngGaussianMV','vdRngGaussianMV'],
 'ss_edit_moments':['vslsSSEditMoments','vsldSSEditMoments'],'ss_edit_missing_values':['vslsSSEditMissingValues','vsldSSEditMissingValues'],
 'vm_linear_frac':['vsLinearFrac','vdLinearFrac'],
 'feast_dense_real':['sfeast_syev','dfeast_syev'],'feast_csr_real':['sfeast_scsrev','dfeast_scsrev'],'feast_banded_real':['sfeast_sbev','dfeast_sbev'],'feast_dense_gen_real':['sfeast_sygv','dfeast_sygv'],
 'feast_dense_complex':['cfeast_heev','zfeast_heev'],'feast_csr_complex':['cfeast_hcsrev','zfeast_hcsrev'],'feast_banded_complex':['cfeast_hbev','zfeast_hbev'],'feast_dense_gen_complex':['cfeast_hegv','zfeast_hegv'],
 'df_edit_pp_spline_1d':['dfsEditPPSpline1D','dfdEditPPSpline1D'],'df_interpolate_1d':['dfsInterpolate1D','dfdInterpolate1D'],'df_integrate_1d':['dfsIntegrate1D','dfdIntegrate1D'],
 'lapacke_lacrm':['LAPACKE_clacrm','LAPACKE_zlacrm'],'lapacke_syconv':['LAPACKE_ssyconv','LAPACKE_dsyconv','LAPACKE_csyconv','LAPACKE_zsyconv'],
 'gemm_pack':[f'cblas_{p}gemm_pack' for p in 'sdcz'],'gemm_compute':[f'cblas_{p}gemm_compute' for p in 'sdcz'],
 'gemm_compact':[f'mkl_{p}gemm_compact' for p in 'sdcz'],'trsm_compact':[f'mkl_{p}trsm_compact' for p in 'sdcz'],'potrf_compact':[f'mkl_{p}potrf_compact' for p in 'sdcz'],
 'getrfnp_compact':[f'mkl_{p}getrfnp_compact' for p in 'sdcz'],'geqrf_compact':[f'mkl_{p}geqrf_compact' for p in 'sdcz'],'getrinp_compact':[f'mkl_{p}getrinp_compact' for p in 'sdcz'],
 'gepack_compact':[f'mkl_{p}gepack_compact' for p in 'sdcz'],'geunpack_compact':[f'mkl_{p}geunpack_compact' for p in 'sdcz'],
}
for line in open(sys.argv[2],encoding='utf-8'):
    m=re.match(r'(.+?):(\d+):\d+: warning: this function has too many arguments \((\d+)/7\)',line)
    if not m: continue
    fl,l,n=m.group(1),int(m.group(2)),int(m.group(3))
    s=open(f'{ROOT}/{fl}',encoding='utf-8').read().split('\n'); sig=s[l-1]
    nm=re.search(r'\bfn\s+(\w+)',sig)
    if not nm or 'unsafe fn' not in sig or nm.group(1) not in MAP: continue
    name=nm.group(1); c={k:protos.get(k) for k in MAP[name] if k in protos}
    ok=bool(c) and all(v==n for v in c.values())
    print(fl,l,name,n,'MIRROR' if ok else 'CHECK',c)
