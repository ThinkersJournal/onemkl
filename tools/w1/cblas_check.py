# Usage (from the repo root): python tools/w1/cblas_check.py <mkl_cblas.h> <flagged.txt>; flagged.txt is the output of
#   cargo clippy -p onemkl --lib --message-format=short -- --force-warn clippy::too_many_arguments
# filtered to "too many arguments" lines. Prints MIRROR when the Rust argument count equals that of every precision variant of the C function.
import re,sys
ROOT='.'
hdr=open(sys.argv[1],encoding='utf-8',errors='replace').read()
protos={}
for m in re.finditer(r'\b(?:void|int|float|double|CBLAS_INDEX|size_t|MKL_INT|CBLAS_\w+)\s+\*?\s*(cblas_\w+)\s*\(([^;]*?)\)\s*(?:NOTHROW|MKL_DEPRECATED[^;]*)?\s*;',hdr,re.S):
    args=m.group(2).strip()
    n=0 if args in('','void') else args.count(',')+1
    protos[m.group(1)]=n
print(len(protos),'prototypes',file=sys.stderr)
rows=[]
for line in open(sys.argv[2],encoding='utf-8'):
    m=re.match(r'(onemkl/src/blas/scalar\.rs):(\d+):\d+: warning: this function has too many arguments \((\d+)/7\)',line)
    if not m: continue
    f,l,n=m.group(1),int(m.group(2)),int(m.group(3))
    src=open(f'{ROOT}/{f}',encoding='utf-8').read().split('\n')
    nm=re.search(r'\bfn\s+(cblas_\w+)',src[l-1])
    if not nm: continue
    base=nm.group(1)[len('cblas_'):]
    cands={}
    for p in ('s','d','c','z','ds','sd','dz','sc','cs','zd','is','id','ic','iz'):
        k='cblas_'+p+base
        if k in protos: cands[k]=protos[k]
    for k in ('cblas_'+base,):
        if k in protos: cands[k]=protos[k]
    ok=[k for k,v in cands.items() if v==n]
    rows.append((l,nm.group(1),n,cands,bool(ok) and len(ok)==len(cands)))
for r in rows:
    print(r[0],r[1],r[2],'MIRROR' if r[4] else 'CHECK',dict(list(r[3].items())[:4]))
print(sum(1 for r in rows if r[4]),'mirror of',len(rows),file=sys.stderr)
