# Usage (from the repo root): python tools/w1/cblas_check.py <mkl_cblas.h> <flagged.txt>; flagged.txt is the output of
#   cargo clippy -p onemkl --lib --message-format=short -- --force-warn clippy::too_many_arguments
# filtered to "too many arguments" lines. Prints MIRROR when the Rust argument count equals that of every precision variant of the C function.
import re,sys
ROOT='.'
hdr=open(sys.argv[1],encoding='utf-8',errors='replace').read()
protos={}
for m in re.finditer(r'\b(?:lapack_int|double|float|void|lapack_logical)\s+(LAPACKE_\w+)\s*\(([^;]*?)\)\s*;',hdr,re.S):
    a=m.group(2).strip(); protos[m.group(1)]=0 if a in('','void') else a.count(',')+1
print(len(protos),'prototypes',file=sys.stderr)
f='onemkl/src/lapack/scalar.rs'
src=open(f'{ROOT}/{f}',encoding='utf-8').read().split('\n')
out=[]
for line in open(sys.argv[2],encoding='utf-8'):
    m=re.match(r'(onemkl/src/lapack/scalar\.rs):(\d+):\d+: warning: this function has too many arguments \((\d+)/7\)',line)
    if not m: continue
    l,n=int(m.group(2)),int(m.group(3))
    nm=re.search(r'\bfn\s+(lapacke_\w+)',src[l-1])
    if not nm: out.append((l,'?',n,'SKIP',{})); continue
    base=nm.group(1)[len('lapacke_'):]
    pre=('s','d','c','z')
    if base.endswith('_complex'): base=base[:-8]; pre=('c','z')
    elif any(('fn lapacke_'+base+'_complex') in x for x in src): pre=('s','d')
    cands={}
    for p in pre:
        k='LAPACKE_'+p+base
        if k in protos: cands[k]=protos[k]
    ok=bool(cands) and all(v==n for v in cands.values())
    out.append((l,nm.group(1),n,'MIRROR' if ok else 'CHECK',cands))
for r in out: print(r[0],r[1],r[2],r[3],dict(list(r[4].items())[:4]))
print(sum(1 for r in out if r[3]=='MIRROR'),'mirror of',len(out),file=sys.stderr)
