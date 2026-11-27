# Usage (from the repo root): python tools/w1/classify.py <flagged.txt> <out.md>; writes docs/W1-TOO-MANY-ARGUMENTS.md.
import re,subprocess,sys,collections
ROOT='.'
def git(*a):
    r=subprocess.run(['git']+list(a),capture_output=True,text=True,cwd=ROOT,encoding='utf-8'); return r.stdout
TAGS=['v0.1.0','v0.2.0','v0.3.0','v0.3.4','v0.3.8','v0.3.10','v0.3.11']
tagsrc={}
def src_at(tag,f):
    k=(tag,f)
    if k not in tagsrc: tagsrc[k]=git('show',f'{tag}:{f}')
    return tagsrc[k]
rows=[]
for line in open(sys.argv[1],encoding='utf-8'):
    m=re.match(r'(.+?):(\d+):\d+: warning: this function has too many arguments \((\d+)/7\)',line)
    if not m: continue
    f,l,n=m.group(1),int(m.group(2)),int(m.group(3))
    lines=open(f'{ROOT}/{f}',encoding='utf-8').read().split('\n')
    sig=lines[l-1]
    nm=re.search(r'\bfn\s+(\w+)',sig)
    if not nm:
        for k in range(l-1,min(l+3,len(lines))):
            nm=re.search(r'\bfn\s+(\w+)',lines[k])
            if nm: break
    name=nm.group(1) if nm else '?'
    unsafe='unsafe fn' in sig
    pub=bool(re.search(r'\bpub\b',sig))
    ind=len(sig)-len(sig.lstrip())
    # doc lines above
    k=l-2; doc=[]
    while k>=0 and (lines[k].strip().startswith('///') or lines[k].strip().startswith('#[')):
        if lines[k].strip().startswith('///'): doc.append(lines[k])
        k-=1
    d=' '.join(doc)
    cname=re.findall(r'`((?:cblas_|LAPACKE_|mkl_|dfti|Dfti|df[sd]|v[sdcz]|sparse_|[sd]_)[\w*?]*)`',d)
    doc_c = bool(cname) or bool(re.search(r'`[^`]*\*[^`]*`',d))
    since=''
    if pub:
        for t in TAGS:
            if re.search(r'\bfn\s+'+re.escape(name)+r'\b',src_at(t,f)): since=t; break
    if unsafe: cls='C-mirror (raw FFI method)' + ('' if doc_c else ': doc does not name the C function')
    elif since: cls=f'public surface (published since {since}); C-mirror if it takes one C call\'s arguments'
    elif pub: cls='unpublished pub fn: restructure (parameter struct, not breaking)'
    else: cls='private fn: restructure or reason'
    rows.append((f,l,name,n,'unsafe' if unsafe else ('pub' if pub else 'fn'),cls))
by=collections.Counter(r[5].split(':')[0].split(' (')[0] for r in rows)
print(len(rows),dict(by),file=sys.stderr)
out=['# W1: functions flagged by `clippy::too_many_arguments` when the source allows are overridden','',
 'Generated 2026-11-27 on `origin/main` 65478879 (rustc / clippy 1.99.1, WSL) from `cargo clippy -p onemkl --lib -- --force-warn clippy::too_many_arguments`; `since` is the earliest of the release tags v0.1.0, v0.2.0, v0.3.0, v0.3.4, v0.3.8, v0.3.10, v0.3.11 in whose copy of the same file a function of that name exists. The classification is mechanical (signature text and doc comment), a FIRST PASS for review: nothing here has been confirmed against the C headers argument by argument, and the C-mirror verdicts mean "likely", not "verified". Counted waivers already carrying a `reason` are in the table too (7 as of this run: 6 from #21, 1 from #31).','',
 '| file | line | function | args | kind | first-pass class |','|---|---|---|---|---|---|']
for f,l,n,a,k,c in rows: out.append(f'| `{f}` | {l} | `{n}` | {a} | {k} | {c} |')
out.append('')
out.append('Summary by class: '+', '.join(f'{k} {v}' for k,v in sorted(by.items()))+f'; total {len(rows)}.')
open(sys.argv[2],'w',encoding='utf-8',newline='\n').write('\n'.join(out)+'\n')
