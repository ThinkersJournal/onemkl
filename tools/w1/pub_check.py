# Usage (from the repo root): python tools/w1/pub_check.py <flagged.txt>; SURFACE = a function of that name with the same argument count exists in the earliest of the release tags listed in TAGS (count only).
import re, sys, subprocess

ROOT = '.'
TAGS = ['v0.1.0', 'v0.2.0', 'v0.3.0', 'v0.3.4', 'v0.3.8', 'v0.3.10', 'v0.3.11']
cache = {}


def git(*a):
    return subprocess.run(['git'] + list(a), capture_output=True, text=True,
                          cwd=ROOT, encoding='utf-8').stdout


def src(tag, f):
    if (tag, f) not in cache:
        cache[(tag, f)] = git('show', f'{tag}:{f}')
    return cache[(tag, f)]


def one(text, i):
    # count the parameters of the signature whose '(' ends just before i
    d = 1
    n = 0
    seen = False
    k = i
    while k < len(text) and d > 0:
        ch = text[k]
        if ch in '([{<':
            d += 1
        elif ch in ')]}':
            d -= 1
        elif ch == '>' and text[k - 1] != '-':
            d -= 1
        elif ch == ',' and d == 1:
            n += 1
            seen = False
        elif not ch.isspace() and d >= 1:
            seen = True
        k += 1
    return n + (1 if seen else 0)


def argcs(text, name):
    pat = r'\bpub\s+fn\s+' + re.escape(name) + r'\b\s*(?:<[^>]*(?:<[^>]*>[^>]*)*>)?\s*\('
    return [one(text, m.end()) for m in re.finditer(pat, text)]


for line in open(sys.argv[1], encoding='utf-8'):
    m = re.match(r'(.+?):(\d+):\d+: warning: this function has too many arguments \((\d+)/7\)', line)
    if not m:
        continue
    f, l, n = m.group(1), int(m.group(2)), int(m.group(3))
    now = open(f'{ROOT}/{f}', encoding='utf-8').read()
    s = now.split('\n')
    sig = s[l - 1]
    nm = re.search(r'\bfn\s+(\w+)', sig)
    if not nm or 'unsafe fn' in sig or not re.search(r'\bpub\b', sig):
        continue
    name = nm.group(1)
    since = None
    for t in TAGS:
        if n in argcs(src(t, f), name):
            since = t
            break
    ok = since is not None and n in argcs(now, name)
    dup = len(argcs(now, name))
    print(f, l, name, n, 'SURFACE' if ok else 'CHECK', since, 'dupnames=%d' % dup)
