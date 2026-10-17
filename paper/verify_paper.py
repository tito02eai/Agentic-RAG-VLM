import re

tex = open('main.tex', 'r', encoding='utf-8').read()
lines = tex.split('\n')
print('Total lines:', len(lines))

sections = re.split(r'\\section', tex)
total = 0
for i, s in enumerate(sections):
    m = re.match(r'\*?\{(.+?)\}', s.strip())
    t = m.group(1) if m else '(preamble/abstract)'
    cleaned = re.sub(r'\\[a-zA-Z]+', '', s)
    cleaned = re.sub(r'[{}$&^_~]', '', cleaned)
    cleaned = re.sub(r'%.*', '', cleaned)
    w = len(cleaned.split())
    total += w
    print(f'  {i}: {w:4d}w | {t[:55]}')
print(f'Total: {total} words')

status = 'OK' if total >= 5000 else 'Need more'
print(f'Target check: {status} (target 5500-6500)')

begins = re.findall(r'\\begin\{(\w+)\}', tex)
ends = re.findall(r'\\end\{(\w+)\}', tex)
ob = tex.count('{')
cb = tex.count('}')
print(f'Braces: open={ob}, close={cb}, match={ob==cb}')
print(f'Envs: {len(begins)} begin / {len(ends)} end')

from collections import Counter
bc = Counter(begins)
ec = Counter(ends)
for env in set(list(bc.keys()) + list(ec.keys())):
    if bc[env] != ec[env]:
        print(f'  MISMATCH env {env}: begin={bc[env]} end={ec[env]}')

cites = set()
for c in re.findall(r'\\cite\{([^}]+)\}', tex):
    for k in c.split(','):
        cites.add(k.strip())
bib = open('references.bib', 'r', encoding='utf-8').read()
bibkeys = set(re.findall(r'@\w+\{(\w+),', bib))
missing = cites - bibkeys
print(f'Citations used: {len(cites)}, missing: {missing if missing else "none"}')

tables = len(re.findall(r'label\{tab:', tex))
figs = len(re.findall(r'label\{fig:', tex))
algs = len(re.findall(r'label\{alg:', tex))
eqs = len(re.findall(r'label\{eq:', tex))
print(f'Tables: {tables}, Figures: {figs}, Algorithms: {algs}, Equations: {eqs}')
