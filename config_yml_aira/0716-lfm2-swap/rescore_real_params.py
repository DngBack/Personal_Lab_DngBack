#!/usr/bin/env python3
"""Rescore phase1_replay_metrics.*.jsonl with the REAL (tight) contest params.
Harness prints ERS with OLD loose params (100/1500/20/45); this uses the real
10/400/1/10. Usage: python3 rescore_real_params.py <dir with metrics jsonl>"""
import json, statistics, sys, glob, os
FT,CT,FP,CP,G,W = 10.0,400.0,1.0,10.0,2,0.5   # REAL params (2026-07-16)
def comp(x,f,c): return max(0.0,min(1.0,(c-x)/(c-f)))**G
def ers(recs):
    s=[]
    for r in recs:
        if r['error'] or r['ttft_ms'] is None: s.append(0.0); continue
        tpot=statistics.fmean(r['tbt_ms']) if r['tbt_ms'] else 0.0
        s.append(W*comp(r['ttft_ms'],FT,CT)+(1-W)*comp(tpot,FP,CP))
    return statistics.fmean(s)
D=sys.argv[1] if len(sys.argv)>1 else '.'
rows=[]
for f in glob.glob(f'{D}/phase1_replay_metrics.*.jsonl'):
    tag=os.path.basename(f).split('phase1_replay_metrics.')[1][:-6]
    recs=[json.loads(l) for l in open(f)]
    ok=[r for r in recs if not r['error'] and r['ttft_ms'] is not None]
    tt=sorted(r['ttft_ms'] for r in ok); tp=sorted(statistics.fmean(r['tbt_ms']) for r in ok if r['tbt_ms'])
    def p(a,q): return a[min(len(a)-1,int(q*(len(a)-1)))] if a else float('nan')
    rows.append((ers(recs),tag,p(tt,.5),p(tt,.95),p(tp,.5),p(tp,.95)))
print(f"{'ERS_real':>9} {'tag':22} {'ttft50':>7} {'ttft95':>7} {'tpot50':>7} {'tpot95':>7}")
for r in sorted(rows,reverse=True): print(f"{r[0]:9.4f} {r[1]:22} {r[2]:7.0f} {r[3]:7.0f} {r[4]:7.1f} {r[5]:7.1f}")
