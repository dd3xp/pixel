"""Inception clean-FID / KID (diag_metric2, fid_kid_fair.py) vs FD-DINOv2 on the same saved samples: rank agreement per R
and the main-row table.  Data: paper_assets/data/{incep_scores.json, kw_fd.txt (all FAIR FD lines from logs)}.  Run from repo root."""
import json,re
from collections import defaultdict
import numpy as np
inc=json.load(open('paper_assets/data/incep_scores.json'))
fd={}
for l in open('paper_assets/data/kw_fd.txt'):
    p=l.split()
    if len(p)==3: fd[(int(p[0]),p[1])]=float(p[2])
rows=defaultdict(dict)
for r in inc:
    m=re.match(r'runs_out/(.*)_matched_eval/s(\d+)',r['gen'])
    if not m: rows[r['size']]['__floor']=(r['fid'],r['kid']*1000); continue
    rows[r['size']][m.group(1)]=(r['fid'],r['kid']*1000)
def spear(a,b):
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b)); return np.corrcoef(ra,rb)[0,1]
for R in sorted(rows):
    xs=[];ys=[];ks=[]
    for tag,(f,k) in rows[R].items():
        if (R,tag) in fd: xs.append(fd[(R,tag)]); ys.append(f); ks.append(k)
    print(f"R={R} floor FID/KID={rows[R].get('__floor')} n_paired={len(xs)} spearman(FD-DINO,FID)={spear(xs,ys):.3f} spearman(FD-DINO,KID)={spear(xs,ks):.3f} pearson={np.corrcoef(xs,ys)[0,1]:.3f}")
main={16:['v7h','v7h_gbk12_w2','v7h_autog10k_w1p5','v7h_gbk12s10k_w1p5','v7h_gbk24_w2','v7h_gbk64_w2','v7h_bk12only','v7h_seed1','v7h_gbk12_w2_seed1','v7h_autog10k_w1p5_seed1','v7h_gbk12s10k_w1p5_seed1','v7h_cfg4_seed2','v7h_gbk12_w2_seed2','v7h_autog10k_w1p5_seed2','v7h_gbk12s10k_w1p5_seed2','v7s_cfg4','v7s_gbk12_w2','v7s_autog10k_w1p5','v7s_gbk12s10k_w1p5','v7s_gbk20_w2','v7h_bku12s10k_w1p5','v7h_gbk12s10k_w1p5_ct1p5','probe_tv_cfg7','probe_tv_cfg10','probe_tv_ddim50','v7h_ddpm50'],
 20:['v7h_r20_cfg4','v7h_r20_bk16_w2','v7h_r20_bk12_w2','v7h_r20_autog10k_w1p5','v7h_r20_bk16s10k_w1p5','v7h_r20_bk24_w2','v7h_r20_bk32_w2','v7s_r20_cfg4','v7s_r20_bk16_w2','v7s_r20_autog10k_w1p5','v7s_r20_bk16s10k_w1p5'],
 24:['v7h_r24_cfg4','v7h_r24_bk16_w2','v7h_r24_bk12_w2','v7h_r24_bk20_w2','v7h_r24_autog10k_w1p5','v7h_r24_bk16s10k_w1p5','v7h_r24_bk32_w2','v7s_r24_cfg4','v7s_r24_bk16_w2','v7s_r24_autog10k_w1p5','v7s_r24_bk16s10k_w1p5'],
 12:['v7h_r12_cfg4','v7h_r12_autog10k_w1p5','v7h_r12_bk16_w2'],
 32:['v7h_r32_cfg4','v7h_r32_bk24_w2','v7h_r32_bk16_w2','v7h_r32_autog10k_w1p5','v7h_r32_bk24s10k_w1p5','v7h_r32_bk48_w2']}
for R in [12,16,20,24,32]:
    print(f"\n== R={R} floor FID={rows[R]['__floor'][0]:.2f} KID={rows[R]['__floor'][1]:.2f}e-3")
    for t in main[R]:
        if t in rows[R]: print(f"{t:34s} FD-DINO {fd.get((R,t),float('nan')):7.2f}  FID {rows[R][t][0]:7.2f}  KID {rows[R][t][1]:6.2f}e-3")
        else: print(f"{t:34s} (missing)")
