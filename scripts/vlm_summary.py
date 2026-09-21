"""Summarise the VLM forced-choice results: win rates per system, per question, with a binomial CI."""
import json
import math
import sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "runs/vlm_pref.jsonl",
                                    encoding="utf-8")]
SYS = ["real", "ours", "gpt", "flux", "sdxl"]


def ci(w, n):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    h = 1.96 * math.sqrt(p * (1 - p) / n)
    return 100 * max(0, p - h), 100 * min(1, p + h)


for q in ("fidelity", "preference"):
    sub = [r for r in rows if r["question"] == q]
    print(f"\n=== {q}  ({len(sub)} judgements)")
    tot = defaultdict(lambda: [0, 0])
    head = defaultdict(lambda: [0, 0])
    for r in sub:
        tot[r["winner"]][0] += 1
        tot[r["winner"]][1] += 1
        tot[r["loser"]][1] += 1
        head[(r["winner"], r["loser"])][0] += 1
        head[(r["winner"], r["loser"])][1] += 1
        head[(r["loser"], r["winner"])][1] += 1
    print(f"{'system':8s} {'wins/trials':>13s} {'win rate':>9s}   95% CI")
    for s in SYS:
        w, n = tot[s]
        lo, hi = ci(w, n)
        if n:
            print(f"{s:8s} {w:6d}/{n:<6d} {100 * w / n:8.1f}%   [{lo:.1f}, {hi:.1f}]")
    print("  head to head (row beats column):")
    for a in SYS:
        cells = []
        for b in SYS:
            if a == b:
                cells.append("   -  ")
                continue
            w, n = head[(a, b)]
            cells.append(f"{100 * w / n:5.0f}%" if n else "   .  ")
        print(f"    {a:6s} " + " ".join(cells))
    print("           " + " ".join(f"{b:>6s}" for b in SYS))
