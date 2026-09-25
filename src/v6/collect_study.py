"""Decode the codes raters paste back and report the study.

Each rater finishes with a block starting `PXS1:`. Put them in a file, one per line (extra text around
them is ignored), and this decodes, de-duplicates by rater id, and reports win rates per system pair.

The screening pairs decide how a rater is counted. They have a ground truth that is a fact about the
file -- one sprite was drawn at 16 px, the other is larger artwork reduced to it -- so a rater who
cannot separate them is not evidence about our samples either way. Raters at or below chance are
reported separately rather than dropped silently, and the pre-registered reading stands: if the real
sprites do not win the fidelity question against the external systems, that question measured polish
and is reported as such.

Usage: python src/v6/collect_study.py --in runs/study_web/replies.txt
"""
import argparse
import base64
import json
import math
import re
from collections import defaultdict
from pathlib import Path

PASS_MIN = 5   # of 6 screening pairs; 6 pairs, chance = 3


def ci(w, n):
    if not n:
        return 0.0, 0.0
    p = w / n
    h = 1.96 * math.sqrt(p * (1 - p) / n)
    return 100 * max(0.0, p - h), 100 * min(1.0, p + h)


def report(rows, label):
    if not rows:
        print(f"\n{label}: no answers")
        return
    print(f"\n=== {label}  ({len(rows)} judgements from {len({r['rater'] for r in rows})} raters)")
    for q in ("fidelity", "preference"):
        sub = [r for r in rows if r["question"] == q]
        if not sub:
            continue
        tot, head = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        for r in sub:
            tot[r["chose"]][0] += 1
            tot[r["chose"]][1] += 1
            tot[r["over"]][1] += 1
            head[(r["chose"], r["over"])][0] += 1
            head[(r["chose"], r["over"])][1] += 1
            head[(r["over"], r["chose"])][1] += 1
        print(f"  {q} ({len(sub)} answers)")
        for s in sorted(tot, key=lambda k: -(tot[k][0] / max(1, tot[k][1]))):
            w, n = tot[s]
            lo, hi = ci(w, n)
            print(f"    {s:6s} {w:4d}/{n:<4d} {100 * w / n:5.1f}%  [{lo:.0f}, {hi:.0f}]")
        pairs = sorted({tuple(sorted(k)) for k in head})
        for a, b in pairs:
            w, n = head[(a, b)]
            if n:
                print(f"      {a} beats {b}: {100 * w / n:.0f}%  (n={n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="runs/study_web/replies.txt")
    ap.add_argument("--json", default=None, help="also write the merged answers here")
    a = ap.parse_args()
    text = Path(a.inp).read_text(encoding="utf-8")
    blobs = re.findall(r"PXS1:([A-Za-z0-9+/=]+)", text)
    print(f"{len(blobs)} codes found in {a.inp}")

    raters = {}
    for b in blobs:
        try:
            p = json.loads(base64.b64decode(b).decode("utf-8"))
        except Exception as e:
            print(f"  skipped an unreadable code: {str(e)[:60]}")
            continue
        raters[p["rater"]] = p          # a re-paste of the same rater replaces, never doubles
    print(f"{len(raters)} distinct raters")

    good, weak, rows_g, rows_w = [], [], [], []
    for rid, p in sorted(raters.items()):
        passed, total = p.get("screenPassed", 0), p.get("screenTotal", 0)
        ok = passed >= PASS_MIN
        (good if ok else weak).append((rid, passed, total))
        for r in p["answers"]:
            if r["kind"] != "judge":
                continue
            r = dict(r, rater=rid)
            (rows_g if ok else rows_w).append(r)
    print(f"\nscreening (pass = {PASS_MIN} of 6, chance = 3):")
    for rid, ps, tt in good:
        print(f"  {rid}  {ps}/{tt}  counted")
    for rid, ps, tt in weak:
        print(f"  {rid}  {ps}/{tt}  reported separately")

    report(rows_g, "raters who passed screening")
    report(rows_w, "raters who did not")
    if a.json:
        Path(a.json).write_text(json.dumps({"raters": raters}, indent=1), encoding="utf-8")
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
