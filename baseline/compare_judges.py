"""Put every judge that has run the study into one table.

Four kinds of evidence have accumulated and they are easy to quote against each other by accident, so
this prints them on one page with the two things that decide how much each is worth: the control score,
and how much of the judge's output had to be thrown away.

  humans          eleven people, the FIRST package (palettes not matched, one blank prompt)
  model raters    N sessions of one model run exactly as a person was, the corrected package
  model reviews   one model answering every trial twice with the sides swapped, corrected package

The three are not interchangeable. The humans ran a package with a fault in it, so their numbers are
comparable only to a model run on that same package. Model raters share one model's biases across
sessions, so the spread between sessions is not the spread between people. Model reviews drop every
trial that changes answer when the sides swap, which is a stronger filter than any rater applies to
themselves. Each row therefore carries what it dropped and what it scored on the control, and the
reading is in those columns as much as in the win rates.

Usage: python baseline/compare_judges.py
"""
import argparse
import base64
import json
import math
import re
from collections import defaultdict
from pathlib import Path

PASS_MIN = 5


def ci(w, n):
    if not n:
        return 0.0, 0.0
    p = w / n
    h = 1.96 * math.sqrt(p * (1 - p) / n)
    return 100 * max(0.0, p - h), 100 * min(1.0, p + h)


def from_codes(path, gate=False):
    """A `PXS1:` file, whether the raters were people or sessions of a model.

    gate=True applies the pre-registered rule -- only raters at 5 of 6 on the control are counted. It is
    the reading the study was designed around, and it is reported separately because for the eleven
    people it leaves two raters and answers nothing, while for ten model sessions it leaves seven and
    changes the ordering."""
    text = Path(path).read_text(encoding="utf-8")
    raters = {}
    for b in re.findall(r"PXS1:([A-Za-z0-9+/=]+)", text):
        try:
            p = json.loads(base64.b64decode(b).decode("utf-8"))
        except Exception:
            continue
        raters[p["rater"]] = p
    rows, scr_w, scr_n, passed, kept = [], 0, 0, 0, 0
    for p in raters.values():
        ok = p.get("screenPassed", 0) >= PASS_MIN
        passed += ok
        if gate and not ok:
            continue
        kept += 1
        scr_w += p.get("screenPassed", 0)
        scr_n += p.get("screenTotal", 0)
        rows += [a for a in p["answers"] if a["kind"] == "judge"]
    return {"n_raters": len(raters), "passed": passed, "screen": (scr_w, scr_n),
            "rows": rows, "dropped": (len(raters) - kept, len(raters)) if gate else None}


def from_review(path):
    """A two-ordering review file; only trials that answer the same way both times survive."""
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    by = defaultdict(dict)
    for r in recs:
        by[(r["kind"], r["id"], r["question"])][r["swap"]] = r
    both = {k: v for k, v in by.items() if len(v) == 2}
    scr = [v for k, v in both.items() if k[0] == "screen"]
    cons_s = [v for v in scr if v[False]["correct"] == v[True]["correct"]]
    jud = {k: v for k, v in both.items() if k[0] == "judge"}
    keep = {k: v for k, v in jud.items() if v[False]["winner"] == v[True]["winner"]}
    rows = [{"kind": "judge", "question": k[2], "chose": v[False]["winner"], "over": v[False]["loser"]}
            for k, v in keep.items()]
    return {"n_raters": 1, "passed": None,
            "screen": (sum(1 for v in cons_s if v[False]["correct"]), len(cons_s)),
            "rows": rows, "dropped": (len(jud) - len(keep), len(jud))}


def rates(rows, q):
    tot = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["question"] != q:
            continue
        tot[r["chose"]][0] += 1
        tot[r["chose"]][1] += 1
        tot[r["over"]][1] += 1
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=["real", "ours", "gpt", "flux"])
    a = ap.parse_args()
    sources = [
        ("humans x11", "first package", from_codes, "runs/study_web/replies.txt"),
        ("humans, screening >=5", "first package", lambda q: from_codes(q, gate=True),
         "runs/study_web/replies.txt"),
        ("gpt-5.6-sol x10 raters", "corrected", from_codes, "runs/ai_raters.txt"),
        ("  same, screening >=5", "corrected", lambda q: from_codes(q, gate=True), "runs/ai_raters.txt"),
        ("gpt-5.6-sol review", "first package", from_review, "runs/ai_review.jsonl"),
        ("gpt-5.6-sol review", "corrected", from_review, "runs/ai_review_fair.jsonl"),
        ("claude-opus-5 review", "corrected", from_review, "runs/ai_review_opus.jsonl"),
        ("claude-sonnet-4-6 review", "corrected", from_review, "runs/ai_review_sonnet.jsonl"),
        ("gemini-3.1-pro review", "corrected", from_review, "runs/ai_review_gemini.jsonl"),
        ("gemini-3.8-flash review", "corrected", from_review, "runs/ai_review_flash.jsonl"),
    ]
    got = []
    for name, pkg, fn, path in sources:
        if not Path(path).exists():
            continue
        # a review is 520 asks; a partial file is a gateway outage, not a verdict, and averaging what
        # got through would quietly weight whichever trials happened to succeed
        if fn is from_review:  # noqa: E721
            n = sum(1 for _ in open(path, encoding="utf-8"))
            if n < 520:
                print(f"skipped {name}: {n}/520 asks returned (gateway failures), not reported")
                continue
        try:
            got.append((name, pkg, fn(path)))
        except Exception as e:
            print(f"skipped {path}: {str(e)[:60]}")

    print(f"{'judge':24s} {'package':14s} {'control':>12s} {'dropped':>10s}")
    for name, pkg, d in got:
        w, n = d["screen"]
        ctrl = f"{w}/{n}" + (f" {100*w/n:.0f}%" if n else "")
        dr = f"{d['dropped'][0]}/{d['dropped'][1]}" if d["dropped"] else "--"
        print(f"{name:24s} {pkg:14s} {ctrl:>12s} {dr:>10s}")
    print("  chance on the control is 50%")

    for q in ("fidelity", "preference"):
        print(f"\n=== {q}   win rate [95% CI]")
        head = "".join(f"{s:>18s}" for s in a.systems)
        print(f"{'judge':24s} {'package':14s}{head}")
        for name, pkg, d in got:
            t = rates(d["rows"], q)
            cells = ""
            for s in a.systems:
                w, n = t.get(s, [0, 0])
                if n:
                    lo, hi = ci(w, n)
                    cells += f"{100*w/n:6.1f} [{lo:2.0f},{hi:3.0f}]".rjust(18)
                else:
                    cells += f"{'--':>18s}"
            print(f"{name:24s} {pkg:14s}{cells}")


if __name__ == "__main__":
    main()
