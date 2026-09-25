"""Run the rater study with a vision--language model in place of the people.

vlm_pref.py was the first attempt and it failed its own sanity check: asked which of two sprites looked
more like real pixel art, it ranked large generators above genuine hand-drawn sprites, so its verdict
about ours could not be read either way. That failure is the reason the screening pairs exist. This
script therefore runs the *same package the humans ran* -- runs/study_web/study.json, the same twenty
screening pairs and the same judgement trials -- so the model's control score sits beside the humans'
3.36 of 6 and both can be read on one scale.

Two differences from vlm_pref.py, both there to keep a win from being an artefact:

  * every trial is asked twice, once with the sides as built and once swapped. A model that answers the
    same way both times is responding to the sprites; one that follows a side is not, and `consistent`
    in the report says which this was. Only consistent trials count toward the win rates;
  * the screening pairs are asked too. They have a ground truth that is a fact about the file, so they
    measure whether the judge can see the distinction at all, exactly as they do for a person.

The gateway key never enters the repository. Set PIXEL_API_KEY in the environment.

Usage:
  PIXEL_API_KEY=... python baseline/ai_review.py --model gpt-5.6-sol --out runs/ai_review.jsonl
  python baseline/ai_review.py --report runs/ai_review.jsonl
"""
import argparse
import base64
import io
import json
import math
import os
import re
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from queue import Queue

from PIL import Image

BASE = os.environ.get("PIXEL_API_BASE", "http://113.45.39.247:3001/v1")

JUDGE_PROMPT = """These are two 16x16 pixel-art game sprites, shown magnified side by side on a
checkerboard background (the checkerboard is transparency, not part of the art). Both were made for this
description:

"{caption}"

{question}

Answer with exactly one word: LEFT or RIGHT."""

SCREEN_PROMPT = """These are two 16x16 game sprites, shown magnified side by side on a checkerboard
background (the checkerboard is transparency, not part of the art). Both are real sprites from a game-art
archive.

{question}

Answer with exactly one word: LEFT or RIGHT."""


def pair_image(a, b, gap=24):
    ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    h = max(ia.height, ib.height)
    out = Image.new("RGB", (ia.width + gap + ib.width, h), (255, 255, 255))
    out.paste(ia, (0, 0))
    out.paste(ib, (ia.width + gap, 0))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call(model, key, prompt, b64, retries=4):
    body = json.dumps({
        "model": model, "max_tokens": 2000,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}],
    }).encode()
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                         headers={"Authorization": "Bearer " + key,
                                                  "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 404):      # a wrong model id should stop the run, not retry it 520 times
                raise SystemExit(f"gateway rejected model {model!r}: {e.code} {e.read()[:200]!r}")
            time.sleep(3 * (a + 1))
        except Exception as e:
            last = e
            time.sleep(3 * (a + 1))
    raise last


def trials_from(study):
    """Every trial in both orderings; `swap` says which way this one was shown."""
    s = json.loads(Path(study).read_text(encoding="utf-8"))
    out = []
    for t in s["judge"]:
        for swap in (False, True):
            out.append({"id": t["id"], "kind": "judge", "question": t["question"], "text": t["text"],
                        "prompt": t.get("prompt", ""), "swap": swap,
                        "left": t["right"] if swap else t["left"],
                        "right": t["left"] if swap else t["right"],
                        "left_img": t["right_img"] if swap else t["left_img"],
                        "right_img": t["left_img"] if swap else t["right_img"]})
    for t in s["screen"]:
        for swap in (False, True):
            ans = t["answer"]
            if swap:
                ans = "left" if ans == "right" else "right"
            out.append({"id": t["id"], "kind": "screen", "question": "fidelity", "text": t["text"],
                        "swap": swap, "answer": ans,
                        "left_img": t["right_img"] if swap else t["left_img"],
                        "right_img": t["left_img"] if swap else t["right_img"]})
    return out


def ci(w, n):
    if not n:
        return 0.0, 0.0
    p = w / n
    h = 1.96 * math.sqrt(p * (1 - p) / n)
    return 100 * max(0.0, p - h), 100 * min(1.0, p + h)


def report(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    by = defaultdict(dict)
    for r in rows:
        by[(r["kind"], r["id"], r["question"])][r["swap"]] = r
    both = {k: v for k, v in by.items() if len(v) == 2}
    print(f"{len(rows)} answers, {len(by)} trials, {len(both)} asked both ways")

    scr = [v for k, v in both.items() if k[0] == "screen"]
    cons = [v for v in scr if v[False]["correct"] == v[True]["correct"]]
    right = sum(1 for v in cons if v[False]["correct"])
    print(f"\nscreening: {len(scr)} pairs, {len(cons)} answered consistently "
          f"({100*len(cons)/max(1,len(scr)):.0f}%), {right}/{len(cons)} correct"
          f"{f' = {100*right/len(cons):.0f}%' if cons else ''}   (chance = 50%)")
    print("  humans on the same pairs: 37/66 = 56%, mean 3.36 of 6")

    jud = {k: v for k, v in both.items() if k[0] == "judge"}
    flip = sum(1 for v in jud.values() if v[False]["winner"] != v[True]["winner"])
    print(f"\njudgement: {len(jud)} trials, {flip} ({100*flip/max(1,len(jud)):.0f}%) flipped with position"
          f" and are dropped")
    for q in ("fidelity", "preference"):
        tot = defaultdict(lambda: [0, 0])
        for k, v in jud.items():
            if k[2] != q or v[False]["winner"] != v[True]["winner"]:
                continue
            w, l = v[False]["winner"], v[False]["loser"]
            tot[w][0] += 1
            tot[w][1] += 1
            tot[l][1] += 1
        if not tot:
            continue
        print(f"  {q}")
        for s in sorted(tot, key=lambda k: -(tot[k][0] / max(1, tot[k][1]))):
            w, n = tot[s]
            lo, hi = ci(w, n)
            print(f"    {s:6s} {w:4d}/{n:<4d} {100*w/n:5.1f}%  [{lo:.0f}, {hi:.0f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="runs/study_web/study.json")
    ap.add_argument("--dir", default="runs/study_web", help="where the trial images live")
    ap.add_argument("--out", default="runs/ai_review.jsonl")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    if a.report:
        return report(a.report)

    key = os.environ.get("PIXEL_API_KEY", "")
    if not key:
        sys.exit("set PIXEL_API_KEY (the gateway key; it must not be committed)")
    d = Path(a.dir)
    trials = trials_from(a.study)
    if a.limit:
        trials = trials[:a.limit]
    done = set()
    if Path(a.out).exists():
        done = {f"{json.loads(l)['kind']}|{json.loads(l)['id']}|{json.loads(l)['question']}|"
                f"{json.loads(l)['swap']}" for l in open(a.out, encoding="utf-8")}
    todo = [t for t in trials if f"{t['kind']}|{t['id']}|{t['question']}|{t['swap']}" not in done]
    print(f"{len(trials)} asks ({len(trials)//2} trials x 2 orderings), {len(todo)} to run", flush=True)

    q = Queue()
    for t in todo:
        q.put(t)
    lock = threading.Lock()
    fh = open(a.out, "a", encoding="utf-8")
    bad = [0]

    def worker():
        while True:
            try:
                t = q.get_nowait()
            except Exception:
                return
            try:
                b64 = pair_image(d / t["left_img"], d / t["right_img"])
                p = (SCREEN_PROMPT.format(question=t["text"]) if t["kind"] == "screen"
                     else JUDGE_PROMPT.format(caption=t["prompt"], question=t["text"]))
                txt = call(a.model, key, p, b64)
            except SystemExit:
                raise
            except Exception as e:
                print(f"failed {t['id']}: {str(e)[:90]}", flush=True)
                continue
            m = re.search(r"\b(LEFT|RIGHT)\b", txt.upper())
            if not m:
                with lock:
                    bad[0] += 1
                print(f"unparsed {t['id']}: {txt[:50]!r}", flush=True)
                continue
            side = m.group(1).lower()
            rec = {"kind": t["kind"], "id": t["id"], "question": t["question"], "swap": t["swap"],
                   "side": side, "model": a.model}
            if t["kind"] == "judge":
                rec["winner"] = t["left"] if side == "left" else t["right"]
                rec["loser"] = t["right"] if side == "left" else t["left"]
            else:
                rec["correct"] = side == t["answer"]
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()

    ts = [threading.Thread(target=worker) for _ in range(a.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(f"AI_REVIEW_DONE  unparsed={bad[0]}", flush=True)
    report(a.out)


if __name__ == "__main__":
    main()
