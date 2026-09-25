"""Stand a model in for the rater pool: N independent sessions, each run exactly as a person's was.

The eleven people who ran the study sat at chance on the control (3.36 of 6) and put real hand-drawn
sprites last, so their answers cannot settle the question and no more people are available. A model can
take their place only if it takes their place exactly: the web page gives each rater six screening pairs
and eighteen judgement trials drawn at random from the package, flips left and right at random, and
records the answer. This does the same thing per session, with a different draw per session, and emits
the same `PXS1:` block the page emits -- so `collect_study.py` reads these and the humans' with the same
decoder, the same screening gate and the same de-duplication, and the two columns are comparable because
nothing downstream knows which is which.

Sessions are independent by construction: one request per trial, no conversation, no shared context. What
they are not is independent *samples of a population* -- ten sessions of one model share its biases the
way ten people do not, so the spread across sessions understates the uncertainty. The report says so.

Usage:
  PIXEL_API_KEY=... python baseline/ai_raters.py --model gpt-5.6-sol --n 10 \
      --study runs/study_web2/study.json --dir runs/study_web2 --out runs/ai_raters.txt
  python src/v6/collect_study.py --in runs/ai_raters.txt
"""
import argparse
import base64
import json
import os
import random
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue

sys.path.insert(0, str(Path(__file__).parent))
from ai_review import JUDGE_PROMPT, SCREEN_PROMPT, call, pair_image  # noqa: E402

N_SCREEN, N_JUDGE = 6, 18


def plan_for(study, rng):
    """The page's buildPlan: draw, shuffle together, then flip sides at random per trial."""
    judge = rng.sample(study["judge"], N_JUDGE)
    screen = rng.sample(study["screen"], N_SCREEN)
    items = judge + screen
    rng.shuffle(items)
    out = []
    for t in items:
        flip = rng.random() < 0.5
        r = dict(t, flip=flip,
                 l_img=t["right_img"] if flip else t["left_img"],
                 r_img=t["left_img"] if flip else t["right_img"])
        if t["kind"] == "judge":
            r["l_sys"] = t["right"] if flip else t["left"]
            r["r_sys"] = t["left"] if flip else t["right"]
        else:
            a = t["answer"]
            r["answer_side"] = ("left" if a == "right" else "right") if flip else a
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="runs/study_web2/study.json")
    ap.add_argument("--dir", default="runs/study_web2")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--n", type=int, default=10, help="sessions, i.e. stand-in raters")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="runs/ai_raters.txt")
    a = ap.parse_args()
    key = os.environ.get("PIXEL_API_KEY", "")
    if not key:
        sys.exit("set PIXEL_API_KEY (the gateway key; it must not be committed)")
    d = Path(a.dir)
    study = json.loads(Path(a.study).read_text(encoding="utf-8"))

    raters = []
    for i in range(a.n):
        rng = random.Random(a.seed + i)
        rid = "M" + f"{a.seed + i:06d}"[-6:]
        raters.append({"rid": rid, "plan": plan_for(study, rng), "answers": [None] * (N_SCREEN + N_JUDGE)})

    jobs = Queue()
    for r in raters:
        for k, t in enumerate(r["plan"]):
            jobs.put((r, k, t))
    total = jobs.qsize()
    print(f"{a.n} sessions x {N_SCREEN + N_JUDGE} trials = {total} asks, model {a.model}", flush=True)
    lock = threading.Lock()
    done = [0]

    def worker():
        while True:
            try:
                r, k, t = jobs.get_nowait()
            except Exception:
                return
            b64 = pair_image(d / t["l_img"], d / t["r_img"])
            p = (SCREEN_PROMPT.format(question=t["text"]) if t["kind"] == "screen"
                 else JUDGE_PROMPT.format(caption=t.get("prompt", ""), question=t["text"]))
            t0 = time.time()
            try:
                txt = call(a.model, key, p, b64)
            except SystemExit:
                raise
            except Exception as e:
                print(f"  {r['rid']} trial {k}: {str(e)[:70]}", flush=True)
                continue
            ms = int(1000 * (time.time() - t0))
            m = re.search(r"\b(LEFT|RIGHT)\b", txt.upper())
            if not m:
                print(f"  {r['rid']} trial {k}: unparsed {txt[:40]!r}", flush=True)
                continue
            side = m.group(1).lower()
            rec = {"id": t["id"], "kind": t["kind"], "question": t["question"], "side": side, "ms": ms}
            if t["kind"] == "judge":
                rec["chose"] = t["l_sys"] if side == "left" else t["r_sys"]
                rec["over"] = t["r_sys"] if side == "left" else t["l_sys"]
            else:
                rec["correct"] = side == t["answer_side"]
            with lock:
                r["answers"][k] = rec
                done[0] += 1
                if done[0] % 25 == 0:
                    print(f"  {done[0]}/{total}", flush=True)

    ts = [threading.Thread(target=worker) for _ in range(a.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    lines = []
    for r in raters:
        ans = [x for x in r["answers"] if x]
        scr = [x for x in ans if x["kind"] == "screen"]
        payload = {"v": 1, "lang": "model", "model": a.model, "rater": r["rid"],
                   "at": datetime.now(timezone.utc).isoformat(),
                   "screenPassed": sum(1 for x in scr if x["correct"]), "screenTotal": len(scr),
                   "answers": ans}
        lines.append("PXS1:" + base64.b64encode(
            json.dumps(payload, ensure_ascii=False).encode("utf-8")).decode())
        print(f"{r['rid']}  {len(ans)}/{N_SCREEN + N_JUDGE} answered  screening "
              f"{payload['screenPassed']}/{payload['screenTotal']}", flush=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {a.out}\nnow: python src/v6/collect_study.py --in {a.out}", flush=True)


if __name__ == "__main__":
    main()
