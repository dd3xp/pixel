"""Run the forced-choice study with a vision--language model as the judge.

This is not the human study and does not replace it. The human study is blocked on a platform decision;
meanwhile the central claim -- that our samples are closer to real low-resolution pixel art than a large
generator's downscaled output -- still rests only on a reference set we defined, and a judge outside our
metric is worth having before reviewers supply one. A VLM is a weak but independent instrument: it has
never seen our reference set and has no stake in FD.

The trials are the ones build_human_study.py writes: pairs drawn from real / ours / flux / gpt / sdxl,
each asked twice, once for fidelity to the medium and once for preference. Position is already
randomised per trial; the model sees the two sprites side by side, magnified, with the caption.

Usage: PIXEL_API_KEY=... python baseline/vlm_pref.py --study runs_out/human_study --out runs/vlm_pref.jsonl
"""
import argparse
import base64
import io
import json
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path
from queue import Queue

from PIL import Image

BASE = "http://113.45.39.247:3001/v1"

PROMPT = """These are two 16x16 pixel-art game sprites, shown magnified side by side on a checkerboard
background (the checkerboard is transparency, not part of the art). Both were made for this description:

"{caption}"

{question}

Answer with exactly one word: LEFT or RIGHT."""
# max_tokens has to be generous: a reasoning model spends tokens before it emits the word, and
# a tight budget comes back as an empty completion rather than a truncated one (09-22).


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
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            time.sleep(3 * (a + 1))
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="runs_out/human_study")
    ap.add_argument("--out", default="runs/vlm_pref.jsonl")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import os
    key = os.environ.get("PIXEL_API_KEY", "")
    if not key:
        sys.exit("set PIXEL_API_KEY")
    study = Path(a.study)
    trials = json.loads((study / "trials.json").read_text(encoding="utf-8"))["trials"]
    if a.limit:
        trials = trials[:a.limit]
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")}
    todo = [t for t in trials if f"{t['index']}|{t['question']}|{t['left']}|{t['right']}" not in done]
    print(f"{len(trials)} trials, {len(todo)} to judge", flush=True)

    q = Queue()
    for t in todo:
        q.put(t)
    lock = threading.Lock()
    fh = open(a.out, "a", encoding="utf-8")

    def worker():
        while True:
            try:
                t = q.get_nowait()
            except Exception:
                return
            tid = f"{t['index']}|{t['question']}|{t['left']}|{t['right']}"
            try:
                b64 = pair_image(study / t["left_img"], study / t["right_img"])
                txt = call(a.model, key, PROMPT.format(caption=t["prompt"], question=t["text"]), b64)
            except Exception as e:
                print(f"failed: {str(e)[:90]}", flush=True)
                continue
            m = re.search(r"\b(LEFT|RIGHT)\b", txt.upper())
            if not m:
                print(f"unparsed: {txt[:40]!r}", flush=True)
                continue
            pick = t["left"] if m.group(1) == "LEFT" else t["right"]
            other = t["right"] if m.group(1) == "LEFT" else t["left"]
            with lock:
                fh.write(json.dumps({"id": tid, "question": t["question"], "winner": pick,
                                     "loser": other, "index": t["index"]}) + "\n")
                fh.flush()

    ts = [threading.Thread(target=worker) for _ in range(a.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("VLM_PREF_DONE", flush=True)


if __name__ == "__main__":
    main()
