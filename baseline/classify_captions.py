"""Classify sprites from their captions alone, so the whole corpus can be labelled cheaply.

classify_sprites.py sends images and costs ~300 tokens per sprite; the training corpus has ~52k sprites. The captions
we already have describe each sprite in 13 words, so a text-only call with 100 captions per request labels the corpus
for a fraction of that. This script is validated against the vision labels on the held-out set before being trusted
(09-18).

Usage:
  python baseline/classify_captions.py --csv data/oga_captions_recap.csv --out runs/capcls_oga.jsonl
  python baseline/classify_captions.py --texts runs/val_captions.txt --out runs/capcls_val.jsonl
"""
import argparse
import csv
import json
import re
import sys
import threading
from pathlib import Path
from queue import Queue

sys.path.insert(0, "baseline")
import recaption as R  # noqa: E402

INSTR = """Below are {k} numbered captions of small game sprites. For EACH caption decide what the sprite is:

- object: a standalone thing - item, character, creature, plant, building, food, tool, prop, vehicle
- effect: fire, smoke, explosion, sparkle, magic, projectile, particle, glow
- fragment: a piece of a larger drawing, a strip, a bar, a border, a frame, a UI element, part of a tile or wall
- glyph: a letter, digit, punctuation mark, logo or written symbol
- junk: empty, noise, or the caption says it is unclear

Reply with exactly {k} lines, each formatted as:
<number>. <class>

Captions:
{body}"""


def call_text(model, prompt, retries=3):
    import json as _j
    import time
    import urllib.request
    body = _j.dumps({"model": model, "max_tokens": 4000,
                     "messages": [{"role": "user", "content": prompt}]}).encode()
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(R.BASE + "/chat/completions", data=body,
                                         headers={"Authorization": "Bearer " + R.KEY,
                                                  "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = _j.load(r)
            return d["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            time.sleep(3 * (a + 1))
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="captions csv with path,text")
    ap.add_argument("--texts", help="plain text file, one caption per line (keys are line numbers)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if not R.KEY:
        sys.exit("set PIXEL_API_KEY")
    if a.csv:
        rows = [(r["path"], r["text"]) for r in csv.DictReader(open(a.csv, encoding="utf-8"))]
    else:
        rows = [(str(i), t.strip()) for i, t in enumerate(open(a.texts, encoding="utf-8")) if t.strip()]
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["path"] for l in open(a.out, encoding="utf-8")}
    todo = [r for r in rows if r[0] not in done]
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    print(f"{len(rows)} captions, {len(todo)} to classify, {len(batches)} calls", flush=True)
    q = Queue()
    for b in batches:
        q.put(b)
    lock = threading.Lock()
    fh = open(a.out, "a", encoding="utf-8")

    def worker():
        while True:
            try:
                batch = q.get_nowait()
            except Exception:
                return
            body = "\n".join(f"{i + 1}. {t}" for i, (_, t) in enumerate(batch))
            try:
                txt = call_text(a.model, INSTR.format(k=len(batch), body=body))
            except Exception as e:
                print(f"call failed: {str(e)[:120]}", flush=True)
                continue
            got = {}
            for line in txt.splitlines():
                m = re.match(r"\s*(\d+)\s*[.):]\s*([a-zA-Z]+)", line.strip())
                if m and 1 <= int(m.group(1)) <= len(batch):
                    got[int(m.group(1)) - 1] = m.group(2).lower()
            with lock:
                for i, (p, _) in enumerate(batch):
                    if i in got:
                        fh.write(json.dumps({"path": p, "class": got[i]}) + "\n")
                fh.flush()
                print(f"{len(got)}/{len(batch)}", flush=True)

    ts = [threading.Thread(target=worker) for _ in range(a.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("CAPCLS_DONE", flush=True)


if __name__ == "__main__":
    main()
