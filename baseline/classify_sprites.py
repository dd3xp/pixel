"""Label each sprite as a standalone object or as junk, with the same grid trick as recaption.py.

Why (09-18): the corpus is built by cutting sprite sheets on alpha connected components, so besides real sprites it
contains glyphs ('h'), UI strips, tile fragments, one-pixel specks and pieces of larger drawings. Those end up both in
the training targets and in the FD reference, which is why held-out "ground truth" sometimes looks like nothing at all.
The colour/contrast filters in clean_cut.py do not catch them; a vision model does.

Classes: object (a standalone thing: item, character, creature, tile-sized prop), effect (fire, smoke, sparkle,
projectile), fragment (part of a bigger drawing, a strip, a border, a UI element), glyph (letter, digit, symbol, logo),
junk (empty, noise, unreadable).

PIXEL_API_KEY must be set. Usage:
  python baseline/classify_sprites.py --imgs runs_out/heldout_native16_gt --out runs/sprite_classes.jsonl
"""
import argparse
import json
import re
import sys
import threading
from pathlib import Path
from queue import Queue

sys.path.insert(0, "baseline")
import recaption as R  # noqa: E402  (call/sheet/cell are reused; the prompt is replaced below)

INSTR = """These are {k} sprites from a game-art corpus, laid out in a numbered grid on a checkerboard background
(the checkerboard is transparency, not part of the art).

Classify EACH numbered cell into exactly one class:
- object: a standalone thing - an item, character, creature, plant, building, food, tool, prop
- effect: fire, smoke, explosion, sparkle, magic, projectile, particle
- fragment: a piece of a larger drawing, a strip, a bar, a border, a frame, a UI element, part of a tile
- glyph: a letter, digit, punctuation mark, logo or written symbol
- junk: empty, near-empty, noise, or impossible to read as anything

Reply with exactly {k} lines, each formatted as:
<number>. <class>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if not R.KEY:
        sys.exit("set PIXEL_API_KEY")
    R.INSTR = INSTR          # R.call formats the module-level prompt, so swap it for the classifier one
    paths = sorted(Path(a.imgs).glob("*.png"))
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["path"] for l in open(a.out, encoding="utf-8")}
    todo = [p for p in paths if p.name not in done]
    batches = [todo[i:i + a.grid] for i in range(0, len(todo), a.grid)]
    print(f"{len(paths)} sprites, {len(todo)} to classify, {len(batches)} calls", flush=True)
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
            try:
                txt, _ = R.call(a.model, R.sheet(batch, size=150, cols=5), len(batch))
            except Exception as e:
                print(f"call failed: {str(e)[:120]}", flush=True)
                continue
            got = {}
            for line in txt.splitlines():
                m = re.match(r"\s*(\d+)\s*[.):]\s*([a-zA-Z]+)", line.strip())
                if m and 1 <= int(m.group(1)) <= len(batch):
                    got[int(m.group(1)) - 1] = m.group(2).lower()
            with lock:
                for i, p in enumerate(batch):
                    if i in got:
                        fh.write(json.dumps({"path": p.name, "class": got[i]}) + "\n")
                fh.flush()
                print(f"{len(got)}/{len(batch)} classified", flush=True)

    ts = [threading.Thread(target=worker) for _ in range(a.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("CLASSIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
