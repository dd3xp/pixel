"""Rate sprite craftsmanship 1-5 with a vision model, so quality can be filtered or conditioned on.

09-18: the model matches the corpus distribution but the corpus is community art of very uneven quality, so the
average sample looks like the average asset rather than like a good sprite. Before spending a training run on
quality conditioning we check that a cheap rating actually separates good sprites from bad ones.

The prompt asks about craft, not about subject matter: readable silhouette, deliberate palette, clean edges, no
stray pixels, looks finished.

Usage: PIXEL_API_KEY=... python baseline/score_quality.py --imgs runs/gt_all/heldout_native16_gt --out runs/quality.jsonl
"""
import argparse
import json
import re
import sys
import threading
from pathlib import Path
from queue import Queue

sys.path.insert(0, "baseline")
import recaption as R  # noqa: E402

INSTR = """These are {k} pixel-art sprites from a game-art corpus, in a numbered grid on a checkerboard background
(the checkerboard is transparency, not part of the art).

Rate the CRAFT of each sprite from 1 to 5. Judge only how well it is drawn, never what it depicts:
5 = clearly deliberate: readable silhouette, purposeful limited palette, clean shading, no stray pixels, finished
4 = solid, minor flaws
3 = plain or generic but coherent
2 = sloppy: muddy colours, unclear shape, unfinished
1 = unusable: noise, a fragment, almost empty

Reply with exactly {k} lines, each formatted as:
<number>. <score>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not R.KEY:
        sys.exit("set PIXEL_API_KEY")
    R.INSTR = INSTR
    paths = sorted(Path(a.imgs).glob("*.png"))
    if a.limit:
        paths = paths[:a.limit]
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["path"] for l in open(a.out, encoding="utf-8")}
    todo = [p for p in paths if p.name not in done]
    batches = [todo[i:i + a.grid] for i in range(0, len(todo), a.grid)]
    print(f"{len(paths)} sprites, {len(todo)} to rate, {len(batches)} calls", flush=True)
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
                m = re.match(r"\s*(\d+)\s*[.):]\s*([1-5])", line.strip())
                if m and 1 <= int(m.group(1)) <= len(batch):
                    got[int(m.group(1)) - 1] = int(m.group(2))
            with lock:
                for i, p in enumerate(batch):
                    if i in got:
                        fh.write(json.dumps({"path": p.name, "score": got[i]}) + "\n")
                fh.flush()
                print(f"{len(got)}/{len(batch)} rated", flush=True)

    ts = [threading.Thread(target=worker) for _ in range(a.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("QUALITY_DONE", flush=True)


if __name__ == "__main__":
    main()
