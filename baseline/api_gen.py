"""Generate the external "big model then downscale" baselines through the user's OpenAI-compatible API.

The comparison a reviewer will demand first is "why not just prompt a strong text-to-image model and downscale?".
Nothing in this project had ever tested it.  Images come back at 1024, get a plain-background cutout, and then go
through the project's own to_tensor(R) path so the FD is computed on identical footing to our own samples.

Key is read from the PIXEL_API_KEY environment variable, never hard-coded.
Usage: PIXEL_API_KEY=... python baseline/api_gen.py --model gpt-image-2 --n 200 --out runs/api_gpt_image_2
"""
import argparse, base64, io, json, os, sys, threading, time, urllib.request
from pathlib import Path
from queue import Queue

import numpy as np
from PIL import Image

BASE = os.environ.get("PIXEL_API_BASE", "http://113.45.39.247:3001/v1")
KEY = os.environ.get("PIXEL_API_KEY", "")
SUFFIX = ", pixel art sprite of a single object, plain flat white background, centred, no text"


def cutout(img, tol=18):
    """Flood the border-connected near-background colour to alpha 0, then crop to content."""
    from collections import deque
    rgb = np.asarray(img.convert("RGB"))
    a = rgb.astype(np.int16)
    h, w, _ = a.shape
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    close = np.abs(a - bg).max(axis=2) <= tol
    seen = np.zeros((h, w), bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if close[y, x] and not seen[y, x]:
                seen[y, x] = True; q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if close[y, x] and not seen[y, x]:
                seen[y, x] = True; q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and close[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True; q.append((ny, nx))
    alpha = (~seen).astype(np.uint8) * 255
    out = np.dstack([rgb, alpha])
    ys, xs = np.where(alpha > 0)
    if len(ys) == 0:
        return Image.fromarray(out, "RGBA")
    return Image.fromarray(out[ys.min():ys.max() + 1, xs.min():xs.max() + 1], "RGBA")


def call(model, prompt, size, retries=3):
    body = json.dumps({"model": model, "prompt": prompt + SUFFIX, "n": 1, "size": size}).encode()
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(BASE + "/images/generations", data=body,
                                         headers={"Authorization": "Bearer " + KEY,
                                                  "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            it = d["data"][0]
            if it.get("b64_json"):
                return Image.open(io.BytesIO(base64.b64decode(it["b64_json"])))
            with urllib.request.urlopen(it["url"], timeout=180) as r:
                return Image.open(io.BytesIO(r.read()))
        except Exception as e:
            last = e
            time.sleep(3 * (k + 1))
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="runs/heldout3000_prompts.txt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--size", type=int, default=16, help="target sprite side")
    ap.add_argument("--gen_size", default="1024x1024")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not KEY:
        sys.exit("set PIXEL_API_KEY")

    prompts = [l.strip() for l in open(args.prompts, encoding="utf-8") if l.strip()][:args.n]
    big = Path(args.out) / "big"; small = Path(args.out) / f"s{args.size}"
    big.mkdir(parents=True, exist_ok=True); small.mkdir(parents=True, exist_ok=True)

    todo = [(i, p) for i, p in enumerate(prompts) if not (big / f"{i:05d}.png").exists()]
    print(f"{len(prompts)} prompts, {len(todo)} to generate, {args.workers} workers", flush=True)
    q = Queue()
    for t in todo:
        q.put(t)
    done = [0]; lock = threading.Lock()

    def worker():
        while True:
            try:
                i, p = q.get_nowait()
            except Exception:
                return
            try:
                im = call(args.model, p, args.gen_size)
                im.save(big / f"{i:05d}.png")
            except Exception as e:
                with lock:
                    print(f"  [{i}] FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
            finally:
                with lock:
                    done[0] += 1
                    if done[0] % 10 == 0:
                        print(f"  [{done[0]}/{len(todo)}]", flush=True)
                q.task_done()

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    print(f"generated {len(list(big.glob('*.png')))} images", flush=True)
    print("API_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
