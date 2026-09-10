"""Re-caption the sprite corpus with a vision model, because the BLIP captions it was trained on are unusable:
15.4% of them are literally "a pixel art sprite" and only 23.5% are distinct, so the model never learned to read text.

Sprites are 16-64 px, so sending them one per call wastes almost all the image-token budget on overhead.  Instead
GRID sprites into one numbered sheet and ask for one caption per cell; that cuts both calls and tokens by ~GRID x.
Each sprite is nearest-upscaled and put on a light checkerboard so transparency is visible to the model.

PIXEL_API_KEY must be in the environment. Usage:
  PIXEL_API_KEY=... python baseline/recaption.py --imgs runs/recap_imgs --n 500 --model gemini-3-flash-preview
"""
import argparse, base64, io, json, os, re, sys, threading, time, urllib.request
from pathlib import Path
from queue import Queue

import numpy as np
from PIL import Image, ImageDraw

BASE = os.environ.get("PIXEL_API_BASE", "http://113.45.39.247:3001/v1")
KEY = os.environ.get("PIXEL_API_KEY", "")

INSTR = """These are {k} pixel-art game sprites, laid out in a numbered grid on a checkerboard background
(the checkerboard is transparency, not part of the art).

For EACH numbered cell write one caption describing what the sprite depicts, for someone who must draw it
from the caption alone. Rules:
- Name the object first, then its distinguishing colours, materials and parts.
- 6 to 15 words. No "pixel art", no "sprite", no "8-bit", no "icon" - those words carry no information here.
- Describe only what is visible. If a cell is genuinely unrecognisable, write exactly: unclear abstract shape.
- Do not mention the checkerboard or the background.

Reply with exactly {k} lines, each formatted as:
<number>. <caption>"""


def cell(img, size, pad=6):
    """Nearest-upscale a sprite onto a checkerboard tile."""
    a = np.indices((size, size)).sum(0) // 8 % 2
    bg = Image.fromarray((225 + 20 * a).astype(np.uint8)).convert("RGBA")
    s = max(img.size)
    f = max(1, (size - 2 * pad) // s)
    im = img.resize((img.width * f, img.height * f), Image.NEAREST)
    bg.alpha_composite(im, ((size - im.width) // 2, (size - im.height) // 2))
    return bg


def sheet(paths, size=150, cols=5):
    rows = (len(paths) + cols - 1) // cols
    hdr = 22
    W, H = cols * size, rows * (size + hdr)
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for i, p in enumerate(paths):
        r, c = divmod(i, cols)
        x, y = c * size, r * (size + hdr)
        d.text((x + 4, y + 4), f"{i + 1}.", fill=(0, 0, 0, 255))
        canvas.paste(cell(Image.open(p).convert("RGBA"), size), (x, y + hdr))
    return canvas.convert("RGB")


def call(model, img, k, retries=3):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = json.dumps({"model": model, "max_tokens": 120 * k + 3000, "messages": [{"role": "user", "content": [
        {"type": "text", "text": INSTR.format(k=k)},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}]}).encode()
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                         headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            return d["choices"][0]["message"]["content"], d.get("usage", {})
        except Exception as e:
            last = e; time.sleep(3 * (a + 1))
    raise last


def parse(text, k):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s*[.):]\s*(.+)", line.strip())
        if m:
            i = int(m.group(1))
            if 1 <= i <= k:
                out[i - 1] = m.group(2).strip().rstrip(".")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--grid", type=int, default=25)
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="runs/recaptions.jsonl")
    a = ap.parse_args()
    if not KEY:
        sys.exit("set PIXEL_API_KEY")

    paths = sorted(Path(a.imgs).glob("*.png"))[:a.n]
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["path"] for l in open(a.out, encoding="utf-8")}
    todo = [p for p in paths if p.name not in done]
    batches = [todo[i:i + a.grid] for i in range(0, len(todo), a.grid)]
    print(f"{len(paths)} sprites, {len(todo)} to caption, {len(batches)} grid calls of <={a.grid}", flush=True)

    lock = threading.Lock(); q = Queue(); tok = [0, 0]
    for b in batches: q.put(b)
    fh = open(a.out, "a", encoding="utf-8")

    def worker():
        while True:
            try:
                batch = q.get_nowait()
            except Exception:
                return
            try:
                txt, usage = call(a.model, sheet([str(p) for p in batch]), len(batch))
                caps = parse(txt, len(batch))
                with lock:
                    for i, p in enumerate(batch):
                        if i in caps:
                            fh.write(json.dumps({"path": p.name, "text": caps[i]}, ensure_ascii=False) + "\n")
                    fh.flush()
                    tok[0] += usage.get("prompt_tokens", 0); tok[1] += usage.get("completion_tokens", 0)
                    print(f"  grid done: {len(caps)}/{len(batch)} parsed | tokens so far in={tok[0]} out={tok[1]}", flush=True)
            except Exception as e:
                with lock:
                    print(f"  GRID FAILED {type(e).__name__}: {str(e)[:90]}", flush=True)
            finally:
                q.task_done()

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    [t.start() for t in ts]; [t.join() for t in ts]
    n = sum(1 for _ in open(a.out, encoding="utf-8"))
    print(f"total captions: {n}")
    print(f"TOKENS  in={tok[0]}  out={tok[1]}  total={tok[0]+tok[1]}")
    if batches:
        print(f"per sprite: {(tok[0]+tok[1])/max(1,len(todo)):.1f} tokens")
    print("RECAPTION_DONE")


if __name__ == "__main__":
    main()
