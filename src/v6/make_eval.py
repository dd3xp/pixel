"""Build a self-contained blind evaluation page for human raters.

Why this is needed now
----------------------
Every number in this project is either a CLIP score or my own subjective read,
and neither can stand in a paper.  CLIP turned out to be actively misleading
here: it ranked the distilled model WORSE on all 100 objects while the sprites
went from unrenderable to clearly recognisable.  My own scores are consistent
but single-rater and unblinded -- I always know which column is ours.

So the page below removes the two things that make our own numbers unusable:
  * identity  -- methods are shuffled per question and labelled A/B/C, and the
                 key stays out of the page until the rater exports
  * order     -- question order is shuffled per rater from a seed, so fatigue
                 does not systematically fall on one method
Two axes are scored separately because they can move in opposite directions,
which is exactly what the distillation experiment did: an object can become
recognisable while the palette gets washed out.

Everything is inlined as base64, so a rater just opens the file -- no server, no
install.  Answers persist in localStorage against accidental refresh, and the
export is a plain textarea to copy, because a downloaded file is easy to lose.

Usage: python src/v6/make_eval.py --out runs_out/human_eval.html
"""
import argparse
import base64
import json
from pathlib import Path

ROOT = Path("/mnt/data/kw/RoundSquisheen/pixel/pixel")
PROMPTS = [l.strip() for l in open(ROOT / "baseline/prompts8.txt", encoding="utf-8")
           if l.strip()]
# method -> path template; kept out of the page except in the export key
METHODS = {
    "ours": "runs_out/v7c_best8/s{size}/{i:02d}.png",
    "sdpixl": "baseline/results/10k_s{size}_p{i1}.png",
    "downscale": "runs_out/dsbaseline3/s{size}/mean_raw/{i:02d}.png",
}
SIZES = [12, 16, 20, 24]


def b64(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs_out/human_eval.html")
    args = ap.parse_args()

    items = []
    for size in SIZES:
        for i, prompt in enumerate(PROMPTS):
            shown = []
            for m, tpl in METHODS.items():
                p = ROOT / tpl.format(size=size, i=i, i1=i + 1)
                if p.exists():
                    shown.append({"m": m, "src": b64(p)})
            if len(shown) >= 2:            # a lone method is not a comparison
                items.append({"size": size, "prompt": prompt, "opts": shown})

    html = """<title>Pixel sprite evaluation</title>
<style>
 :root{--bg:#fff;--fg:#111;--line:#d8d8d8;--card:#fafafa}
 @media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#16181c;--fg:#e8e8e8;--line:#333;--card:#1e2126}}
 :root[data-theme=dark]{--bg:#16181c;--fg:#e8e8e8;--line:#333;--card:#1e2126}
 body{background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif;margin:0;padding:24px}
 .wrap{max-width:900px;margin:0 auto}
 .q{border:1px solid var(--line);border-radius:10px;padding:16px;margin:18px 0;background:var(--card)}
 .head{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:12px}
 .opts{display:flex;gap:18px;flex-wrap:wrap}
 .opt{flex:1 1 200px;min-width:180px;text-align:center}
 img{width:160px;height:160px;image-rendering:pixelated;background:#fff;border:1px solid var(--line);border-radius:6px}
 .row{display:flex;gap:6px;justify-content:center;margin-top:8px;flex-wrap:wrap}
 .row b{width:100%;font-weight:500;font-size:13px;opacity:.75}
 label{cursor:pointer;padding:3px 7px;border:1px solid var(--line);border-radius:5px;font-size:13px}
 input{position:absolute;opacity:0}
 input:checked+span{font-weight:700;text-decoration:underline}
 textarea{width:100%;height:220px;font-family:ui-monospace,monospace;font-size:12px}
 button{padding:9px 18px;font-size:15px;border-radius:7px;border:1px solid var(--line);
        background:var(--card);color:var(--fg);cursor:pointer}
 .bar{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);z-index:5}
</style>
<div class="wrap">
<h2>像素精灵盲评</h2>
<p>每题给同一个提示词的几张 16 色以内低分辨率精灵。<b>方法已随机打乱并匿名为 A/B/C</b>,请独立打分。两个维度分开评:</p>
<ul>
<li><b>可辨性</b>:不看提示词的话,能认出这是那个东西吗?(1=完全认不出,5=一眼认出)</li>
<li><b>像素画质量</b>:轮廓是否干净、配色是否像手绘像素画、有无噪点或糊边。(1=很差,5=很好)</li>
</ul>
<p>可辨性和质量可能相反(画得漂亮但认不出,或认得出但很脏),请分别独立评分。</p>
<div class="bar"><span id="prog"></span> &nbsp; <button onclick="dump()">完成 / 导出结果</button></div>
<div id="qs"></div>
<h3>导出</h3><textarea id="out" placeholder="点上面的导出按钮"></textarea>
</div>
<script>
const ITEMS = __ITEMS__;
const KEY = 'pixeleval_v1';
let ans = {};
try { ans = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { ans = {}; }

// per-rater shuffle so fatigue does not always land on the same method
let seed = 0;
try { seed = +(localStorage.getItem(KEY + '_seed') || 0); } catch (e) {}
if (!seed) { seed = Math.floor(Math.random() * 1e9) || 1;
  try { localStorage.setItem(KEY + '_seed', seed); } catch (e) {} }
function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
function shuffle(a) { a = a.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]]; } return a; }

const order = shuffle(ITEMS.map((_, i) => i));
const view = order.map(i => ({ i, opts: shuffle(ITEMS[i].opts) }));

function save() { try { localStorage.setItem(KEY, JSON.stringify(ans)); } catch (e) {}
  const need = view.reduce((n, v) => n + v.opts.length * 2, 0);
  document.getElementById('prog').textContent =
    `已答 ${Object.keys(ans).length} / ${need}`; }

const qs = document.getElementById('qs');
view.forEach((v, qi) => {
  const it = ITEMS[v.i];
  const d = document.createElement('div'); d.className = 'q';
  let h = `<div class="head"><b>${qi + 1}. ${it.prompt}</b><span>${it.size}×${it.size}</span></div><div class="opts">`;
  v.opts.forEach((o, oi) => {
    const tag = String.fromCharCode(65 + oi);
    const id = `${v.i}_${o.m}`;
    h += `<div class="opt"><img src="${o.src}" alt="${tag}"><div><b>${tag}</b></div>`;
    ['rec', 'qual'].forEach(ax => {
      h += `<div class="row"><b>${ax === 'rec' ? '可辨性' : '像素画质量'}</b>`;
      for (let s = 1; s <= 5; s++) {
        const k = `${id}_${ax}`;
        h += `<label><input type="radio" name="${k}" value="${s}"${ans[k] == s ? ' checked' : ''}
               onchange="ans['${k}']=${s};save()"><span>${s}</span></label>`;
      }
      h += `</div>`;
    });
    h += `</div>`;
  });
  d.innerHTML = h + '</div>';
  qs.appendChild(d);
});
save();

function dump() {
  const rows = [['item', 'size', 'prompt', 'method', 'axis', 'score']];
  view.forEach(v => { const it = ITEMS[v.i];
    v.opts.forEach(o => ['rec', 'qual'].forEach(ax => {
      const k = `${v.i}_${o.m}_${ax}`;
      if (ans[k]) rows.push([v.i, it.size, it.prompt, o.m, ax, ans[k]]);
    })); });
  document.getElementById('out').value =
    rows.map(r => r.join(',')).join('\\n');
}
</script>"""
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("__ITEMS__", json.dumps(items)), encoding="utf-8")
    mb = out.stat().st_size / 2**20
    print(f"{len(items)} questions ({sum(len(i['opts']) for i in items)} sprites) "
          f"-> {args.out}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
