p = "src/v6/sample_e.py"
s = open(p, encoding="utf-8").read()
if "--src_bucket" in s:
    print("already patched"); raise SystemExit
a = '    lab = torch.full((n,), BUCKETS.index(size), device=device, dtype=torch.long)'
assert s.count(a) == 1, s.count(a)
s = s.replace(a, '''    lab = torch.full((n,), BUCKETS.index(size), device=device, dtype=torch.long)
    if src_bucket is not None:  # probe_src: one embedding per (target, source) pair, label = target*7 + source
        lab = lab * len(BUCKETS) + BUCKETS.index(int(src_bucket))''')
# thread the argument through sample()
b = "def sample(model, scheduler, cond, uncond, size, device, steps=100, cfg=4.0, seed=0, guide=None,"
assert s.count(b) == 1
s = s.replace(b, b.replace("guide=None,", "guide=None, src_bucket=None,"))
c = '    p.add_argument("--steps", type=int, default=None, help="DDPM steps (default 100; es models default to their few-step count)")'
assert s.count(c) == 1
s = s.replace(c, c + '\n    p.add_argument("--src_bucket", type=int, default=None,\n'
                     '                   help="probe_src only: which SOURCE bucket the (target,source) class label should encode")')
# pass it at the call site
import re
m = re.search(r"args\.steps, args\.cfg, args\.seed \+ i // args\.chunk, guide", s)
assert m, "call site not found"
s = s[:m.end()] + ", src_bucket=args.src_bucket" + s[m.end():]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched sample_e with --src_bucket")
