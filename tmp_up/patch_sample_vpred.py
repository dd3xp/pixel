p = "src/v6/sample_e.py"
s = open(p, encoding="utf-8").read()
if '"--pred"' in s:
    print("already patched"); raise SystemExit
a = '        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")'
assert s.count(a) == 1, s.count(a)
s = s.replace(a, '        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2",\n'
                 '                                  prediction_type=args.pred)')
b = '    p.add_argument("--steps", type=int, default=None, help="DDPM steps (default 100; es models default to their few-step count)")'
assert s.count(b) == 1, s.count(b)
s = s.replace(b, b + '\n    p.add_argument("--pred", default="epsilon", choices=["epsilon", "v_prediction", "sample"],\n'
                     '                   help="must match how the checkpoint was trained (probe_vpred uses v_prediction)")')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("sample_e patched")
