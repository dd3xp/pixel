p = "src/v6/train_vpred.py"
s = open(p, encoding="utf-8").read()
def sub(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)
# train with v-prediction instead of eps-prediction; everything else identical to v7h
sub('scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")\n    opt =',
    'scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="v_prediction")\n    opt =')
sub("""        pred = model(scheduler.add_noise(x, noise, t), t, encoder_hidden_states=cond, class_labels=b).sample
        loss = F.mse_loss(pred, noise)""",
    """        pred = model(scheduler.add_noise(x, noise, t), t, encoder_hidden_states=cond, class_labels=b).sample
        loss = F.mse_loss(pred, scheduler.get_velocity(x, noise, t))""")
# the in-training preview sampler builds its own scheduler; keep it consistent
s = s.replace('DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")',
              'DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="v_prediction")')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("v_prediction occurrences:", s.count("v_prediction"))
