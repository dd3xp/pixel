# 3 Method

All numbers are taken from `paper_outline.md` (Tables a, a′, a″, b, d, e, i) and the experiment log; values the outline marks TBD are flagged.

## 3.1 Problem setup

We generate RGBA sprites at a target side length $R \in \{12, 16, 20, 24, 32\}$ px from a caption $c$. The generator is a standard conditional diffusion model and is not a contribution: a 4-channel RGBA UNet (`UNet2DConditionModel`, 72.5 M parameters [TBD: outline asks to verify the exact count; 72.5 M is the value in the timing log]) with cross-attention to a frozen CLIP text encoder [TODO: cite CLIP], and a *resolution bucket* $b \in \mathcal{B} = \{12, 16, 20, 24, 32, 48, 64\}$ supplied as a class embedding. It is trained with $\epsilon$-prediction [TODO: cite Ho et al. 2020], sampled with 100 DDPM steps and evaluated with EMA weights. Training data are OGA-derived sprites with BLIP captions [TODO: cite BLIP]; each sprite is also fed, BOX-downsampled, to every lower bucket, so all buckets see the same sprite population. The main model (v7h) is trained from random initialisation for 80 k steps on 180,533 rows with the 5,928 evaluation sprites excluded; EMA snapshots are saved every 5 k steps. A second model (v7s; width 96, 41.3 M parameters, 60 k steps) shares recipe and exclusion list.

Write $\epsilon_\theta(x_t, t, c, b)$ for the noise prediction at noisy input $x_t$, timestep $t$, caption $c$ and bucket $b$. The baseline is classifier-free guidance (CFG) [TODO: cite Ho & Salimans 2022] with the empty caption $\varnothing$ and $w_{\text{cfg}} = 4$:

$$
e_{\text{cfg}} = \epsilon_\theta(x_t,t,\varnothing,b_R) + w_{\text{cfg}}\big[\epsilon_\theta(x_t,t,c,b_R) - \epsilon_\theta(x_t,t,\varnothing,b_R)\big].
$$

On the clean 16 px baseline this gives FD-DINOv2 $21.52 \pm 0.47$ (3 seeds) against a real-data floor of $3.45$, and every stronger CFG setting is worse (CFG $w=7$: 42.95; $w=10$: 62.12; CADS and interval variants 32.57–88.61; Table a″). The gap is not a conditioning deficit; it is a systematic model error, the regime that autoguidance [Karras et al. 2024, arXiv 2406.02507] targets by extrapolating away from a *worse but compatible* version of the same model.

## 3.2 Cross-resolution self-guidance

Every guidance rule we consider has the form: a strong prediction $e_{\text{strong}} = \epsilon_\theta(x_t,t,c,b_R)$, a weak reference $e_{\text{ref}}$ on the same $x_t, t$, and

$$
\boxed{\; e = e_{\text{ref}} + w\,\big(e_{\text{strong}} - e_{\text{ref}}\big) \;} \qquad \text{(2 network evaluations per step).}
$$

CFG is $e_{\text{ref}} = \epsilon_\theta(x_t,t,\varnothing,b_R)$; autoguidance is $e_{\text{ref}} = \epsilon_{\theta_{\text{snap}}}(x_t,t,c,b_R)$ with $\theta_{\text{snap}}$ an early EMA snapshot of the same run (step 10 k, $w = 1.5$).

Our observation is that a bucketed model already contains a compatible weak model without any second set of weights: the same network, on the same $x_t$ and caption, queried under a **lower** bucket label $b_{\text{low}} < b_R$,

$$
e_{\text{ref}} = \epsilon_\theta(x_t,t,c,b_{\text{low}}), \qquad w = 2 .
$$

The rule replaces CFG — no unconditional forward is computed — so its cost is that of CFG. At 16 px with $b_{\text{low}} = 12$ it gives $8.52 \pm 0.29$ (3 seeds), within seed noise of snapshot autoguidance ($8.73 \pm 0.26$), down from $21.52 \pm 0.47$, with no snapshot stored (Table a). We call $\epsilon_\theta(\cdot, b_{\text{low}})$ the model's *lower-resolution belief*: its prediction of the same sprite for the same caption in the lower bucket, rendered on the target grid. Sampled alone it is a clearly worse model (FD 36.43 vs 21.98 at 16 px, seed 0) — autoguidance's "bad model" premise — and it is structure-aligned with the strong prediction by construction, since only the label differs.

**Choice of $b_{\text{low}}$.** The nearest lower bucket is the default (12 for 16 px, 16 for 20 px, 24 for 32 px). At 20 px the nearest bucket beats the farther one (32.75 ± 0.60 over three seeds vs 35.85, seed 0, for bucket 16 vs 12); at 32 px the farther bucket is clearly worse (91.26 vs 77.45, seed 0). The exception is 24 px, where the nearest lower bucket (20) is the worst choice (66.20, seed 0) and we use bucket:16 (59.32 ± 1.37 over three seeds; 60.41 for seed 0), while bucket:12 (57.70, seed 0) is marginally better still; this choice was made on the reported FD. At 12 px no lower bucket exists and the method does not apply; only snapshot autoguidance is available (Table b). We state this as a limitation.

## 3.3 Composed reference

The two weakening axes — earlier weights and a lower label — can be applied in a single forward pass. The composed reference evaluates the snapshot under the lower label:

$$
\boxed{\; e^{\text{comp}}_{\text{ref}} = \epsilon_{\theta_{\text{snap}}}(x_t,t,c,b_{\text{low}}), \qquad e = e^{\text{comp}}_{\text{ref}} + w\,\big(\epsilon_\theta(x_t,t,c,b_R) - e^{\text{comp}}_{\text{ref}}\big), \quad w = 1.5 . \;}
$$

This is one weak reference, not a sum of two guidance terms; the sampler still performs two evaluations per step. At 16 px it reaches $7.53 \pm 0.19$, below every seed of either single reference, and the ordering bare $>$ label reference $\approx$ autoguidance $>$ composed holds at every resolution with a lower bucket (16/20/24/32 px, Table b) and on the second model (Table c).

The two references are complementary (Table a′, seed 0). Splitting FD into mean and covariance terms, the label reference has the lowest mean term (2.55 vs 3.88 for autoguidance and 12.13 for CFG) but weaker covariance term and coverage (6.04, .892); the snapshot has the best covariance term and coverage (5.09, .928); the composed reference keeps the label's mean term (2.40) and most of the snapshot's covariance gain (5.27, between 5.09 and 6.04; coverage .909). The snapshot matters only in the low-noise half: using its weights only for $t/T \le 0.5$ ("snaplo"; outside the interval the reference is the final weights under $b_{\text{low}}$) gives $7.63 \pm 0.43$, indistinguishable from the full rule, while $[0.5, 1]$ gives 8.99, close to the label reference alone (Table e). Snaplo is a diagnostic, not a saving: evaluations per step are unchanged; only the number of steps needing the second weight set halves.

## 3.4 Relation to existing guidance rules

*Autoguidance* [Karras et al. 2024] takes $e_{\text{ref}}$ from a smaller or less-trained network and requires the degradation to be *compatible* with the strong model's error; its experiments vary capacity and training time, not the condition. Our reference is the limiting case of compatibility (identical weights and input) with the degradation carried by the label alone; §3.3 combines the two axes. *SDXL micro-conditioning* [Podell et al. 2023, arXiv 2307.01952] is the closest precedent: the `negative_original_size` practice feeds a small `original_size` to the negative branch of CFG and is documented to induce "simpler patterns" [TODO: cite the diffusers documentation]. It differs in three respects: it is the negative branch of CFG (the caption is dropped too), it encodes source-image quality rather than the target bucket, and no quantitative or directional analysis exists. *ICG/TSG* [arXiv 2407.02687] use a random condition or perturbed timestep embedding; ours is not interchangeable with a random label — a *higher* bucket hurts at every resolution (Table f), and mixing the wrong label with the unconditional branch ("bucketmix", 9.42) lands between the label reference (8.59) and CFG (21.98; all seed 0). Weakening the *input* rather than the belief fails here: a 2×2 box-blurred $x_t$ as reference gives 223.07 ($w=2$), a 1-px cyclic shift 18.95 (Table e). Trained low-frequency reference branches, the 16 px analogue of intermediate-layer weak heads [TODO: cite SGG-BR 2603.20584, IG 2512.24176, SSG 2607.29122], are analysed in §5.5.

## 3.5 Cost

Table i (1000 samples at 16 px, A100-80GB, 100 DDPM steps, batch 500, model load included): no guidance 42.5 s / 4951 MiB; CFG $w=4$ 73.8 s / 4951 MiB; label reference 72.7 s / 4951 MiB; autoguidance 73.3 s / 5507 MiB; composed 74.0 s / 5507 MiB. The label reference costs exactly what CFG costs (same weights, same memory, 1.5 % faster wall-clock); snapshot-based references hold one extra weight copy (+556 MiB) at the same wall-clock. The 3-NFE alignment variant of Appendix Table g″ costs 105.9 s (+43 %).

## 3.6 What makes a reference effective: an operational rule

Section 5 supports a selection rule that needs no training and no second network, only statistics of the reference sampled on its own (`--cfg 0`). Let $\mathrm{TV}$ be the mean adjacent $|\Delta\mathrm{RGB}|$ over opaque pixels of the pure-reference samples, compared with the strong model's samples at the same resolution.

1. **Lower TV than the strong model is necessary.** At 16 px (strong TV 32.0; all statistics in this subsection are seed 0) the effective references have TV 21.6 (bucket:12) and 27.3 (snapshot); the ineffective ones 31.9 (bucket:24, guided FD 17.27) and 40.9 (bucket:64, 19.34). Likewise at 20 px (strong 31.0; bucket:16 21.8 effective; bucket:24 28.5 ineffective) and 24 px (strong 30.3; bucket:16 18.2 effective; bucket:32 31.3 harmful). Higher-bucket labels are the reverse control (Table f).
2. **Structure alignment is also necessary.** The reference must be a same-caption, same-input belief whose other statistics (colour count, opacity, spatial structure) match the strong prediction. Low TV from explicit degradation — a trained block-average branch (TV 14.2, harmful) or a trained contrast-shrunk branch (TV 15.9; 14.71 vs 9.43 for the free label on the same weights) — does not substitute (§5.5).
3. **Weight.** The label reference's $w$-curve is flat (10.21 / 8.59 / 11.20 / 13.87 for $w = 1.5, 2, 2.5, 3$), the snapshot's steep (8.98 → 30.11 for $w = 1.5 \to 3$); we use $w = 2$ for the label reference and $w = 1.5$ whenever the snapshot is involved. The same shape holds at 20 and 24 px (seed 0): the label reference gives 40.96 / 37.48 / 32.89 / 35.15 / 38.53 at 20 px and 71.12 / 63.09 / 60.41 / 65.77 / 74.61 at 24 px for $w = 1.25, 1.5, 2, 2.5, 3$ (worst-over-sweep / best = 1.25× and 1.23×), whereas the snapshot reference gives 36.75 / 31.78 / 33.66 / 42.02 / 59.04 at 20 px (1.86×) and 64.85 / 53.36 / 52.52 at 24 px for $w = 1.25, 1.5, 2$ (higher weights: [TBD]). At 24 px the snapshot optimum moves to $w = 2$, so the composed row at $w = 1.5$ is not tuned in the snapshot's favour.

The bucket label satisfies both conditions for free: it is a monotone contrast knob on the same network (pure-belief TV 21.6 → 31.9 → 40.9 for bucket 12 → 24 → 64 at 16 px) while colour count stays close to the strong model (78 vs 73) and the layout is that of the same denoising trajectory.
