# 免训练引导基线规格：CDG（Han et al.）、SEG、ICG / TSG

写于 2026-09-14。三篇论文都读的是 arXiv 的 **LaTeX 源码**（`arxiv.org/src/<id>`），公式照源码原样抄录；公式编号和 arXiv HTML 渲染版做过核对。CDG 和 SEG 另外读了官方代码，凡是只在代码里出现、论文正文没写的细节，都标成"（代码）"。论文和代码都查不到、属于我自己推断的内容，标 **[未核实]**。

我们的配置（照 `src/v6/sample_e.py` 和 `train_v7.py` 核对过）如下：
- `UNet2DConditionModel`，eps 预测，DDPM 100 步（训练 T=1000）。像素空间，4 通道 RGBA，输入 `x ∈ R^{B×4×S×S}`，S ∈ {12,16,20,24}。
- 文本条件：`CLIPTokenizer/CLIPTextModel("openai/clip-vit-base-patch32")`，`padding="max_length", max_length=77`，取 `last_hidden_state`（77×512）作为 `encoder_hidden_states`。训练时以 10% 概率把文本置为 `""`，所以 null 分支是训练过的。
- 类别条件：分辨率桶 `class_labels`，经 `class_embedding` 加到 `time_embedding` 的输出上（`emb = time_emb + class_emb`），训练中从不 drop。
- 块结构：down0 = CrossAttnDown @S，down1 = CrossAttnDown @S/2，down2 = DownBlock2D @S/4（无注意力），mid = UNetMidBlock2DCrossAttn @S/4，up0 = UpBlock2D @S/4（无注意力），up1 = CrossAttnUp @S/2，up2 = CrossAttnUp @S。
  - **mid 块的自注意力只有 3×3 / 4×4 / 5×5 / 6×6 个 token**（S = 12/16/20/24）。这对 SEG 影响很大，见 §2。
- 采样器已支持 `e = e_ref + w (e_strong − e_ref)`，以及 `pag_forward`（attn1 处理器替换，层名 `mid|d0|d1|u1|u2`）。

---

## 1. CDG：Condition-Degradation Guidance（Han et al., CVPR 2026）

- **标题**：*Guiding Diffusion Models with Semantically Degraded Conditions*
- **作者**：Shilong Han, Yuming Zhang, Hongxia Wang（arXiv 备注 "Accepted to CVPR 2026"）
- arXiv: https://arxiv.org/abs/2603.10780 ｜ HTML: https://arxiv.org/html/2603.10780v1 ｜ 代码: https://github.com/Ming-321/Classifier-Degradation-Guidance
- 论文只在 Transformer/MMDiT 模型上做了实验（SD3、SD3.5、FLUX.1-dev、Qwen-Image），**没有 UNet 实验**。

### 1.1 原文要点

**CFG（Sec. 3, Eq. 3）：**
```
D^CFG_θ(x_σ; σ, c) = D_θ(x_σ; σ, c) + (w−1) ( D_θ(x_σ; σ, c) − D_θ(x_σ; σ, ∅) )
```

**CDG（Sec. 4, Eq. 5）**：把 ∅ 换成退化条件 c_deg，其余与 CFG 完全相同：
```
D^CDG_θ(x_σ; σ, c) = D_θ(x_σ; σ, c) + (w−1) ( D_θ(x_σ; σ, c) − D_θ(x_σ; σ, c_deg) )
```
原文："replaces the semantically distant null condition ∅ in CFG with a semantically degraded condition c_deg"。所以 CDG **替代 CFG，不是叠加在 CFG 上**。

Appendix B 的 Algorithm 写明每步两次前向（`ε_cond ← D_θ(x_t,t,c)`，`ε_deg ← D_θ(x_t,t,c_deg)`），组合方式是 `ε̂ ← ε_cond + (w−1)·(ε_cond − ε_deg)`。**NFE = 2/步，和 CFG 一样**。

**退化什么（Sec. 5）**：退化的是**文本编码器输出的 token 嵌入序列**，不是类别标签。
- token 分成两类：
  - **content tokens**（有实际语义的词）
  - **context-aggregating tokens（CtxAgg）**。Sec. 1 原文："padding and special tokens that originally lack intrinsic semantics but acquire rich global context through attention"。
- **统一退化比例（Eq. 9）**：
  ```
  r_content = min(R_deg, 1.0),   r_CtxAgg = max(R_deg − 1.0, 0),   R_deg ∈ [0, 2]
  ```
  两类各自按重要性排序，取前 k 个替换：`k_content = ⌊r_content·|T_content|⌋`，`k_CtxAgg = ⌊r_CtxAgg·|T_CtxAgg|⌋`。
- **掩码（Eq. 10）**：
  ```
  m_i = 0  if i ∈ T_content and rank_i ≤ k_content
        0  if i ∈ T_CtxAgg  and rank_i ≤ k_CtxAgg
        1  otherwise
  ```
- **逐位置插值（Eq. 11）**：
  ```
  c_deg = m ⊙ c + (1 − m) ⊙ ∅
  ```
  其中 ∅ 是空串 `""` 的编码结果，按 token 位置逐个对应。
- **重要性排序**用 Weighted PageRank（Eq. 8：`s^(k+1) = A^T s^(k) / ||A^T s^(k)||_1`），A 是第 λ_block 个 transformer 块中文本 token 的自注意力图；多头之间用 VHF+EIR 融合（App. A）。
  - **默认 R_deg = 1.0 时完全不需要 WPR**，原文："all content tokens are degraded at this boundary, bypassing WPR entirely"。此时操作就是"所有 content token 换成 ∅ 的对应位置，所有 padding/CtxAgg token 保留"。
- **无时间区间限制**：所有步都用。原文："we compute the mask m only once at the first denoising step and reuse it throughout generation"。

**默认超参（App. C.3 表 "Optimal hyperparameters for different models"）：**

| 模型 | w | 步数 | λ_block | r_content \| r_CtxAgg |
|---|---|---|---|---|
| SD3 | 7 | 28 | 1 | 1.0 \| 0.0 |
| SD3.5 | 3.5 | 28 | 2 | 1.0 \| 0.0 |
| FLUX.1 | 1.5 (true_cfg) | 28 | 1 | 1.0 \| 0.0 |

w 的原文："set based on default values or official examples"，**也就是直接沿用该模型 CFG 的默认 w**。Sec. 6.3.2 与 App. C.4 显示 R_deg 在 1.0–1.3 附近是平台区。

**同一张表里，CDG 论文自己跑的基线配置（SD3）**，可直接对照：
- ICG：w=7，"seed 42 … used to select a random token ID for each prompt"。
- SEG：γ=3，σ=5，层 13，CFG=1（即关闭 CFG）。
- PAG：γ=3，层 13。

### 1.2 代码层面确认的细节（官方 repo）

- **content/padding 的划分**（`models/pipelines/cdg/process_token.py::analyze_token_content_status`）：
  - CLIP 部分：token 文本为 `<|endoftext|>` 时，**第一个**算 content（句末标记），之后的都算 padding；**其余 token（包括 BOS `<|startoftext|>`）一律算 content**。
  - T5 部分：`<pad>` 算 padding。
  - SD3 中 CLIP 和 T5 两段分开退化（`separate_clip_t5: true`）。
- **"degrade_ratio" 和 "keep_ratio" 的关系**：`utils.py` 把 `degrade_ratio` 转成 `keep_ratio = 1 − degrade`，插值写作 `keep_mask*positive + (1−keep_mask)*negative`，与论文 Eq. 11 一致。配置 `{"content":1,"padding":0}` 的意思是 content 全部换成 ∅、padding 全部保留。
- **pooled 向量不退化**（`use_negative_pooled_prompt_embeds: false`）：c_deg 分支用的仍是正向 prompt 的 pooled CLIP 向量。论文正文没提这一点。
- **λ_block 的含义**（`sd3/mm_dit.py`）：MMDiT 的文本流会逐块更新。退化分支在块 0…λ−1 上用原始 c 计算，到块 λ 时把当时的文本流按掩码与 `context_embedder(∅)` 插值，此后的块沿用插值结果。
- **组合方式**：代码走的是 diffusers 的 `uncond + g·(text − uncond)`，其中 uncond 位置放的是 c_deg 分支，g = w。这与 Eq. 5 等价。

### 1.3 映射到我们的配置

**(a) 公式。** 设 `c = CLIP("caption")`，`∅ = CLIP("")`，二者都用 77 长度的 max_length padding；`M ∈ {0,1}^77` 是 tokenizer 的 `attention_mask`。HF CLIP tokenizer 的 attention_mask 覆盖的正是 BOS、所有词、第一个 EOS，与官方代码的 content 定义完全一致；`openai/clip-vit-base-patch32` 的 `pad_token` 就是 `<|endoftext|>`，这点已在 HF 核对。
```
c_deg[i] = ∅[i]   若 M[i] = 1   (content: BOS, 词, 第一个 EOS)
c_deg[i] = c[i]   若 M[i] = 0   (padding: 后续 EOS, 即 CtxAgg)
e = e(x_t, t, c_deg, lab) + w · ( e(x_t, t, c, lab) − e(x_t, t, c_deg, lab) )
```
- 两个分支的 `lab` 都是**目标桶**，不改。
- 直接套采样器现成的形式即可：`e_ref = e(c_deg)`，`e_strong = e(c)`。
- 这是 R_deg = 1.0 的情形。若要 R_deg ∈ (1, 2]：CtxAgg 位置中再按排序取 `⌊(R−1)·|T_CtxAgg|⌋` 个换成 ∅。
  - 论文 Tab. 3 与 App. C.8 表明，在分层框架内**随机排序与 WPR 效果相当**（FID 34.17 vs 33.89）。所以我们用固定种子随机排序就够了，不必移植 WPR。
  - 我们的 UNet 里没有文本 token 之间的自注意力。若硬要做 WPR，只能改用 CLIP 文本编码器最后一层的自注意力。这属于移植性改动，**论文没有做过**。

**(b) 要 hook 的组件。** 不需要改 UNet。
- 只需在采样前按 prompt 构造好 `c_deg`，作为 `encoder_hidden_states` 传给所有 attn2。
- λ_block 在 UNet 中没有对应物：UNet 的文本嵌入不随块更新，所有 cross-attn 看到的是同一份 `encoder_hidden_states`。所以在输入端替换一次就等价于"从第一块起全部使用 c_deg"。
- 我们没有 pooled 文本向量，这项不适用。

**(c) 扫描建议。**
- `w ∈ {1.5, 2, 3}`：论文做法是沿用 CFG 的 w，而我们 v7r 的 16px CFG 最优是 w=1.5；CDG 的差分向量比 CFG 小，可能需要略大的 w。
- `R_deg ∈ {1.0, 1.2}`。可选 0.7：只退化最重要的 70% content，需要随机排序。

**(d) NFE**：2/步。

**(e) 陷阱**
1. **长 caption 会退化成 CFG。** 如果 caption 被截断到 77（没有 padding），所有位置都是 content，于是 `c_deg = ∅`，CDG 与 CFG 完全相同。重打标（v7r recap）后的 caption 可能很长，**要先统计 `attention_mask.sum()` 的分布**，并报告 CtxAgg 数少于某阈值的样本比例。
2. **BOS 位置替换不起作用。** CLIP 文本编码器用因果掩码，BOS 位置的输出与 prompt 无关，c 与 ∅ 在该位置本来就相同。真正起作用的是词位置和第一个 EOS 位置换成了 ∅ 的 EOS/pad 嵌入，而 padding 位置保留了 c 的全局语义：因果注意力下，pad 位置能看到整句。
3. ∅ 必须和训练时 dropout 所用的 `""` 编码方式一致（同样的 max_length padding）。现在的 `embed([""])` 满足这一点。
4. 不要同时再加 CFG 项。论文中 CDG 就是全部引导；叠加 CFG（3 NFE）属于我们自己的扩展，要单独标注。

### 1.4 与我们方法的区别（简述）

两者都用同一网络，把参考分支的条件"降级"成一个"几乎正确"的版本。但降级的维度相反：

| | CDG（Han et al.） | 我们 |
|---|---|---|
| 被降级的条件 | **文本**：content token 换成 ∅，保留 padding 里的全局语义 | **分辨率桶标签**：换成更低的桶，caption 完全不变 |
| 保持不变的条件 | 分辨率/类别 | 文本 |
| 差分方向的语义 | 细粒度**语义/组合**差异（"good vs almost good" 的文本对齐） | **同一内容在更高分辨率网格下的结构/细节**差异，基本不含文本对齐分量（所以我们有 `cfg_text` 选项） |
| 对训练的依赖 | 真正免训练：只要求文本编码器有 padding/special token | 依赖网络在多桶阶梯上训练过（低桶标签必须有"低分辨率版本"的含义）。可选的早期 EMA 快照属于 autoguidance 式的模型退化，CDG 没有这一项 |
| 替代 CFG？ | 是，2 NFE | 是，2 NFE（加快照调度或 cfg_text 时另计） |
| 预期受益指标 | CLIP/VQA/组合性，FID 小幅改善 | FD/质量/像素网格保真度 |

两种退化正交，可以组合，例如参考分支同时用 `c_deg` 和低桶标签。做论文对比时，CDG 应同时报 FD 和 CLIP score。

---

## 2. SEG：Smoothed Energy Guidance（Hong, NeurIPS 2024）

- **标题**：*Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention*
- **作者**：Susung Hong
- arXiv: https://arxiv.org/abs/2408.00760 ｜ HTML: https://arxiv.org/html/2408.00760v2 ｜ 代码: https://github.com/SusungHong/SEG-SDXL（`pipeline_seg.py`）

### 2.1 原文要点（Sec. 3.3）

**注意力 logits 模糊（Eq. 6）：**
```
(QK^⊤)_seg = G * (QK^⊤)
```
原文：G 是标准差为 σ 的 2D 高斯核，`*` 为 2D 卷积；"we replace the original attention weights with (QK^⊤)_seg and compute the final value as in ordinary self-attention"。模糊作用在 **softmax 之前**的 logits 上（Theorem 3.1："applying a Gaussian blur to the attention weights a before the softmax operation"）。

**等价的 query 模糊（Proposition 3.1，Eq. 7–8）**：
```
G ∗ (QK^⊤) = B(QK^⊤) = (BQ)K^⊤ = (G ∗ Q)K^⊤
```
即只需把 **Q 按其空间排布做 2D 高斯模糊**，K 和 V 不动。这样避免了 O(N²) 的开销。

**SEG（Eq. 9）**：
```
dx = [ f(x,t) − g(t)² ( γ_seg s_θ(x,t) − (γ_seg − 1) s̃_θ(x,t) ) ] dt + g(t) dw̄
```
s̃_θ 是 attention 被模糊后的预测。换算成 eps：`e = ẽ + γ_seg (e − ẽ)`。

**SEG+CFG（Eq. 10）：**
```
dx = [ f(x,t) − g(t)² ( (1 − γ_cfg + γ_seg) s_θ(x,t) + γ_cfg s_θ(x,t,c) − γ_seg s̃_θ(x,t) ) ] dt + g(t) dw̄
```
即 `e = e_u + γ_cfg (e_c − e_u) + γ_seg (e_u − ẽ_u)`。在论文公式里，模糊分支是**无条件**的。

**默认值与建议：**
- Sec. 5.1："we choose the same attention layers (mid-blocks) and guidance scale as PAG"，"We set γ_seg to 3.0"。
- Sec. 3.3：主张**固定 γ、只调 σ**。σ→0 时等于原模型；σ→∞ 时"the attention weights merely adopt a single mean value across spatial axes"，即 query 被替换成空间均值。
- Tab. 2（SDXL 文本条件，无 CFG）：σ = 1/2/5/10/∞ 对应 FID 48.3/41.8/33.8/29.3/26.2，σ 越大越好但 LPIPS 偏离越大。
- 论文**无时间区间限制**。

### 2.2 代码层面的细节（`SEGCFGSelfAttnProcessor`）

- **核大小**：`kernel_size = ceil(6σ) + 1 − ceil(6σ) % 2`（奇数）。`gaussian_blur_2d` 里再截断为 `min(kernel_size, H − (H%2 − 1))`，然后用 `reflect` padding，并对各通道分组卷积。
- **排布**：query 先 reshape 成 `(B, heads·head_dim, H, W)` 再模糊，`H = W = isqrt(N)`（假设方形）。
- **σ→∞**：`seg_blur_sigma > 9999` 时执行 `query_ptb[:] = query_ptb.mean(dim=(-2,-1), keepdim=True)`。README 建议"Controlling it exponentially (e.g., 1, 4, 16, ...)"。
- **层选择**：`seg_applied_layers=['mid']`，替换 UNet 中所有名字含 `attn1` 且以 `mid` 开头的模块的 processor。也可按 `d4, m0` 这类索引指定。
- **组合公式与论文不一致**，需特别注意：
  - 仅 SEG（`guidance_scale=1`）：`e = e_c + seg_scale·(e_c − ẽ_c)`，两个分支都带**文本条件**。换成论文的形式，**γ_seg = seg_scale + 1**。
  - SEG+CFG：`e = e_c + (g−1)(e_c − e_u) + seg_scale·(e_c − ẽ_c)`。模糊分支带**文本条件**（batch 是 `[neg, pos, pos]`），而 Eq. 10 用的是无条件模糊分支。
  - 论文说 γ_seg = 3，代码默认 `seg_scale = 3`。**论文的 γ=3 指的是论文形式还是代码形式 [未核实]**，两种都要覆盖。

### 2.3 映射到我们的配置

**(a) 公式。** 我们的模型文本条件化，参照论文 Tab. 2 那种"不开 CFG 的文本条件 SEG"，也就是代码的做法：
```
ẽ = e(x_t, t, c, lab | attn1 的 Q 被 σ 高斯模糊)
e = ẽ + w (e(x_t,t,c,lab) − ẽ)
```
- 在我们的 `e_ref + w(e_strong − e_ref)` 形式下：论文形式 w = γ_seg；代码形式 w = seg_scale + 1。
- 可选的 3 NFE 版本（按代码）：`e = e_u + w_cfg(e_c − e_u) + s·(e_c − ẽ_c)`。

**(b) 要 hook 的组件。** 写一个 attn1 处理器（与现有 `_IdentityAttnProcessor` / `pag_forward` 同一套机制），只作用于被扰动的那次前向：
```
q = attn.to_q(h); k = attn.to_k(h); v = attn.to_v(h)          # h: (B, HW, C)，BasicTransformerBlock 内 attn1 输入为 3D
q -> (B, heads, HW, d) -> (B, heads*d, H, W),  H = W = isqrt(HW)   # 我们的桶都是方形
σ 有限: q = gaussian_blur_2d(q, ksize(σ), σ)  (reflect pad, 分组卷积)
σ = ∞ : q = q.mean((-2,-1), keepdim=True)
out = SDPA(q, k, v) -> to_out[0] -> to_out[1]
```
按块，层名沿用 `pag_forward` 的约定：

| 层名 | 模块 | 空间尺寸 |
|---|---|---|
| `mid` | `mid_block.attentions.0.transformer_blocks.0.attn1` | S/4 |
| `d1` | `down_blocks.1.attentions.{0,1}` | S/2 |
| `u1` | `up_blocks.1.attentions.{0,1,2}` | S/2 |
| `d0` | `down_blocks.0.attentions.*` | S |
| `u2` | `up_blocks.2.attentions.*` | S |

**(c) 扫描建议。**
- 层：`{mid（论文默认）, d1+u1, all}`
- σ：`{1, 2, ∞}`
- w：`{3, 4}`，分别对应"论文 γ=3"和"代码 seg_scale=3"。

**(d) NFE**：2/步（仅 SEG）；3/步（叠加 CFG）。

**(e) 陷阱**
1. **mid 块只有 3–6 px 见方。** σ 以该层 token 为单位，所以在 S/4 的地图上 σ ≥ 2 基本等于 σ=∞。论文里的 σ=5/10 是针对 SDXL mid 块 32×32 的 token 图。因此不能照搬 σ，应当同时报"等效相对尺度"σ/H。在 12px 桶上，mid 块的 H = 3，核会被截断到 3。
2. reflect padding 要求 pad < H。按上面的截断公式，H=3/4/5/6 时 pad 为 1/2/2/3，都合法。但如果自己改写了核大小逻辑，要重新检查。
3. 只模糊被扰动那一份 batch，并且前向结束后必须恢复原来的 processor。`pag_forward` 用 `try/finally` 和 dict 拷贝，照搬即可。
4. σ=∞ 并不等于"对 V 取平均"：所有位置共享同一个均值 query，所以输出在空间上是常数 `softmax(q̄K^⊤)V`。它和 PAG（恒等注意力）正好是两个极端，可以一起报。

---

## 3. ICG / TSG（Sadat et al., ICLR 2025）

- **标题**：*No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion Models*
- **作者**：Seyedmorteza Sadat, Manuel Kansy, Otmar Hilliges, Romann M. Weber（arXiv 备注 "Published as a conference paper at ICLR 2025"）
- arXiv: https://arxiv.org/abs/2407.02687 ｜ HTML: https://arxiv.org/html/2407.02687v2 ｜ 论文未给官方代码链接；pseudocode 在 App. G

### 3.1 ICG：Independent Condition Guidance（Sec. 4）

**原文公式**
- CFG（Eq. 5）：`D̂_θ(z_t,t,y) = D_θ(z_t,t,y_null) + w_CFG ( D_θ(z_t,t,y) − D_θ(z_t,t,y_null) )`
- 理论依据（Eq. 7）：`∇ log p_t(z_t | ŷ) ≈ ∇ log p_t(z_t) + ∇ log q(ŷ) = ∇ log p_t(z_t)`，其中 ŷ ~ q(ŷ) 与 z_t 独立。
- Algorithm "Sampling with ICG"（App. G）：
  ```
  D̂_ICG(z_t, t, y) = D(z_t, t, ŷ) + w_ICG ( D(z_t, t, y) − D(z_t, t, ŷ) )
  ```
  注意 "Pick a random ŷ independent of z_t" 这一步写在**每个时间步的循环内部**。

**ŷ 怎么抽（Sec. 4 "Implementation details" 与 App. G 的 pseudocode）**
- 两种方式：高斯噪声，或"a random condition from the conditioning space, such as a random class label or random clip tokens"。
- pseudocode 原文：
  ```python
  y_random = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, ))                  # 随机类别
  random_idx = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, MAX_LENGTH))       # 随机文本 token
  random_tokens = text_encoder(random_idx, attention_mask=None)[0]
  noise_embedding = torch.randn_like(embeddings) * embeddings.std()          # 高斯嵌入
  ```
  随机文本就是整条 77 长的**均匀随机 token id**，不特意插 BOS/EOS，送入文本编码器编码。
- Sec. 7 的 Tab.（DiT）：Gaussian FID 5.50，random condition 5.55，两者相近。作者对 random condition 略有偏好（"stays closer to the conditioning distribution"）。

**默认超参（App. G 表 "Hyperparameters used for the ICG experiments"）：**

| 模型 | ICG 模式 | ICG scale | CFG scale |
|---|---|---|---|
| DiT-XL/2 | random class | 1.4 | 1.5 |
| Stable Diffusion | random text | 3.0 | 4.0 |
| Pose-to-Image | Gaussian | 3.0 | 4.0 |
| MDM | Gaussian | 2.5 | 2.5 |
| EDM | random class | 1.05 | 1.1 |
| EDM2 | random class | 1.25 | 1.25 |

- 规律：**ICG 的 w 等于或略低于 CFG 的 w**。
- 论文**无时间区间限制**。
- 用的是哪一版 Stable Diffusion，主表只写了引用 Rombach et al. **[未核实]**。App. E 与 SAG/PAG 对比时用的是 SD 2.1。

### 3.2 TSG：Time-Step Guidance（Sec. 5）

**原文公式（Eq. 8）：**
```
D̂_θ(z_t, t) = D_θ(z_t, t̃) + w_TSG ( D_θ(z_t, t) − D_θ(z_t, t̃) )
```
- 条件模式下（Algorithm "Sampling with TSG"）两个分支都带同一个 y：
  ```
  D̂_TSG = D(z_t, t̂_emb, y) + w_TSG ( D(z_t, t_emb, y) − D(z_t, t̂_emb, y) )
  ```
- **扰动方式**（Sec. 5 "Implementation details"）："perturbing the time-step embedding with zero-mean Gaussian noise according to t̃_emb = t_emb + s t^α n where n ~ N(0, I)"。s 与 α 的选取原则是"the scale of the noise portion becomes comparable to the scale of the time-step embedding"。
- **可只扰动部分层**。Sec. 5 原文："using t̃_emb for the first 10 layers and t_emb for the rest"。App. G 原文："the first N layers of the encoder and decoder in UNet-based architectures"。
- **App. G pseudocode：**
  ```python
  def get_power_schedule(t_emb, t, std_scaling=True):
      if t < T_MIN or t > T_MAX: return t_emb
      noise_scale = S * t ** (ALPHA)
      if std_scaling: noise_scale = noise_scale * t_emb.std()
      return t_emb + torch.randn_like(t_emb) * noise_scale
  ```
  constant schedule 就是 α=0 的特例。作者推荐 power schedule，并说"We also found it useful to apply TSG only at intervals during the sampling, i.e., for t ∈ [T_min, T_max]"。

**默认超参（App. G 表 "Hyperparameters used for the TSG experiments"）：**

| 模型 | 模式 | schedule | w_TSG | 参数 |
|---|---|---|---|---|
| DiT-XL/2 | 无条件 | constant | 5.0 | T∈[200,800], s=1.0 |
| DiT-XL/2 | 条件 | power | 2.5 | T∈[0,1000], α=1, s=2 |
| SD | 无条件 | constant | 3.0 | T∈[100,900], s=1.25 |
| SD | 条件 | power | 4.0 | T∈[400,1000], s=3.0, α=0.25 |

- Sec. 7 消融（DiT）：s ∈ {1, 2, 2.5} 时 FID 为 10.23/6.85/7.94；α ∈ {0.75, 1, 1.25} 时为 7.22/6.39/6.47；最大层 ∈ {5, 10, 15} 时为 7.84/6.85/7.65。
- ICG+TSG 组合在 DiT 上 FID 5.76，好于单用 ICG 的 6.47 和单用 TSG 的 9.55。但**论文没有给出组合公式 [未核实]**。

**未核实点**
- power schedule 里的 `t` 是归一化的 t∈[0,1]，还是 0–1000 的整数步，**[未核实]**。Background 定义 t ∈ [0,1]；但 T_MIN/T_MAX 用的是 0–1000 刻度；而 DiT 设置 α=1、s=2 时，如果 t 取到 1000，噪声会达到 2000×std，显然不合理。**据此推断 `t^α` 用的是 t/1000**。
- `t_emb` 指正弦投影后的向量，还是 MLP 之后的时间嵌入，**[未核实]**。"只扰动前 N 层"的说法暗示是各层都会消费的那个时间嵌入，也就是 MLP 之后的那个。

### 3.3 映射到我们的配置

#### ICG

**(a) 公式。** `e = e(x_t,t,ŷ,·) + w ( e(x_t,t,c,lab) − e(x_t,t,ŷ,·) )`。建议跑两个变体：
- **ICG-text（主基线，对标 CFG）**：`ŷ = CLIP(randint(low, high, (B,77)))`，桶标签保持 `lab`。可选 Gaussian 版 `ŷ = randn_like(c) * c.std()`，std 为整张量的标量。
- **ICG-label（与我们方法最接近的对照）**：文本保持 c，桶标签从全部桶里**均匀随机**抽 `lab_hat`。这直接检验"**低**桶"这个结构是否必要，还是任意错误的桶都行。注意均匀抽样有 1/7 的概率抽到正确桶，那一步的引导为零；也可能抽到更高的桶。

**(b) 要 hook 的组件**：无，只改条件输入。

**(c) 扫描建议**
- `w ∈ {1.5, 2, 3}`
- ŷ 类型 `{random text, Gaussian}`
- 抽样频率 `{每步重抽（Algorithm 原意）, 每样本固定一次}`

**(d) NFE**：2/步。

**(e) 陷阱**
1. 随机数要用**独立的 `torch.Generator`**，否则会改变 DDPM 的噪声序列，导致同 seed 下各方法不可比。
2. token 范围：pseudocode 用的是 `[0, NUM_TOKENS)`，会抽到 BOS(49406)/EOS(49407)。CDG repo 的复现则避开了首 0.1% 和末 5% 的 id。两种都可以，但要写清楚用的是哪种。
3. 我们的 null 分支是训练过的。ICG 本来是为"没训练 null"的模型设计的，在我们这里只作为参照基线，预期大致和 CFG 持平，不会超过 CFG。

#### TSG

**(a) 公式。** 条件模式，两个分支的文本和桶都相同：
```
temb   = unet.time_embedding(unet.time_proj(t))          # (B, 512)，加 class_emb 之前
temb~  = temb + s · (t/1000)^α · temb.std() · n,  n ~ N(0,I)   仅当 T_min ≤ t ≤ T_max
e = e(x_t, temb~ + class_emb) + w ( e(x_t, temb + class_emb) − e(x_t, temb~ + class_emb) )
```

**(b) 要 hook 的组件**
- 全层版本：在 `unet.time_embedding` 上注册 `register_forward_hook`，返回 `out + noise`。只在被扰动的那次前向里生效，结束后移除。
- 由于 `emb = time_emb + class_emb` 是加法，在 time_embedding 输出上加噪声，等价于在总 emb 上加同样的噪声，标签信息本身不受影响。
- 部分层版本（论文说 UNet 用"encoder 和 decoder 的前 N 层"）：给选定的 `ResnetBlock2D` 注册 `forward_pre_hook(with_kwargs=True)`，把 `temb` 参数换成扰动版。可以取 down 路径前 N 个 resnet 加 up 路径前 N 个 resnet；我们每个 down 块 2 个、每个 up 块 3 个 resnet。**这是按原文描述的移植**，具体取哪几个 resnet 属于我们自己的选择。

**(c) 扫描建议**
- `w ∈ {1.5, 2.5, 4}`
- `(s, α) ∈ {(2, 1), (3, 0.25)}`：分别是论文 DiT 条件模式和 SD 条件模式的设置。
- 区间 `{[0,1000], [400,1000]}`

**(d) NFE**：2/步。若叠加 CFG 则为 3/步。叠加公式 `e = e_c + (w_cfg−1)(e_c−e_u) + (w_tsg−1)(e_c−e_t̃)` 是**我们自己构造的，论文没有给出**。

**(e) 陷阱**
1. diffusers 的 t 是 0–999 的整数；`t^α` 请用 t/1000，理由见上方未核实点。
2. `temb.std()` 必须取加 class_emb **之前**的时间嵌入。
3. 噪声 n 每步重抽，每个样本独立，同样要用独立的 generator。
4. 我们的类别嵌入正好是加到 temb 上的。所以从结构上看，**TSG 是我们"换低桶标签"的随机方向对照**：TSG 在 emb 上加各向同性高斯噪声，我们则是沿着"低桶 − 目标桶"这个有语义的方向做替换。ICG-label 是随机标签对照。这两个对照正好把"扰动方向是否有结构"与"扰动有多大"区分开。

---

## 4. 汇总

| 方法 | 参考分支 | 替代/叠加 CFG | NFE/步 | 需 hook | 论文默认 | 我们的扫描 |
|---|---|---|---|---|---|---|
| CDG | 同网络；文本 content token 换成 ∅，保留 padding | 替代 | 2 | 无（输入端构造 c_deg） | w 同 CFG，R_deg=1.0，全程 | w{1.5,2,3} × R{1.0,1.2} |
| SEG | 同网络同条件；attn1 的 Q 做 σ 高斯模糊 | 替代（可 +CFG，3 NFE） | 2 | mid/d1/u1 等的 attn1 processor | 层=mid，γ=3，σ 可到 ∞ | 层{mid,d1+u1,all} × σ{1,2,∞} × w{3,4} |
| ICG | 同网络；随机文本（或随机桶）/高斯嵌入 | 替代 | 2 | 无 | w 等于或略低于 CFG | w{1.5,2,3} × {text,gauss,label} |
| TSG | 同网络同条件；time emb 加噪 | 替代（可 +CFG，3 NFE） | 2 | time_embedding 前向 hook（或 resnet 预 hook） | SD 条件：w=4，s=3，α=0.25，t∈[400,1000] | w{1.5,2.5,4} × (s,α){(2,1),(3,.25)} × 区间 2 种 |

所有方法都要和我们的方法在**相同 NFE**下比较，即 2 NFE 对 2 NFE；3 NFE 的组合版本单独列出。
