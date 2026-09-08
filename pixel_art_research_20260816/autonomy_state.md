# 自主推进状态 (cron 每次先读此文件, 再行动, 做完更新此文件)

用户 2026-09-06 授权: 用户几天不在, Claude 自行推进"调研→架构→探针→不行就换"循环, 宽泛调研不死磕。
目标不变: ICLR 论文, 核心=极低分辨率(12-24px)像素画生成的**新机制/新架构**, 客观指标要赢过 v7。

## 判据与硬约束
- **指标(2026-09-06 修正)**: **公平 FD-DINOv2@16px**(src/v6/fd_fair.py, 参考集与生成同走 to_tensor 管线; 旧 fd_dino 数值全部作废, 见 experiment_log "指标修正")。n=3304(413 prompts×8, eval_probe.sh)。**真实地板 3.45, v7 = 53.21, 最强简单基线 probe_tv(TV w=0.1) = 42.82**, 越低越好。
- **胜负线(2026-09-06 19:40 三次修正; 22:30 四次修正: 基线重训)**: 旧协议(413 词表 prompt×8)有 ~15 点 prompt 分布错配伪影, **作废**。新主指标 = **matched FAIR FD@16**: baseline/eval_matched.sh(3000 张 held-out 真实精灵的 caption, n=1, seed 0; 参考集不变, 地板 3.45)。**但 v7/probe_tv 的 matched 数字(16.66 / 28.02)被记忆污染**(held-out 只从参考集排除, 训练集含全部 oga; v7 8.4% 近逐像素复现), 只作参考。**干净基线 = v7h**(v7 配方随机初始化, 训练排除 ref∪held 5,928 张, 80k 步, 快照每 5k; 在训, 09-07 ~06:00 UTC 完)。判据: 探针(**必须从 v7h 微调或同配方训练**) matched FD 比 v7h(及 v7h+最佳零训练引导) 低 ≥4 点才算信号; 差异 <4 不作结论; 每个探针跑 fd_decomp.py 看 recall/coverage。**(09-07 07:50 五次修正) 对照线 = v7h + 自引导(弱=同 run step10k EMA, 替代 CFG): w=1.5 → 8.98 / q16 7.82; w=2 → 11.33 / q16 6.71**(取各自最优: 原始 8.98, q16 6.71)(v7h 裸 21.98/12.64)。任何新机制探针也必须允许配自引导后再比(公平: 对照有自引导, 探针也有), 目标 ≤7.3 / q16 有实质下降(地板 3.45)。自引导本身是 Karras 2024 已发表方法, novelty=0, 只能作对照/组件。
- **一个探针最多 ~1 天**(训练+采样+FD)。超时未出结果 → 杀掉记原因换下一个。
- **每轮最多 2 探针并行**(GPU2 + GPU3)。优先"从 v7 断点微调 20k 步"的廉价形式; 只有机制上必须从头训时才从头(≤40k步)。
- **硬规则**: 只用 node03(ssh emnlp) GPU2/GPU3; **node09(kw) 一律不碰**(ljq的); 共享账号只在 /mnt/data/kw/RoundSquisheen/pixel/pixel 内读写; 后台任务必走 supervise.sh + tmux; node03 需 PYTHONNOUSERSITE=1; 判断在跑看产出增长。
- **每个有结论的步骤**: 写 experiment_log.md(结果) / arch_ideation_log.md(想法与调研), 更新本文件, `cd c:/Codes/pixel && git add -A && git commit && git push`。

## 已证伪/已堵死(别再回头) —— 公平 FD@16(旧值括号内, 已作废)
| 方向 | 公平 FD@16 | 结论 |
|---|---|---|
| 真实 held-out 3000(地板) | **3.45** | 任何模型的理论下限 |
| v7 连续原生前馈(基线) | **53.21** (64.80) | 主基线 |
| **probe_tv 连续+TV 平坦先验 w=0.1** | **42.82** (70.65) | **最强简单基线**; 旧指标误杀, 分片恒定/硬边是 v7 主短板 |
| probe_pal 连续+软调色板吸附 w=0.1 | 54.24 (66.26) | ≈v7, 无效 |
| probe_tv3 / probe_tv10 TV w=0.3 / 1.0 | 46.57 / **569.35** | TV 甜点极窄(0.1); w=1.0 完全崩坏(过平滑成空白/平色) |
| probe_struct oracle(真 S=[alpha,4级明度]) | 13.52 (52.14) | ≈413 源地板; **数值被记忆污染**(源在训练集里), 只取定性: 难度全在 S |
| probe_paltok oracle(真 8 色调色板 token) | 18.18 | **记忆污染实证**: 无空间信息却空间复现 → oracle 源必须剔除训练 |
| **probe_sgen 两阶段(v7微调只生成 S → probe_struct 上色)** | **61.66** | **杀**; 阶段一结构域 FD 19.17(v7 23.55, tv 18.94, 地板 1.40): 只学低熵 S 也没变容易; 上色器对生成 S 曝光偏差崩 |
| **probe_tv + 事后 octree 量化 8/12/16/24/32/48 色** | 65.03/40.83/**35.24**/35.59/36.81/37.69 | **零学习的新最强基线 35.24**; 甜点 15-20 色/张(真实中位 34, P10 9); 8 色立刻崩 |
| **probe_palhead 调色板因子化 x0 头 K=16(硬末步 / 软末步 / 软+q16)** | **78.34** / 52.09 / 45.97 | **杀**; 每张实际只用 ~7 色(远低于真实 P10=9) → 硬吸附致命; 软末步≈v7 说明头未改善底层预测; 学到的量化劣于事后量化 |
| **probe_selfq 投影入环自条件 K=16(从 probe_tv 微调 20k) / 配对对照 probe_tv_cont / selfq 采样置零消融** | **42.55 → +q16 38.45** / 46.91 → 39.91 / 47.23 → 37.51 | **杀(无信号)**; 原始域 −4.4 vs 配对全是"少色"效应(唯一色 66 vs 73), +q16 后消失(38.45 vs 37.51); 训练 sc=1/0 的 main 损失无差 → 网络基本忽略投影通道; 结构域 18.43(tv 18.94) 持平。配对对照暴露 **run 间方差 ±4** |
| v6f 朴素离散 absorbing | (160.7, 未重测) | 崩坏, 离散劣于连续 |
| v_ord 有序调色板离散 | (252.3, 未重测) | 更差 |
| SD-πXL(SDS优化式) | 极低分辩率崩 | 对照基线 |
| **能量分数/随机去噪器 probe_es(纯 / hyb300 混合)** | matched 28.47 / 20.69 (配对 ctrl@100 15.03) | **杀(两枚)**: ξ 被使用但后验采样引入像素级散点, DINOv2@16 重罚; 无 recall/coverage 收益 |
| CFG 7/10、CADS、引导区间 (v7h 上零训练) | 42.95 / 62.12 / 44.6~88.6 / 32.57 (v7h 21.98) | 全劣; v7h 是"引导越弱越好"的模型, 缺的是纠错不是条件对齐 |
| 自引导弱参考 = blur 输入 / 1px 平移输入 / 更高分辩率桶(20/24/64) | 223 / 18.95 / 17.1~19.3 | 输入退化崩(预测不可比); 高分辩率信念只收缩 recall; 通道解耦(rgb/alpha 权重)无信号 |
| APG 第二组件(Sadat 2024, 叠 stack, rgb/full/仅投影/仅动量) | stack w1.5 9.94 / w2 10.9~13.4 (无 APG 7.67 / 10.83) | **杀(组件)**: 纠错方向=径向补对比度, 正是 APG 去掉的平行分量; 动量有害 |
| FDG 频域分权(Sabour 2025, 1 层 2×2, w_low<w_high) / 双参考区间调度(快照仅 [0,.5]/[.5,1]/[.2,.8]) | FDG 8.96~18.33 / 区间 7.70(低噪声)/8.99(高噪声)/8.84(中段) vs 7.67 | **杀(两组件)**: 削低频 mean_term 单调升(补对比度本身是低频量); 快照贡献只在低噪声半程(诊断价值, 无增益). 引导侧第二组件三连无 → 不再从引导侧找 |
| **probe_cc 训练版对比度不足参考(v7h 微调 20k, RGB 向均值收缩 0.6 第二标签集)** | coarse w1.5/2/3 = **14.71**/19.98/44.90(裸 20.63; 同权 bk12 9.43, 自引导 8.77) | **杀(组件), 留作机制消融**: 与块平均版相反, 有效(−6) → "低对比度+结构对齐"必要条件被受控证实; 但远弱于免费的 bucket:12 参考 → 人为退化 ≠ 模型自身分辩率信念; 论文零训练版为主 |
| **probe_cg 训练版粗信念引导(v7h 微调 20k, 2×2 块平均粗视图第二标签集)** | coarse w1.5/2/3 = 21.53/35.22/59.38(裸 19.17; 同权 bk12 10.12, 自引导 9.10) | **杀**: 显式低通分支不是正确弱参考, w↑ mean_term 13→42 爆; 证伪"低分辩率信念=低通"解释 → 改为"条件诱导简单性先验" |
教训: ①离散调色板整族作废; ②"手工先验没用"是旧指标误判, 硬边/分片恒定先验有效但 TV 本身无 novelty, 只能当基线; ③结构 S 决定一切, **但把 S 分解出来单独生成并不更容易**(sgen 19.17 vs 地板 1.40), 两阶段还引入曝光偏差 → "先结构再上色"大方向 1 杀+反证, 仅留"单模型非对称调度"当低优先备胎; ④先验证指标再解读 Δ; ⑤结构域 FD(fd_struct.py)显示 TV 也是结构最好的 → 短板是**局部离散结构(分片恒定/硬边/少色)**, 不是全局布局。⑥**颜色维度已被平凡手段(TV+q16)基本解决**, 剩余 35 中结构域占 15.5(地板 1.4); "架构化少色瓶颈"方向 1 杀且预期上限≈事后量化 → 不再投; 下一机制必须提升**单模型内部的局部结构质量**, 且 +q16 后 < 32。⑦(cycle 4) **"离散/少色假设"整族(有序离散、调色板损失、调色板头、投影自条件)5 个探针全 ≤ 事后量化 → 该假设族标废, 不再投任何"让输出更少色/更硬边"的机制**; 差异 <4 点不作结论(run 间方差 ±4), 2-3 点差别需多 seed。剩余 35→3.45 的差距**先诊断再建**: 采样/引导侧(CFG、步数、自引导)零训练扫描 + 训练目标层面"逃离均值回归"(评分规则/能量距离、对抗去噪、分类式输出、AR)。

## 循环协议(每 cycle)
1. **RESEARCH**: 起 2-3 个 subagent 并行——(a)该方向文献机制/怎么做的, (b)novelty 撞车检查(最像的 3 篇+相似度), (c)极低分辩率可行性/坑。汇总写 arch_ideation_log.md。判定: 有新机制且未撞车 → BUILD; 否则 → 下一个候选。
2. **BUILD**: 写探针代码(优先复用 train_v7/train_probe/sample_e/eval_probe 骨架), 服务器上冒烟 200 步, 确认 loss 正常、能采样。
3. **TRAIN**: supervise.sh+tmux 起在空闲 GPU, 日志 logs/<name>.log。cron 每 15 分只报一行进度; .FAILING 则诊断修复重启(同错 2 次则杀)。
4. **EVAL**: 训练完 → eval_probe.sh(或侧条件探针 eval_cond.sh) 16px 采样 3304 张 + 公平 FD; 结果 runs_out/<name>_fd.json(字段 fair_fd16), 汇总在 runs_out/fair_fd16.json。
5. **DECIDE**: <38 → 深挖(消融/放大/12&24px 也测/写论文级图); 否则杀, 记入"已证伪", 回到 1 取下一候选。
**不死磕**: 同一"大方向"连续 2 个探针都 >45 → 整个大方向标废, 换大方向。

## 候选队列(按优先级; 调研后可重排/增删)
0. ~~[离散假设入环] probe_selfq 投影入环自条件~~ **已杀(42.55/38.45, 消融证明效应=少色)**; B 置信评论员 restart / 区域亲和头同属"离散假设族", 一并不再投。
0'. **[cycle 5 诊断, 零训练] 采样/引导侧扫描**(probe_tv): CFG {1.5, 2.5, 7}(现 4) ×(原始/+q16); DDIM 50 步; **自引导**(Karras 2024: 用弱模型 v7 代替无条件项/或 tv+v7 混合)。目的: 判定 35 中多少是引导/采样伪影(过饱和、均值偏移、多样性损失), 而不是模型能力。任一 +q16 < 32 → 立即成为新基线且重定判据。
0''. **[cycle 5 调研] 逃离均值回归的训练目标**: (i) 评分规则/能量距离去噪器(Distributional Diffusion / energy score, 输出是 p(x0|xt) 的**样本**而非均值); (ii) 对抗去噪(DDGAN/Diffusion-GAN/ADD 的判别器仅在 16px); (iii) 分类式像素输出(固定全局码本 CE, 模式寻求) + 连续噪声; (iv) 光栅 AR / PixelCNN 式 256-token 文本条件模型(16px 下序列极短)当"没人跑过的强基线"。判定标准: 机制必须改变**损失/参数化/采样目标**, 而不是"多给网络看一眼"。
1. ~~[架构化少色/分片恒定机制] 调色板因子化去噪头~~ **已杀(78.34), 不再投**。原文: 证据链 = TV 是目前唯一有效手段(42.82, 结构域也最好), 短板是局部离散结构。把它做成**架构瓶颈而非损失**: 去噪网络在 x0 参数化下输出 (a) 每图 K 个颜色(全局 token 头, K≈8-16) + (b) 每像素对 K 的 logits, x0 = softmax(logits/τ)·palette(τ 随训练退火/随 t 调度), eps 由 x0 反推, 损失不变。任何样本天然少色+分片恒定+硬边, 且调色板可解释/可编辑(论文卖点)。与已废的 probe_pal(软吸附**损失**, 无效)和 v_ord(离散扩散)不同: 连续噪声+离散输出流形。风险: softmax 平均出灰色(需 τ 退火/straight-through)、K 固定; novelty 撞车检查: differentiable colour quantisation(ColorCNN)、VQ-x0、"discrete-continuous" 扩散、PaletteNet。RESEARCH 后 BUILD: 从 v7 微调 20k, 只换 conv_out 头 → probe_palhead。
2. **[加强基线] TV 权重扫描 已完**: w=0.1 42.82 / 0.3 46.57 / 1.0 569.35(崩): 平凡先验能走多远, 新机制的增益才诚实。
3. **[逐尺度一致性/多分辩率联合] loop-F**: 控制实验 32→16 BOX 更差(v7 84, tv 53) → 降为低优先。
4. **[区域图生成] loop-O**(区域=颜色分片, 与 1 一脉; 若 1 有信号可作其"显式区域"升级)。
5. **[结构优先备胎] 单模型模态非对称噪声调度**(S 通道快调度): 前提已被 sgen 阶段一削弱, 仅当 1/3/4 都死再考虑。
6. **[精确似然/EBM] loop-E**; 7. **[宽泛再调研]**。

## 当前状态
- **cycle**: 7
- **phase**: **cycle 8 论文整合 + 补漏实验**。paper_assets/paper_outline.md 已出(含 17 项缺口清单)。dseedR 完(泛化表两 seed 齐, experiment_log 19:55)。**v7s 完**(V7S_DONE 09-08 03:50, experiment_log 同刻): 干净第二模型 @16 裸 28.00 → bk12 12.80 / autog 13.95 / 复合 **10.54**(−62%), 反向 bk20 25.31(q16 24.14 远差) —— 排序与分解与 v7h 完全一致, paper_outline 表(c) 已换 v7s 为主。**在跑**: GPU3 `diag_v7s2`(supervise.sh 03:53, logs/diag_v7s2.log/.supervisor/.FAILING, 末行 DIAG_V7S2_DONE, .done 可续跑, GPU3 独占 16px ~4 min/项 共 22 项 ~2h): v7s seed1 四行@16 + v7s @20/24 四行(cfg4/bk16/autog/bk16s10k) + v7h seed2 @20/24 四行 + @12 两行 + fd_decomp。dalign **完**(04:10, experiment_log 同刻; paper_outline 表(g″)): 7 行修法无一回到 CLIP 30.06, FD–CLIP 一条前沿, 复合∘bucketu:12 w1.5 = 8.60/CLIP 29.77(真实水平) 为操作点 → 主表不换, 限制节 + 附录; 不再深挖。diag_metric2 **完**(05:05, experiment_log; paper_outline 表(h), #9 DONE): Inception FID/KID 与 FD-DINO Spearman .86@16(n=121)/.96@20/.68@24; **复合在每个 R/模型/度量下最优**, 反向参考处处有害; 但**纯标签参考的增益 Inception 弱可见**(24/32px FID 不降, KID 略升) → 论文推荐复合参考为方法本体, 标签参考单用只在 DINO 下强。**GPU2 空**。dmisc **完**(experiment_log 21:20: v7_lowres seed1 一致; DDPM50 6.81 更好; DDIM 本身崩; 20px bk24/bk32 46.6/59.1 反向)。dmech2 **完**(experiment_log 09-08 02:00: 20/24px 信念 TV 模式复现 — 有效参考 TV 21.8/16.9/18.2 < 强 31/30, 无效 bk24@20 28.5, 有害 bk32@24 31.3; 32px 裸 96.85 → bk24 77.45 / autog 75.23 / 复合 69.27, bk48 113.93 反向; 排序在 16/20/24/32 全成立)。**GPU2 空**。probe_cc **完**(19:00): coarse w1.5 14.71(裸 20.63; 同权 bk12 9.43) → 方向证实/强度不足, 杀(组件)留作消融(experiment_log 19:05)。dseed2 **完**(三 seed: 裸 21.52±.47 / autog 8.73±.26 / bk12 8.52±.29 / 叠加 7.53±.19 / snaplo 7.63±.43; 见 experiment_log 17:35)。GPU2 `dseedR`(tmux, 17:36 UTC, logs/diag_seedR.log, DIAG_SEEDR_DONE): 20/24/12px 关键行 seed1 共 10 项(~25 min/项, ~4h)。论文素材已出: paper_assets/fig_qual_16px.png(6 行×24 同 seed 同 prompt)、fig_tv_vs_fd.png(7 点; 注意 uncond 点 TV 27 但 CFG 21.98 = 反例需在文中说明: 无条件不是同条件下的结构对齐信念)。引导侧第二组件 APG/FDG/区间调度**三连杀**; dmech/dres20/dres24 全完
- **勘误(19:50)**: "叠加"= 快照权重在 bucket:12 标签下的**单一复合弱参考**(NFE 2/步, 非双参考); snaplo 不省 NFE。论文统一措辞 composed weak reference。
- **direction**: **跨分辩率自引导**(同一多分辩率模型以"更低分辩率标签下的自己"为弱参考, 零训练) —— 当前最优 = 叠加 快照10k+bucket:12 w1.5 **7.67 / seed1 7.32**(裸 v7h 21.98/21.05; 地板 3.45; matched seed 方差 ≈0.3~0.5)。**训练版 probe_cg 杀**(见已证伪), 机制(dmech 完, 09-07 14:20): **有效弱参考 = 局部对比度(TV)低于强模型且结构对齐的自身信念**; 分辩率标签是单调对比度旋钮(bk12 21.6 / bk24 31.9 / bk64 40.9 vs 强 32.0); "简单性先验"假说证伪; 显式低通分支有害因引入块结构。**泛化成立**: 20px 45.9→bk16 32.9/叠加 29.7; 24px 79.8→bk16 60.4(自引导 53.4); 12px 高桶参考变差 14.45(方向性); 第二模型 v7_lowres 16.66→7.40。论文定位: 推理侧引导机制 + 机制分析, 需 20/24px 泛化 + 第二模型 + 第二组件才够 ICLR。
- **在跑**(tmux, 13:40 UTC 起, 每项 ~25 min):
  - GPU2 `dres24` → logs/diag_res24.log, 末行 DIAG_RES24_DONE: 24px matched@24(eval_matched_r.sh, fd_fair --size 24, 地板同算): v7h_r24_cfg4 / bk16_w2 / autog10k_w1p5 / bk20_w2 / bk12_w2 / bk16s10k_w1p5 / bk32_w2 + fd_decomp。**判据**: bk16 (或 bk12/bk20) 明显低于 cfg4 且 ≤ autog → 泛化成立; bk32 应无效(更高桶)。
  - GPU3 `dres20` → logs/diag_res20.log, DIAG_RES20_DONE: 20px: cfg4 / bk16 / bk12 / autog / bk16s10k; 12px: cfg4 / autog / bk16(更高桶, 预期无效—诚实限制); 然后 v7_lowres bucket:12 w2 @16(第二模型; 其裸 16.66 受记忆污染, 只看相对下降)。
  - GPU3 `dmech`(等 DIAG_RES20_DONE) → logs/diag_mech.log, DIAG_MECH_DONE: 纯弱参考样本 @16 (`--cfg 0`): bk12only / bk24only / snap10konly / uncond0 / bk64only → fd_fair + fd_decomp + stats_simplicity.py(色数/平区比/TV vs real16 / 块平均 real16 / 真实 native12/16/24)。**判据**: bk12only 色数少、flat 高、TV 不降(≈简单精灵) 而非 TV 大降(≈低通) → "条件诱导简单性先验"成立。已有基线统计: real16 ncol 34 flat .20 tv 30; v7h cfg4 ncol 73 flat .03 tv 32; bk12_w2 ncol 61 flat .045; real16 块平均 ncol 21 flat .61 tv 14; native12@16 ncol 5 flat .51 tv 28。
  - **GPU3 `dapg`(14:34 移到 GPU3, NOWAIT=1 立即跑; GPU2 与用户 vLLM 共卡慢)** → logs/diag_apg.log, DIAG_APG_DONE: **APG(Sadat 2024, sample_e `--apg eta,r,beta[,rgb]`, x0 空间投影, rgb 变体只投 RGB)** 叠在 stack(bk12+10k)上 w 1.5/2/3 + full/proj-only/momentum 消融 + 单参考 bk12/autog w2 + fd_decomp。基线 stack w1.5 7.67(seed1 7.32)。**判据**: 任一 apg 行 < 7.2 → 组件有信号, 补 seed1; 全 ≥ 7.5 → APG 无益, 试 FDG(频域分权)/区间调度。
  - 已出: FD@20 地板 12.14, v7h_r20_cfg4 45.92(20px 差距 33.8 > 16px 的 18.5)。每项 ~6 min。
- **在跑(15:20 起)**: GPU3 `probe_cc`(supervise.sh 直起, 15:20; **教训: 从 tmux 等待脚本里起 setsid nohup supervise 会静默死, 以后一律 ssh 直接起 + `< /dev/null`**; `OUT=workdir/probe_cc CG_ARGS="--degrade contrast --f 0.6" bash baseline/run_probe_cg.sh`; ~3.5 h 训 + 7 项评测; logs/probe_cc.log/.supervisor/.FAILING; 末行 CG_DONE)。dres24 **完**(bk32@24 = 101.67 反向对照, 高桶有害再证; DIAG_RES24_DONE)。GPU2 `dsched`(tmux, 15:25): 零训练组件 (a) 双参考区间 `--gi_snap`(快照仅 t/T∈[.2,.8]/[0,.5]/[.5,1]) (b) FDG 1 层 `--fdg w_low`(w_high=cfg 1.5/2/2.5, w_low 1.0/1.25) 共 9 项 + fd_decomp; logs/diag_sched.log, DIAG_SCHED_DONE; 判据同 APG: 任一 < 7.2 才算。
- **下一动作(cycle 8, 更新 09-08 02:20)**: ⓪ dclip **完**(experiment_log 09-08 02:15; paper_outline 表(g)/(g′); #8/#10/#16 DONE): CLIP 代价随 R 增大(−0.55@16 → −0.94@32), 与方法无关; q16@20/24/32 后复合仍最优但差距缩到 −20/−20/−8%。**发现(16px 3 seed 一致): 引导有小幅文本对齐代价** — 100cos 裸 CFG4 30.06 > 真实 29.80 > autog 29.58 ≈ 复合 29.51 > bk12 29.37; R@1/100 18.3/16.4/14.4/13.1%; autog 同样掉 → 是"w=2 参考引导替代了 w=4 CFG"的通病, 非本方法特有, 但要诚实报告并尝试修。GPU2 `dalign`(**02:49 发现 tmux 会话 dalign 连同已完成的 dclip 会话一起消失, 第 1 项采样中被静默杀; 02:57 改 supervise.sh 直起** `NEED_MB=8000 setsid nohup bash supervise.sh diag_align 2 bash baseline/diag_align.sh 2`, logs/diag_align.log/.supervisor/.FAILING, .done 标记可续跑; 教训: 一次性评测扫描也可能被 tmux 清扫, 以后长批次也走 supervise.sh; DIAG_ALIGN_DONE): 零训练修法 (a) `bucketu:12` = 错桶+无文本参考(2 NFE) (b) `--cfg_text wt` 附加 CFG 项(3 NFE), 7 项 @16 + CLIP + fd_decomp; 判据: CLIP 回到 ≥ 30.06 且 FD 不比对应未修行差 > 0.3。dclip 完 → experiment_log 写 CLIP 表 + q16@20/24/32, paper_outline 加表(g) 文本对齐(#10/#8 DONE); dalign 完 → 达标则主表复合行换带修法版(注明 NFE), 否则限制节写"对齐代价 ~0.5 CLIP 点/5% R@1, 与 autoguidance 相同"。#16 定性图 **完**(paper_assets/fig_qual_{12,16,20,24,32}px.png, make_qual_r.py 按 prompt 对齐; 旧 16px 图 real 行第 10 列后错位, 已修)。剩不需 GPU: #7 每分辩率 TV-vs-FD 图。 ① v7s **完**(见上; #4 DONE)。⓪′ diag_metric2 **完**(表 h)。⓪″ GPU2 空 → 排 #6 20/24px w 扫描(bk16 w 1.25/1.5/2.5/3 + autog w 1.25/2/2.5/3, 共 16 项, GPU2 慢 ~30 min/项)或等 diag_v7s2 完后在 GPU3 跑(4 min/项, 更划算) —— 选后者, GPU2 先排 #15 NFE/wall-clock 计时(4 项各 200 张即可)。①′ diag_v7s2 完 → v7s 两 seed 16px 表 + v7s 20/24px 行(第二模型泛化) + v7h 12/20/24 三 seed mean±sd → paper_outline 表(b)/(c) 更新, 提交。② dmisc 出数 → 采样器鲁棒性与 20px 反向对照入表(e)/(f)。③ 之后按 paper_outline §5 缺口清单继续(优先: #10 CLIP score 脚本(已存样本, 无需重采), #7 20/24px 纯低桶信念 TV 统计(bk16only@20/24), #8 20/24px q16 附表, #14 32/48/64px 适用范围, #16 12/20/24 定性图); 每次 GPU 空就排下一批。④ 论文正文起草(subagent): Intro/Method/Mechanism 各节初稿到 paper_assets/draft_*.md。旧:  ⓪ GPU2 空 → 排论文素材: 定性图(同 seed 同 prompt: v7h 裸/autog/bk12/stack @16, 各 32 张拼图)+ 关键行 seed2(stack w1.5, bk12 w2, autog w1.5, 裸 cfg4 @16)+ 快照仅 [0,.5] 省算力版可作默认; 不需要 GPU 的: 弱参考 TV vs FD 单调图(7 点, dmech 表)。① probe_cc: CG_DONE 后 coarse_w* vs 同权 bk12/自引导; 成立(< 同权 bk12 且 ≤7.67) → 机制预测被证实 = 论文核心证据(训练版 12px 也测: 排 eval_matched_r.sh 12/20/24 用 `--guide_mode coarse`); 不成立 → 机制解释需再修(看其 TV 统计), 论文仍以零训练版为主。② dres24 完 → 泛化表补全(bk32 反向对照)。③ GPU2 空后零训练: (a) 双参考分区间调度(快照仅中段 t, bk12 全程: 需 sample_e 加 `--gi_snap lo hi`), (b) FDG 1 层拉普拉斯分权 w_low<w_high; 各 3~4 项。④ 论文素材: 定性图 + 弱参考 TV vs FD 单调图(7 点) + seed2 关键行。
- **旧下一动作(cycle 7)**: ① 两个 dres 出数 → 写 experiment_log 分辩率泛化表(含地板@20/24/12), 判定泛化。② subagent 报告 → 写 arch_ideation_log cycle 7 RESEARCH, 选第二组件做零训练扫描(优先 APG: sample_e 加 `--apg` 投影, 叠在 bk12+10k 上, w 扫 1.5/2/3; 若 APG 允许更大 w 而不爆颜色 → 组件成立)。③ 机制实验(GPU 空后): `--cfg 0 --guide_mode bucket:12` 纯低桶信念样本 3000 张 @16 → FD + 统计(唯一色数/平区比例/TV) vs 真实 12px & 16px 精灵 & 块平均版; 同法 bucket:24 信念。写小脚本 src/v6/stats_simplicity.py。④ 若 20/24px 泛化失败 → 机制降为 16px 组件, 回 RESEARCH 换大方向(候选: 逃离均值回归的训练目标里剩余未试的分类式像素输出/AR 强基线)。⑤ 论文素材: 定性对比图(v7h 裸 vs 自引导 vs bk12 vs 叠加, 同 seed 同 prompt)。
- **更新时间**: 2026-09-08 05:15 服务器时(UTC)

## 历史(每 cycle 一行)
- cycle 0 (09-05~06): 有序离散 v_ord 探针 → 252.3 杀; 连续+TV/调色板双探针 → 旧指标 70.65/66.26 "杀"(**后证 TV 被误杀, 公平 FD 42.82 优于 v7 53.21**)。
- cycle 1 (09-06): 结构/调色板 oracle 诊断。probe_struct oracle 公平 FD 13.52 = 地板(难度全在结构); **发现并修正 FD 参考集管线不匹配**(fd_fair.py), 全表重测, 判据重定(<38 信号 / >45 杀)。
- cycle 2 (09-06): 两阶段 probe_sgen(只生成 S → 上色) **61.66 杀**; 结构域 FD 诊断: 阶段一 19.17 ≈ TV 18.94, 分解不降难度; 上色器曝光偏差。结构优先方向降级。TV w=0.3 → 46.57(甜点窄)。
- cycle 3 (09-06): RESEARCH 调色板因子化 x0 头(novelty 清) → BUILD train_palhead.py → probe_palhead **硬 78.34 / 软 52.09 杀**(每张仅 ~7 色, 头未改善底层预测)。色数控制实验: **tv+q16 事后量化 = 35.24 成新最强基线**, 甜点 15-20 色; 判据改为 +q16 < 32。"架构化少色瓶颈"方向 1 杀不再投; 剩余难度=结构。
- cycle 5 (09-06~07): 协议修正(matched, 记忆污染发现, v7h 干净基线 21.98); es 探针两杀(28.47/20.69 vs ctrl 15.03); 零训练引导扫描发现自引导 −13 点(8.98)。
- cycle 6 (09-07): 文献占位图 → 跨分辩率自引导 bucket:12 8.59 / 叠加 7.67(seed1 7.32; 零训练, 分辩率单调, 互补); 通道解耦无信号; BUILD probe_cg 训练版 → **杀**(21.5/35/59, 显式低通分支非正确弱参考); 机制解释改为条件诱导简单性先验。
- cycle 7 (09-07~): EVAL 20/24/12px 泛化 + 第二模型 v7_lowres; RESEARCH 第二组件(APG/CFG++/…)与 novelty 撞车。
- cycle 4 (09-06): 控制 32→16 BOX: v7 84.07 / tv 52.82(由细到粗更差, loop-F 降级); 三路调研 → "投影入环自条件" probe_selfq **42.55/+q16 38.45 杀**(配对对照 46.91/39.91, 置零消融 47.23/37.51: 效应=少色, 量化后消失); 暴露 run 间方差 ±4。"离散/少色假设族"5 探针全 ≤ 事后量化 → 整族标废。
- cycle 5 (09-06~): 诊断三连: FD 分解(差距=coverage) → prompt 错配(旧协议作废, matched 协议) → 记忆污染(v7 matched 16.66 不可信, 8.4% 复现) → **重训干净基线 v7h**(排除评测集, 随机初始化, 快照); 零训练扫描: CFG7 有益, DDIM/以 v7 为弱模型的自引导无效; 调研综合: 能量分数随机去噪 / CADS+自引导 / 非泄漏增广+调度平移。
