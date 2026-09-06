# 自主推进状态 (cron 每次先读此文件, 再行动, 做完更新此文件)

用户 2026-09-06 授权: 用户几天不在, Claude 自行推进"调研→架构→探针→不行就换"循环, 宽泛调研不死磕。
目标不变: ICLR 论文, 核心=极低分辨率(12-24px)像素画生成的**新机制/新架构**, 客观指标要赢过 v7。

## 判据与硬约束
- **指标(2026-09-06 修正)**: **公平 FD-DINOv2@16px**(src/v6/fd_fair.py, 参考集与生成同走 to_tensor 管线; 旧 fd_dino 数值全部作废, 见 experiment_log "指标修正")。n=3304(413 prompts×8, eval_probe.sh)。**真实地板 3.45, v7 = 53.21, 最强简单基线 probe_tv(TV w=0.1) = 42.82**, 越低越好。
- **胜负线**: 新架构必须同时赢 v7 和 probe_tv: 探针公平 FD **< 38** 才算"有信号"深挖; 38-45 持平记录; **> 45 杀**。oracle 类探针(eval_cond.sh, 413 源)地板是 13.82, 单独解读。
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
| probe_struct oracle(真 S=[alpha,4级明度]) | 13.52 (52.14) | ≈413 源地板; **数值被记忆污染**(源在训练集里), 只取定性: 难度全在 S |
| probe_paltok oracle(真 8 色调色板 token) | 18.18 | **记忆污染实证**: 无空间信息却空间复现 → oracle 源必须剔除训练 |
| **probe_sgen 两阶段(v7微调只生成 S → probe_struct 上色)** | **61.66** | **杀**; 阶段一结构域 FD 19.17(v7 23.55, tv 18.94, 地板 1.40): 只学低熵 S 也没变容易; 上色器对生成 S 曝光偏差崩 |
| v6f 朴素离散 absorbing | (160.7, 未重测) | 崩坏, 离散劣于连续 |
| v_ord 有序调色板离散 | (252.3, 未重测) | 更差 |
| SD-πXL(SDS优化式) | 极低分辩率崩 | 对照基线 |
教训: ①离散调色板整族作废; ②"手工先验没用"是旧指标误判, 硬边/分片恒定先验有效但 TV 本身无 novelty, 只能当基线; ③结构 S 决定一切, **但把 S 分解出来单独生成并不更容易**(sgen 19.17 vs 地板 1.40), 两阶段还引入曝光偏差 → "先结构再上色"大方向 1 杀+反证, 仅留"单模型非对称调度"当低优先备胎; ④先验证指标再解读 Δ; ⑤结构域 FD(fd_struct.py)显示 TV 也是结构最好的 → 短板是**局部离散结构(分片恒定/硬边/少色)**, 不是全局布局。

## 循环协议(每 cycle)
1. **RESEARCH**: 起 2-3 个 subagent 并行——(a)该方向文献机制/怎么做的, (b)novelty 撞车检查(最像的 3 篇+相似度), (c)极低分辩率可行性/坑。汇总写 arch_ideation_log.md。判定: 有新机制且未撞车 → BUILD; 否则 → 下一个候选。
2. **BUILD**: 写探针代码(优先复用 train_v7/train_probe/sample_e/eval_probe 骨架), 服务器上冒烟 200 步, 确认 loss 正常、能采样。
3. **TRAIN**: supervise.sh+tmux 起在空闲 GPU, 日志 logs/<name>.log。cron 每 15 分只报一行进度; .FAILING 则诊断修复重启(同错 2 次则杀)。
4. **EVAL**: 训练完 → eval_probe.sh(或侧条件探针 eval_cond.sh) 16px 采样 3304 张 + 公平 FD; 结果 runs_out/<name>_fd.json(字段 fair_fd16), 汇总在 runs_out/fair_fd16.json。
5. **DECIDE**: <38 → 深挖(消融/放大/12&24px 也测/写论文级图); 否则杀, 记入"已证伪", 回到 1 取下一候选。
**不死磕**: 同一"大方向"连续 2 个探针都 >45 → 整个大方向标废, 换大方向。

## 候选队列(按优先级; 调研后可重排/增删)
1. **[架构化少色/分片恒定机制] 调色板因子化去噪头(palette-factorised x0 head)**: 证据链 = TV 是目前唯一有效手段(42.82, 结构域也最好), 短板是局部离散结构。把它做成**架构瓶颈而非损失**: 去噪网络在 x0 参数化下输出 (a) 每图 K 个颜色(全局 token 头, K≈8-16) + (b) 每像素对 K 的 logits, x0 = softmax(logits/τ)·palette(τ 随训练退火/随 t 调度), eps 由 x0 反推, 损失不变。任何样本天然少色+分片恒定+硬边, 且调色板可解释/可编辑(论文卖点)。与已废的 probe_pal(软吸附**损失**, 无效)和 v_ord(离散扩散)不同: 连续噪声+离散输出流形。风险: softmax 平均出灰色(需 τ 退火/straight-through)、K 固定; novelty 撞车检查: differentiable colour quantisation(ColorCNN)、VQ-x0、"discrete-continuous" 扩散、PaletteNet。RESEARCH 后 BUILD: 从 v7 微调 20k, 只换 conv_out 头 → probe_palhead。
2. **[加强基线] TV 权重扫描** w=0.3(probe_tv3, 评估中)/1.0(probe_tv10, GPU3 在训): 平凡先验能走多远, 新机制的增益才诚实。
3. **[逐尺度一致性/多分辩率联合] loop-F**。
4. **[区域图生成] loop-O**(区域=颜色分片, 与 1 一脉; 若 1 有信号可作其"显式区域"升级)。
5. **[结构优先备胎] 单模型模态非对称噪声调度**(S 通道快调度): 前提已被 sgen 阶段一削弱, 仅当 1/3/4 都死再考虑。
6. **[精确似然/EBM] loop-E**; 7. **[宽泛再调研]**。

## 当前状态
- **cycle**: 3
- **phase**: RESEARCH(候选 1 调色板因子化去噪头) ‖ 后台 EVAL(probe_tv3, GPU2) + TRAIN(probe_tv10, GPU3, 10:50 起, 约 14:30 完)
- **direction**: 架构化少色/分片恒定机制(候选 1); 结构优先方向已判 1 杀+反证, 降级为候选 5
- **GPU**: GPU2 = eval_probe_tv3(logs/eval_probe_tv3.log → runs_out/probe_tv3_fd.json), 之后空闲; GPU3 = probe_tv10(logs/probe_tv10.log, TV w=1.0)
- **cycle 2 结果**: probe_sgen 两阶段 61.66 杀(见 experiment_log 09-06 "probe_sgen 判决"); 新诊断工具 src/v6/fd_struct.py(结构域 FD: --gen S-as-RGBA 目录 / --struct_of RGBA 目录, 结果 runs_out/fair_fd16_struct.json)。
- **下一动作**:
  ① RESEARCH(本 tick 或下 tick): 起 3 个 subagent——(a) 机制: 可微颜色量化/调色板因子化输出头如何训得稳(τ 退火、straight-through、K 选择、x0 vs eps 参数化下如何接入 DDPM), (b) novelty: 最像的 3 篇(ColorCNN/可微 k-means 量化、VQ/离散-连续混合扩散、PaletteNet、pixel-art 生成里的调色板头), (c) 可行性: 16px、v7 微调 20k 是否够, 潜在失败模式(灰色平均、调色板塌缩)与对策。汇总写 arch_ideation_log.md, 判定 BUILD 或换候选 3(loop-F)。
  ② BUILD(若通过): src/v6/train_palhead.py(复用 train_probe 骨架; UNet conv_out → K 路 logits + 全局调色板头; x0 参数化损失或由 x0 反推 eps), sample_palhead.py 或让 sample_e.py 兼容; 冒烟 200 步; 起 GPU2 → probe_palhead → eval_probe.sh → DECIDE(<38/38-45/>45)。
  ③ 背景: probe_tv3 fd.json 出 → 若 <42.82 更新"最强简单基线"及 patrol_prompt.txt/eval_probe.sh 里的 tv_fair; probe_tv10 完(PROBE_TV3_DONE 字样, 日志 logs/probe_tv10.log) → `tmux new-session -d -s ev_tv10 "setsid nohup bash supervise.sh eval_probe_tv10 3 bash baseline/eval_probe.sh probe_tv10 3 </dev/null >/dev/null 2>&1 & disown; sleep 5"`。
- **更新时间**: 2026-09-06 10:55 服务器时(UTC)

## 历史(每 cycle 一行)
- cycle 0 (09-05~06): 有序离散 v_ord 探针 → 252.3 杀; 连续+TV/调色板双探针 → 旧指标 70.65/66.26 "杀"(**后证 TV 被误杀, 公平 FD 42.82 优于 v7 53.21**)。
- cycle 1 (09-06): 结构/调色板 oracle 诊断。probe_struct oracle 公平 FD 13.52 = 地板(难度全在结构); **发现并修正 FD 参考集管线不匹配**(fd_fair.py), 全表重测, 判据重定(<38 信号 / >45 杀)。
- cycle 2 (09-06): 两阶段 probe_sgen(只生成 S → 上色) **61.66 杀**; 结构域 FD 诊断: 阶段一 19.17 ≈ TV 18.94, 分解不降难度; 上色器曝光偏差。结构优先方向降级。TV 扫描 w=0.3/1.0 在跑。
