# 自主推进状态 (cron 每次先读此文件, 再行动, 做完更新此文件)

用户 2026-09-06 授权: 用户几天不在, Claude 自行推进"调研→架构→探针→不行就换"循环, 宽泛调研不死磕。
目标不变: ICLR 论文, 核心=极低分辨率(12-24px)像素画生成的**新机制/新架构**, 客观指标要赢过 v7。

## 判据与硬约束
- **指标**: FD-DINOv2@16px vs 真实精灵(n≈3304, 237 prompts×8+, eval_probe.sh 配方), **v7 基线 = 64.80**, 越低越好。
- **胜负线**: 探针 FD **< 61**(比 v7 好 ≥3.8, 约6%)才算"有信号"值得深挖; 61-65 算持平(记录, 不深挖); >65 直接杀。
- **一个探针最多 ~1 天**(训练+采样+FD)。超时未出结果 → 杀掉记原因换下一个。
- **每轮最多 2 探针并行**(GPU2 + GPU3)。优先"从 v7 断点微调 20k 步"的廉价形式; 只有机制上必须从头训时才从头(≤40k步)。
- **硬规则**: 只用 node03(ssh emnlp) GPU2/GPU3; **node09(kw) 一律不碰**(ljq的); 共享账号只在 /mnt/data/kw/RoundSquisheen/pixel/pixel 内读写; 后台任务必走 supervise.sh + tmux; node03 需 PYTHONNOUSERSITE=1; 判断在跑看产出增长。
- **每个有结论的步骤**: 写 experiment_log.md(结果) / arch_ideation_log.md(想法与调研), 更新本文件, `cd c:/Codes/pixel && git add -A && git commit && git push`。

## 已证伪/已堵死(别再回头)
| 方向 | FD@16 | 结论 |
|---|---|---|
| v7 连续原生前馈(基线) | **64.80** | 至今最强 |
| v6f 朴素离散 absorbing | 160.7 | 离散远劣于连续 |
| v_ord 有序调色板离散(OKLab序高斯核) | 252.3 | 更差, 承重机制反向 |
| probe_pal 连续+软调色板吸附 w=0.1 | 66.26 | 持平略差 |
| probe_tv 连续+TV平坦先验 w=0.1 | 70.65 | 明显有害 |
| SD-πXL(SDS优化式) | 极低分辩率崩(去风险#1已证) | 作为对照基线保留 |
教训: 离散调色板整族作废; 单纯往 v7 上加手工像素先验 loss 不行; 需要的是**改变生成的分解/条件结构**, 不是改 loss 项。

## 循环协议(每 cycle)
1. **RESEARCH**: 起 2-3 个 subagent 并行——(a)该方向文献机制/怎么做的, (b)novelty 撞车检查(最像的 3 篇+相似度), (c)极低分辩率可行性/坑。汇总写 arch_ideation_log.md。判定: 有新机制且未撞车 → BUILD; 否则 → 下一个候选。
2. **BUILD**: 写探针代码(优先复用 train_v7/train_probe/sample_e/eval_probe 骨架), 服务器上冒烟 200 步, 确认 loss 正常、能采样。
3. **TRAIN**: supervise.sh+tmux 起在空闲 GPU, 日志 logs/<name>.log。cron 每 15 分只报一行进度; .FAILING 则诊断修复重启(同错 2 次则杀)。
4. **EVAL**: 训练完 → eval_probe.sh 式 16px 采样 3304 张 + FD; 结果 runs_out/<name>_fd.json。
5. **DECIDE**: <61 → 深挖(消融/放大/12&24px 也测/写论文级图); 否则杀, 记入"已证伪", 回到 1 取下一候选。
**不死磕**: 同一"大方向"连续 2 个探针都 >65 → 整个大方向标废, 换大方向。

## 候选队列(按优先级; 调研后可重排/增删)
1. **[结构→上色 两阶段] 用户点名方向**: 先生成目标分辩率的**结构**(线稿/轮廓/区域分割/灰度形状), 再条件化**上色**。用户扭转: "压缩线稿"——高分线稿逐级下采样蒸馏到目标分辩率, 避开彩图下采样的色彩混叠, 再在低分线稿上着色。需查 novelty(H那半撞 2305.18387 的部分要绕开)。探针形式: 阶段1 = v7 骨架生成 1ch 结构图; 阶段2 = 以结构图为额外输入通道的 v7 上色(可从 v7 微调)。
2. **[逐尺度一致性/多分辩率联合] loop-F**: 同一物体 12/16/24 联合生成并强制跨尺度一致(下采样一致 loss 或共享 latent)。机制: 把"多分辩率"从条件变成结构约束。
3. **[区域图生成] loop-O**: 先生成"区域邻接图/色块布局"再填色, 像素画本质是少量色块的拓扑。风险高(离散结构), 但与"结构→上色"可合并成一条线。
4. **[精确似然/EBM] loop-E**: 极低分辩率像素空间小到可以做近似精确密度/能量模型, 连续扩散在 12px 上"杀鸡用牛刀"。新机制潜力大但工程风险高, 排后。
5. **[宽泛再调研]** 若 1-4 全灭: 起 subagent 宽泛扫 2024-2026 低分辩率/精灵/图标/emoji/字体字形生成、离散图像 token 化、结构化图像生成, 找新分解方式。

## 当前状态
- **cycle**: 1
- **phase**: TRAIN
- **direction**: 候选 1 结构→上色 —— 先做 **oracle 诊断**(调研结论: 两阶段本身 novelty 弱, 只有 stage-1 模态是 v7 的真瓶颈时才值得建; 详见 arch_ideation_log.md "cycle 1 调研"): 把真实精灵的侧条件喂给 v7 微调, 看上限。
- **GPU**: GPU3 = probe_struct, GPU2 = probe_paltok (2026-09-06 03:20 启动, ~200 步/分, 20000 步 ≈ 1.7h)
- **在跑探针**: `probe_struct`(结构图 S=[alpha, 4 级 OKLab 明度草图] 作输入通道 conv_in 4→6 零初始化), `probe_paltok`(真实 8 色调色板 → PalTok 8 token 拼到 CLIP 77 token 后)。均 src/v6/train_cond.py 从 v7 微调, 日志 logs/probe_struct.log / logs/probe_paltok.log, ckpt workdir/<name>/model_latest.pt。
- **评估**: **必须用 `bash baseline/eval_cond.sh <name> <gpu>`**(不是 eval_probe.sh): 从 413 张 held-out 真实精灵(与 FD 参考集不交)取侧条件+各自 caption, 每张采 8 = 3304 张 16px, FD → runs_out/<name>_fd.json (`"oracle": true`)。训练完 GPU 一空就 tmux 起它。
- **判定(oracle 专用, 不套 <61 规则)**: oracle FD ≪ 64.8(比如 <55) → 该模态是 v7 瓶颈, 下一 cycle BUILD 该模态的 stage-1 生成器(或"模态非对称噪声调度"单模型: 结构通道快调度、颜色通道慢调度, 这是绕开两阶段撞车的新机制候选); ≈64.8 或更差 → 该模态无 headroom。两者都 ≈64.8 → 结构→上色整方向杀, 取候选 2。
- **下一动作**: cron 报进度; `[20000/20000]` + `PROBE_*_DONE` 后 → `tmux new-session -d -s ev_<name> "setsid nohup bash supervise.sh eval_<name> <gpu> bash baseline/eval_cond.sh <name> <gpu> </dev/null >/dev/null 2>&1 & disown; sleep 5"` → 看 runs_out/<name>_fd.json + grid_s16.png(第 1 行真实参考, 第 2 行生成, 肉眼确认结构/调色板是否被遵循) → DECIDE。
- **更新时间**: 2026-09-06 03:25 (探针启动)

## 历史(每 cycle 一行)
- cycle 0 (09-05~06): 有序离散 v_ord 探针 → 252.3 杀; 连续+TV/调色板双探针 → 70.65/66.26 杀。纯 v7 最强。
