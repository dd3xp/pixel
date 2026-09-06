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
| probe_struct oracle(真 S=[alpha,4级明度]) | 13.52 (52.14) | = 413 源地板 → 给定 S 上色已解决; 难度全在 S |
| v6f 朴素离散 absorbing | (160.7, 未重测) | 崩坏, 离散劣于连续 |
| v_ord 有序调色板离散 | (252.3, 未重测) | 更差 |
| SD-πXL(SDS优化式) | 极低分辩率崩 | 对照基线 |
教训: ①离散调色板整族作废; ②"手工先验没用"是旧指标误判, 硬边/分片恒定先验有效但 TV 本身无 novelty, 只能当基线; ③结构 S 决定一切 → 生成结构才是问题核心; ④先验证指标再解读 Δ。

## 循环协议(每 cycle)
1. **RESEARCH**: 起 2-3 个 subagent 并行——(a)该方向文献机制/怎么做的, (b)novelty 撞车检查(最像的 3 篇+相似度), (c)极低分辩率可行性/坑。汇总写 arch_ideation_log.md。判定: 有新机制且未撞车 → BUILD; 否则 → 下一个候选。
2. **BUILD**: 写探针代码(优先复用 train_v7/train_probe/sample_e/eval_probe 骨架), 服务器上冒烟 200 步, 确认 loss 正常、能采样。
3. **TRAIN**: supervise.sh+tmux 起在空闲 GPU, 日志 logs/<name>.log。cron 每 15 分只报一行进度; .FAILING 则诊断修复重启(同错 2 次则杀)。
4. **EVAL**: 训练完 → eval_probe.sh(或侧条件探针 eval_cond.sh) 16px 采样 3304 张 + 公平 FD; 结果 runs_out/<name>_fd.json(字段 fair_fd16), 汇总在 runs_out/fair_fd16.json。
5. **DECIDE**: <38 → 深挖(消融/放大/12&24px 也测/写论文级图); 否则杀, 记入"已证伪", 回到 1 取下一候选。
**不死磕**: 同一"大方向"连续 2 个探针都 >45 → 整个大方向标废, 换大方向。

## 候选队列(按优先级; 调研后可重排/增删)
1. **[结构优先 stage-1] 用户点名方向, oracle 已证"难度全在 S"**: 探针 probe_sgen = v7 微调成"只生成 S"(S 编码为 RGBA: R=G=B=明度级, A=alpha, 20k 步), 采样 → 量化回 5 态 S → 喂 probe_struct 上色 → 公平 FD。赢 42.82 才有信号。S 是 5 态离散低熵模态, 正是文献里"两阶段能降 FID"的条件; 后续消融 S 信息量(2 级/仅 alpha)找最低熵仍有增益的结构表示 → 这是机制贡献("需要多少结构才够")。novelty 风险: 两阶段本身撞车, 需靠"极低分辩率+低熵结构表示+消融曲线"或"模态非对称噪声调度单模型"(结构通道快调度、颜色慢调度)撑新意。
2. **[加强基线] TV 权重扫描** w=0.3/1.0(probe_tv3/probe_tv10): 廉价, 必须知道平凡先验能走多远, 新机制的增益才诚实。与 1 并行填 GPU2。
3. **[逐尺度一致性/多分辩率联合] loop-F**。
4. **[区域图生成] loop-O**: 与 1 合并——S 就是区域图的粗版。
5. **[精确似然/EBM] loop-E**; 6. **[宽泛再调研]**。

## 当前状态
- **cycle**: 1 → 2 过渡
- **phase**: TRAIN (GPU3 = probe_sgen 已起 07:55 UTC, ~4h; GPU2 = probe_paltok 收尾 17k/20k)
- **direction**: 结构优先 stage-1(候选 1) + TV 加强基线(候选 2)
- **GPU**: GPU3 = probe_sgen(logs/probe_sgen.log, workdir/probe_sgen); GPU2 = probe_paltok(约 08:20 UTC 完)
- **已完成本 cycle**: probe_struct oracle 公平 FD 13.52 = 地板 → 结构决定一切。**指标修正**(fd_fair.py)已落地, eval_probe.sh/eval_cond.sh 已切换, experiment_log 已记。
- **待办 paltok**: 训练完 → `tmux new-session -d -s ev_paltok "setsid nohup bash supervise.sh eval_probe_paltok 2 bash baseline/eval_cond.sh probe_paltok 2 </dev/null >/dev/null 2>&1 & disown; sleep 5"` → 读 runs_out/probe_paltok_fd.json(fair_fd16; 地板 13.82, v7 53.21)。预期: 调色板信息量远小于 S, oracle 会明显高于 13.5; 记录"调色板 vs 结构谁是瓶颈"即可, 不改方向。
- **cycle 2 BUILD 计划**:
  - GPU3: **probe_sgen** = 从 v7 微调 20k 步生成 S-as-RGBA(R=G=B=4 级明度 ∈{0,85,170,255}, A=alpha; 训练目标 = to_tensor(x) 经 make_struct 再编码), 采样 3304 张 → 量化回 S → 用 workdir/probe_struct 上色 → 公平 FD。代码: src/v6/train_sgen.py(复用 train_cond 骨架, 只改 target) + src/v6/sample_twostage.py。
  - GPU2(paltok 完后): **probe_tv3** = train_probe.py --probe tv w=0.3(加强基线)。
- **已 BUILD 并冒烟通过**: src/v6/train_sgen.py(v7 微调生成 S-as-RGBA), src/v6/sample_twostage.py(sgen→量化 S→probe_struct 上色), baseline/run_probe_sgen.sh, baseline/run_probe_tv3.sh(W 默认 0.3), baseline/eval_twostage.sh <sgen_name> <gpu> [color=probe_struct]。
- **下一动作**: ① paltok 完 → GPU2 起 eval_cond(见上) → 读 fair_fd16 记录; ② eval 完 GPU2 空 → `tmux new-session -d -s px_tv3 "setsid nohup bash supervise.sh probe_tv3 2 bash baseline/run_probe_tv3.sh </dev/null >/dev/null 2>&1 & disown; sleep 5"`; ③ probe_sgen 完(PROBE_SGEN_DONE) → 看 workdir/probe_sgen/samples/step_020000_s16.png(行1 真 S/行2 生成/行3 量化) → `tmux new-session -d -s ev_sgen "setsid nohup bash supervise.sh eval_probe_sgen 3 bash baseline/eval_twostage.sh probe_sgen 3 </dev/null >/dev/null 2>&1 & disown; sleep 5"` → runs_out/probe_sgen_fd.json → DECIDE(<38 信号; 38-45 持平; >45 杀)。
- **更新时间**: 2026-09-06 07:58 UTC

## 历史(每 cycle 一行)
- cycle 0 (09-05~06): 有序离散 v_ord 探针 → 252.3 杀; 连续+TV/调色板双探针 → 旧指标 70.65/66.26 "杀"(**后证 TV 被误杀, 公平 FD 42.82 优于 v7 53.21**)。
- cycle 1 (09-06): 结构/调色板 oracle 诊断。probe_struct oracle 公平 FD 13.52 = 地板(难度全在结构); **发现并修正 FD 参考集管线不匹配**(fd_fair.py), 全表重测, 判据重定(<38 信号 / >45 杀)。
