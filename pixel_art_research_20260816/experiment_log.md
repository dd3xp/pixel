# 实验记录

> 归档原来只存了调研结论，缺实验元数据。此文件补记，以后每个实验往下追加一条。

---

## 环境（2026-08-16 实地核实）

- **服务器**: emnlp（ssh alias；223.72.199.116:60022, user kw）= **a100-node03，8× A100-SXM4-80GB**
- **项目目录**: `/mnt/data/kw/RoundSquisheen/texture/SD-piXL`（SD-piXL 官方仓库 clone）
- **conda 环境**: `/mnt/data/kw/anaconda3/envs/SD-piXL`
- **启动方式**: `cd .../SD-piXL && nohup env HF_ENDPOINT=https://hf-mirror.com .../envs/SD-piXL/bin/python -u main.py -c config_petal_db32.yaml >> logs/petal_db32_train.log 2>&1 &`
- **输出目录**: `workdir/<config名>/<时间戳-sdxl-im16x16-dawnbringer32>/`
- **注意**: 另有一个无关的 LLM 训练服务（ga_vllm 环境, torchrun 8卡）常驻，每张卡占 ~15.8GB——GPU1-7 并非完全空闲，可用显存约 65GB/卡
- **工作流**: 本地（c:\Codes\pixel）开发 → 同步到服务器运行；只有服务器有 GPU

## 实验条目

### EXP-001 | 2026-08-16 | SD-πXL, 4色 slowly.hex
- 目标: 蓝白色花瓣图 → 16×16 像素画
- 配置: 16×16, slowly.hex (4色), 10000步, guidance_scale=40
- 输出: `workdir/config_petal/2026-08-15-06-12-00-sdxl-im16x16-slowly/final_argmax.png`（此前 8-14 晚有约 11 次短命重启目录）
- 结果: **失败**（已确认最终图）— 几乎全为灰底，仅中下部几个白色块 + 1 个深蓝块，完全不成形状

### EXP-002 | 2026-08-16 | SD-πXL, 32色 DawnBringer32
- 目标: 蓝白色花瓣 → 16×16 像素画
- **prompt**: `a single flower petal, blue and white, simple, centered`（negative prompt 为空）
- 配置: 16×16, `assets/palettes/lospec/dawnbringer32.hex`, 10001步, guidance_scale=40.0, seed=0, teacher=SDXL, 初始化 palette-bilinear, init_distance=l1, lr=0.025
- 配置文件: `config/config_petal_db32.yaml`；日志: `logs/petal_db32_train.log`
- 输出: `workdir/config_petal_db32/2026-08-16-03-47-44-sdxl-im16x16-dawnbringer32/`（03-21-56 和 03-22-41 两个目录是夭折的启动残留）
- 中间结果: `png_logs/` 每 50 步存 `{step}_hard/soft.png`；checkpoint 每 5000 步
- 状态: GPU0 运行中（优化自 03:47 起；05:45 时已过 5000/10001 步 = 50%）
- **step 5000 中间观察**（样本已拉到本地 samples/）: 32色明显好于4色——已有浅蓝花体+蓝色花心+灰蓝背景的可读结构；但**花瓣轮廓未解析出来**，整体是菱形色块，2500→5000 变化很小（基本收敛）
- **关键发现（step 0→1000 退化序列，samples/db32_0/200/1000_hard.png）**: 初始化（palette-bilinear，即朴素缩小+量化）本来有白色高光和多档色阶；SDS 优化在 ~1000 步内把渐变/高光/光影全部磨平，塌缩成两档色的平色块。**单尺度 SDS 不只是保不住结构，而是主动破坏已有细节。** 机制: ① 16→1024 bilinear 放大+VAE 使像素级明暗对 SDS 梯度不可见/被当噪声惩罚 ② distorsion_prob=0.5 等增强使精细结构不稳定 ③ CFG=40 模式寻求 ④ fft 损失只锚低频。→ 论文 motivation 证据链：锚定机制是必需品，不是加分项
- 预期: 32色应足以表达花瓣曲线；CFG=40 偏高，可能过饱和（VSD/ISD 文献警示）
- ⚠️ **发现**: prompt 写的是 "a single flower **petal**"（单片花瓣），但 SDXL 选中的参考图 `select_sample.png` 是**一整朵花**（蓝心白瓣、蓝灰背景）——优化目标与字面意图不符，如确需"单片花瓣"需改 prompt 重跑
- 结果: **完成**（final 在 samples/exp002_final.png）。与 step 5000 几乎无差——**约 2500-5000 步即收敛，后半程步数浪费**。最终图：干净但平板，~4-5 色，无白色高光，无花瓣结构，蓝色方形花心

### P0 对比结论（EXP-002 vs EXP-003，同参同源图）
| | EXP-002 (SD-piXL) | EXP-003 (自研) |
|---|---|---|
| 耗时 | ~3.5h | **~50min** |
| 画面 | 干净、平板、4-5色、无白 | 层次多、9色、白色保住、有杂散像素 |
| 共同点 | **都无花瓣结构、同等"潦草"** | 同左 |
- **P0 出口判据达成**：自研实现行为正确（同族结果，差异可由已知实现差解释）
- 三个带走的科学结论：① 16×16 单尺度 SDS 的质量天花板与实现无关 ② 重增强（distortion/grayscale）是磨平细节的主力之一 ③ 万步预算后半程浪费（自适应停止有据可依）

> 另: `workdir/config_petal/`（8-15）应为 EXP-001 的 4 色输出；`logs/petal_train.log`、`logs/train_petal.log`（8-14）为早期尝试日志

### EXP-003 | 2026-08-16 | 自研代码 P0 复刻（对照 EXP-002）
- 目的: 验证从零实现的 SDS + Gumbel-softmax 渲染器行为正确（P0 出口判据）
- 代码: 本仓库 `src/` + `scripts/run.py`（commit a5c4f17）；100 步冒烟测试已通过
- 配置: `configs/petal_16_db32_full.yaml` = 与 EXP-002 同参（16×16, DB32, 10000步, CFG 40, lr 0.025, 同一张蓝白花源图），差异: 无 ControlNet、增强只有 hflip（无 distortion/grayscale）
- 运行: emnlp GPU1, PID 357089, 日志 `pixel/logs/exp003_replication.log`, 输出 `pixel/workdir/petal_16_db32_full/<时间戳>/`
- 判据: final_hard 与 EXP-002 final 定性相近（同等"潦草"即算复刻成功）；metrics.csv 应复现出细节退化曲线（colors_used/lum_levels 随步数下降）
- 结果: **完成**（06:10 启动，~50 分钟跑完 10000 步——远快于 SD-piXL 的 3.5h，因无 ControlNet/双渲染）。样本在 samples/exp003_*.png，指标在 samples/exp003_metrics.csv
- **定性**: 与 EXP-002 同族——中心花团 + 花心结构，同量级的"潦草"，行为复刻成立。差异：背景从灰蓝漂成饱和蓝（CFG=40 过饱和更明显）、有零散杂色像素（绿/青/灰）
- **定量意外**: 我们的版本 colors_used 5→9、lum_levels 5→7（不降反升），白色保住了——与 EXP-002 的塌缩相反。解释：EXP-002 的 distortion/grayscale 增强是把细节磨平的主力，我们只有 hflip。**增强强度 = 杂散像素鲁棒性 vs 细节保留的权衡**——这是 P2 锚定设计的重要输入（EXP-002 的"细节被抹掉"部分归因于增强，不全是 SDS 本身）

### EXP-004 | 2026-08-16 | P1 多尺度金字塔（go/no-go 实验）
- 目的: 验证多尺度渐进是否优于单尺度（P1 出口判据）
- 配置: `configs/petal_pyramid_db32.yaml` — 与 EXP-003 唯一差异是尺度调度：256→128→64→32→16 五级，步数 [400,300,250,150,150]（共 1250 步 vs 单尺度 10000 步），无锚定
- 运行: emnlp GPU1，总耗时 **~6 分钟**；输出 `workdir/petal_pyramid_db32/2026-08-16-14-53-46/`，样本在 samples/exp004_*.png
- **结果: GO——多尺度明显胜出**：
  - **花瓣结构保住了**：16×16 final 有清晰的花瓣凸起轮廓（vs EXP-002/003 的菱形色块）
  - **背景忠实**：保持参考图的灰蓝（vs EXP-003 漂成饱和蓝）——高分辨率级先固定了内容，低分辨率级来不及漂移
  - **无杂散像素**（vs EXP-003 的绿/灰杂点），同时保留青色高光等层次
  - 64×64 级的结果尤其好：花瓣、花心、光影俱全，说明"逐级压缩"传递的信息质量高
- 局限（诚实记录）: 单一 case、单 seed；16×16 相比 32×32 级仍损失明显；步数分配未调优
- **P1 出口判据达成**。方案核心假设成立，进入 P2（锚定机制）

### EXP-005 | 2026-08-16 | 步数分配消融：低分辨率级加重
- 假设: 32/16 级的花瓣断裂能靠多给低分辨率级步数由 SDS 自行修复
- 配置: 与 EXP-004 唯一差异 steps_per_scale [400,300,250,150,150]→[200,200,250,300,400]
- **结果: 假设否定**（samples/exp005_final.png）——16×16 在 400 步里**重新塌回色块**：花瓣凸起比 EXP-004 弱、花心变大偏移、左上角出现杂散像素。32 级与 EXP-004 相当（断裂主因在降采样不在步数）
- **结论**: ① 低分辨率 SDS 步数是"侵蚀结构"的方向，不是修复方向（与 EXP-003 一致）② 断裂/粘连的根因是结构无关的降采样，必须靠 P2 锚定+结构感知机制解决 ③ 步数分配应保持头重脚轻（EXP-004 的 [400,300,250,150,150] 更优）

---

**下一步**: P2 内容锚定——多图多 seed 扩大验证 + 锚定损失变体矩阵（参见 project_plan.md）
