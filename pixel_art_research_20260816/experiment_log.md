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
- 结果: （待填）

> 另: `workdir/config_petal/`（8-15）应为 EXP-001 的 4 色输出；`logs/petal_train.log`、`logs/train_petal.log`（8-14）为早期尝试日志

---

**下一步（来自 novelty_assessment.md）**: 阶段0验证实验——同一批图，多尺度 vs 单尺度 SD-πXL 直接对比质量
