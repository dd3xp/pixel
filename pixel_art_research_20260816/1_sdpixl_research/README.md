# SD-πXL论文深度调研

**调研时间**: 2026-08-16  
**调研方法**: deepresearch_sop（4节点并行）

---

## 📁 文件结构

```
dr_sdpixl_related/
├── README.md              # 本文件（调研索引）
├── final_report.md        # 综合报告（完整版，5.8KB）
├── dag.md                 # DAG执行日志
└── n1/ n2/ n3/ n4/        # 各节点任务目录
```

---

## 🎯 核心发现

### 1. SDS算法已有5个重要改进（2022-2024）
- **Magic3D** (CVPR 2023): 两阶段粗到精，加速2×
- **ProlificDreamer (VSD)** (NeurIPS 2023): 变分框架，解决过饱和/过平滑
- **Invariant SDS** (2024-07): 梯度分解，降低CFG敏感性
- **Reparametrized DDIM** (2024-05): 用DDIM反转替代噪声采样

### 2. 像素画生成有3种主流方法
- **GAN**: 需要大量训练数据，只能生成特定风格
- **VAE**: Pixel VQ-VAE学习离散表示
- **Diffusion+SDS**: SD-πXL，无需训练但慢（小时级/张；论文6000步≈1.5h，实测10000步≈3.5h）

### 3. 扩散模型+调色板约束是活跃研究方向
- Palette Aligned Diffusion (2025-09，早期笔记误标2024): 训练时引入palette conditioning
- 与SD-πXL对比：他们训练模型，SD-πXL用SDS推理时约束

### 4. SD-πXL目前是该领域首个工作
- SIGGRAPH Asia 2024发表（CCF A类）
- 10个引用（发表仅10个月）
- MIT开源，代码活跃

---

## 💡 对当前训练的建议

**你正在跑的配置**:
- 16×16分辨率 + 32色DawnBringer调色板
- guidance_scale=40.0（较激进）

**预期**:
- ✅ 32色足够表达花瓣（远强于4色slowly.hex）
- ⚠️ CFG=40可能过饱和（参考ProlificDreamer/ISD论文）

**后续实验建议**:
1. 成功后尝试降低CFG到20-30
2. 尝试Magic3D两阶段：8色快速 → 32色精细
3. 参考VSD引入可学习噪声预测器（可能加速）

---

## 📚 论文资源

完整论文列表和链接见 `final_report.md`

**关键论文**:
- DreamFusion (2022): SDS原始论文
- ProlificDreamer (2023): VSD改进
- SD-πXL (2024): 本次复现的论文
- Palette Aligned Diffusion (2025): 训练时约束对比

---

**生成时间**: 2026-08-16 04:30  
**数据来源**: Google Scholar, arXiv, CVPR/NeurIPS/SIGGRAPH
