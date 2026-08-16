# 像素画生成技术调研与讨论汇总

**打包时间**: 2026-08-16
**总文件数**: 33+

---

## 📁 目录结构

### 1_sdpixl_research/
SD-πXL论文专项调研（2026-08-16早期）
- `sdpixl_paper_notes.md`: SD-πXL论文核心要点笔记
- `final_report.md`: SDS算法改进综合报告
- `README.md`: 快速索引
- 子目录: n1-n5各节点详细调研

**核心内容**:
- SDS算法及5个改进方向（Magic3D, VSD, ISD等）
- 像素画生成方法对比（GAN/VAE/Diffusion）
- 扩散模型+调色板约束的4篇论文
- SD-πXL论文评价和后续工作

### 2_pixel_art_full_survey/
像素画生成全面调研（2026-08-16下午）
- `comprehensive_report.md`: 18KB完整调研报告
- `README.md`: 快速导航
- `dag.md`: 9节点并行执行日志

**覆盖范围**:
1. 深度学习方法（GAN/VAE/扩散/Transformer）
2. 传统算法（抖动/量化/矢量转换）
3. 图像转换（Depixelization/超分/风格迁移）
4. 交互工具（Aseprite/Piskel等）
5. 游戏资产生成
6. 艺术创作辅助
7. 数据集与评估
8. 最新趋势（2023-2026）
9. 商业应用

**关键发现**:
- **PixelDiT** (CVPR 2026 Best Paper Finalist): 移除VAE的像素空间扩散
- **SD-πXL** (SIGGRAPH Asia 2024): 推理时调色板约束
- **Magic3D** (CVPR 2023): 两阶段3D生成，启发多尺度优化

### experiment_log.md
实验元数据记录（环境 + 各实验配置/结果，持续追加）

### 4_novelty_assessment/
方案定义与创新点评估（2026-08-16晚补充）
- `novelty_assessment.md`: 相似工作全景（SD-πXL/ALPS/CSD等）+ 查重风险 + 补强方向

### 3_discussion_summary/
核心技术讨论总结
- `discussion_summary.md`: 30+轮对话精华整理

**核心议题**:
1. **SD-πXL原理详解**
   - SDS (Score Distillation Sampling)
   - Gumbel-softmax可微调色板约束（论文术语；早期笔记误称"Soft Rasterization"）
   - 为什么不需要训练神经网络，但需要梯度优化

2. **多尺度渐进式架构（用户提出）**
   - 任务本质：把一张**任意尺寸**的大图逐级压缩成**任意目标分辨率**的像素画（16×16、256×256均可）
   - 级数不固定，由 源尺寸/目标尺寸 按实际所需决定（如1024→16为逐级减半的示例，非固定7级）
   - 每级：常规算法缩小 → SDS refinement 修整形状/边缘 → 投影到调色板
   - **首要目标是效果**：每一级的信息损失都被修复过，最终小图才干净可读；加速只是副产品

3. **Magic3D与像素画的关系**
   - 3D vs 2D，但优化策略相同
   - 粗到精 + 多尺度扩散模型指导
   - 用户方案的优势：2D更简单，可用更多阶段

4. **PixelDiT架构解析**
   - 双层级DiT（Patch-level + Pixel-level）
   - Pixel Token Compaction降低计算成本
   - 与SD-πXL的潜在结合方向

5. **技术对比与创新点**
   - SD-πXL: 慢但精确（3.5h, 调色板约束）
   - PixelDiT: 快但无约束（分钟级, 1024²）
   - 用户方案: **以质量为首要目标**（多尺度逐级修整 + 调色板约束，任意源/目标分辨率；速度提升是副产品）

---

## 🎯 核心结论

### 当前最佳方案（SD-πXL）
- ✅ 精确调色板控制
- ✅ 任意分辨率支持
- ❌ 慢：小时级优化（论文报告6000步≈1.5h @RTX 4090；本项目实验10000步≈3.5h）

### 改进方向（多尺度SDS）【2026-08-16 修正定位】
**方案定位**（以用户修正为准，早期"加速14×"的叙事不是出发点）:
- **首要目标：生成质量**——一步到位的降采样+量化会毁掉结构，逐级降采样并在每级用SDS修整，保证每一步的信息损失都被修复
- **任务形态：图像→像素画**——输入任意尺寸大图（不限于1024×1024），输出任意目标分辨率（16×16到256×256均可）
- **级数自适应**——由源尺寸/目标尺寸决定，不是固定7级
- 实现: 基于SD-πXL代码改造；速度提升（步数远少于单尺度10000步）是副产品，不是评价标准
- 评价标准: 对源图忠实度、形状可读性、边缘干净度、调色板协调性

**参考工作**:
- Magic3D: 两阶段验证有效（3D领域）
- Cascaded Diffusion: 多尺度生成pipeline
- VAR: 多尺度autoregressive预测

### 前沿方向（PixelDiT + 调色板）
- 结合PixelDiT的像素空间架构
- 加入SD-πXL的调色板约束
- 可能做出比SD-πXL快10倍的方法

---

## 📚 推荐阅读顺序

### 快速上手（30分钟）
1. `1_sdpixl_research/README.md` - SD-πXL快速了解
2. `2_pixel_art_full_survey/README.md` - 像素画全景概览
3. `3_discussion_summary/discussion_summary.md` - 核心技术讨论

### 深入研究（2-3小时）
1. `1_sdpixl_research/sdpixl_paper_notes.md` - SD-πXL详细笔记
2. `2_pixel_art_full_survey/comprehensive_report.md` - 40+篇论文汇总
3. `1_sdpixl_research/final_report.md` - SDS算法改进分析

### 实现参考
- 讨论总结中的"实现建议"章节
- Magic3D的超参数设置
- 多尺度优化的伪代码

---

## 🔗 重要链接

### 论文
- SD-πXL: https://igl.ethz.ch/projects/sd-pixl/ （代码: https://github.com/AlexandreBinninger/SD-piXL ，arXiv:2410.06236。注意早期笔记里的 sd-pixl.github.io 是错误链接，404）
- PixelDiT: https://github.com/NVlabs/PixelDiT （arXiv:2511.20645）
- Magic3D: https://arxiv.org/abs/2211.10440
- DreamFusion: https://dreamfusion3d.github.io/

### 工具
- Aseprite: https://github.com/aseprite/aseprite
- GameTileNet: arXiv:2507.02941（2025年论文，早期笔记误标为2024）

---

**打包内容**: 所有调研报告 + 技术讨论 + 实现建议
**适用对象**: 像素画生成研究者、多尺度优化实现者
**时效性**: 截至2026-08-16，包含CVPR 2026最新论文
