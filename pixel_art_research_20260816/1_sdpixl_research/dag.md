# DR: SD-πXL相关论文调研

ROOT: 查找SD-πXL相关的论文，包括SDS算法改进、像素画生成方法、扩散模型约束生成

## 节点列表

### 第一批（并行）
- [N1] WEB | 子问题：DreamFusion之后SDS算法有哪些改进（VSD/ProlificDreamer/Magic3D等） | 依赖：无
- [N2] WEB | 子问题：像素画/低分辨率图像生成的深度学习方法有哪些 | 依赖：无
- [N3] WEB | 子问题：扩散模型+离散约束（调色板/量化）的相关工作 | 依赖：无
- [N4] WEB | 子问题：SD-πXL论文本身的引用情况和后续工作 | 依赖：无

### 第二批（综合）
- [N5] SYNTH | 汇总四个方向的论文，提取核心改进点和可借鉴思路 | 依赖：N1,N2,N3,N4

## 节点状态
N1: [✓] N2: [✓] N3: [✓] N4: [✓] N5: [✓]

## 执行日志
- 2026-08-16 04:15: N1-N4并行启动（4个subagent进程）
- 2026-08-16 04:28: 所有节点完成
- 2026-08-16 04:30: 综合报告生成完毕

## 节点结论摘要

**N1 - SDS算法改进**:
找到5个核心改进方向：Magic3D(两阶段加速)、ProlificDreamer/VSD(变分框架)、ISD(梯度分解)、Reparametrized DDIM(反转近似)

**N2 - 像素画生成方法**:
三大类：GAN-based(Pix2Pix变体、sprite生成)、VAE-based(Pixel VQ-VAE)、Diffusion-based(即SD-πXL)

**N3 - 扩散模型约束生成**:
找到4篇核心论文：Palette Aligned Diffusion、Color Transfer、Palette Guidance、Constrained Discrete Diffusion

**N4 - SD-πXL后续**:
论文信息：SIGGRAPH Asia 2024、10引用、MIT开源、ETH Zurich团队。暂无直接后续工作（领域较新）

**N5 - 综合分析**:
已生成final_report.md，包含全部4个方向的详细分析+对当前DawnBringer32训练的建议
