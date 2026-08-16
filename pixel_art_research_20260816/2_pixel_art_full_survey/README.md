# 像素画生成技术全面调研

**调研时间**: 2026-08-16  
**输出文件**: comprehensive_report.md (38KB, 详细版)

---

## 📁 快速导航

### 核心章节
1. **深度学习方法** - GAN/VAE/扩散模型/Transformer
2. **传统算法** - 抖动/量化/矢量转换
3. **图像转换** - Depixelization/超分/风格迁移
4. **交互工具** - Aseprite/Piskel/辅助工具
5. **游戏资产** - 数据集/程序化生成
6. **艺术辅助** - Sketch-to-pixel/自动上色
7. **数据集评估** - 公开数据集/评估指标
8. **最新趋势** - 2023-2026顶会论文
9. **商业应用** - AI工具/API服务
10. **技术对比** - 方法优劣势总结

---

## 🎯 关键发现

### 深度学习（3类主流方法）
- **GAN**: Pokemon GAN, SpriteHand (快但需大数据)
- **扩散模型**: SD-πXL (你在用的), PixelDiT (CVPR 2026最佳论文候选)
- **VAE**: Pixel VQ-VAE (潜在空间编辑)

### 传统算法（仍广泛使用）
- **Floyd-Steinberg**: 1976年经典抖动算法，误差扩散
- **K-means量化**: 调色板生成标准方法
- **Depixelization** (SIGGRAPH 2011): 无损放大到矢量

### 工具生态
- **专业**: Aseprite (27k stars, 最流行)
- **免费**: Piskel (在线), LibreSprite (开源)
- **AI辅助**: Pixelover.io, LlamaGen PixelBox

### 数据集
- GameTileNet (语义标注)
- LPC 4-View (扩散训练)
- Pokemon (huggan/pokemon)

---

## 📊 技术路线对比

| 方法 | 速度 | 质量 | 灵活性 | 数据需求 |
|------|------|------|--------|---------|
| SD-πXL | 慢(3.5h) | 最高 | 极高(任意调色板) | 无(推理时) |
| PixelDiT | 中 | 最高 | 高 | 大 |
| GAN | 快 | 高 | 中 | 大 |
| 传统算法 | 极快 | 中 | 低 | 无 |

---

## 🔥 最新趋势 (2024-2026)

1. **像素空间回归**: PixelDiT移除VAE，直接像素空间扩散
2. **推理时约束**: SD-πXL无需训练，SDS蒸馏SDXL知识
3. **商业化爆发**: LlamaGen, Sprite AI等工具激增
4. **程序化+AI**: GameTileNet语义标注指导生成

---

## 💡 对你的SD-πXL训练建议

基于调研发现：

1. **速度优化方向**:
   - 参考**VSD (ProlificDreamer)**: 可学习噪声预测器
   - 参考**Magic3D**: 两阶段(8色快速→32色精细)
   
2. **调色板选择**:
   - DawnBringer32是经典选择✓
   - 可尝试LPC调色板(游戏资产标准)

3. **评估标准**:
   - 主观: 艺术家/玩家用户研究
   - 客观: Pixel Alignment, Color Coherence

---

## 📚 核心论文列表

### 扩散模型
- SD-πXL (SIGGRAPH Asia 2024) - 你在复现的
- PixelDiT (CVPR 2026 Best Paper Finalist)
- ProlificDreamer/VSD (NeurIPS 2023)

### GAN
- SpriteHand (2024) - 缺失数据填充
- Pokemon GAN (Hugging Face)

### 传统
- Depixelizing Pixel Art (SIGGRAPH 2011)
- Floyd-Steinberg (1976)

完整列表见 comprehensive_report.md

---

**生成时间**: 2026-08-16 07:45  
**参考文献**: 40+ 论文/工具  
**覆盖会议**: SIGGRAPH, CVPR, ICCV, ECCV
