# 像素画生成技术讨论总结

**时间**: 2026-08-16 12:45:49

---

## ⚠️ 用户修正（2026-08-16，以此为准）

本文档早期版本把多尺度方案框定为"固定7级金字塔、以加速为卖点"，与用户真实意图不符。修正如下：

1. **目标是效果，不是加速**。"700步 vs 10000步、14×加速"不是出发点；速度提升只是副产品，不作为评价标准。
2. **任务是把一张大图压缩成像素画**（image-to-pixel），不是从文本从头生成。
3. **源图分辨率任意**（不一定是1024×1024），**目标分辨率任意**（可能16×16，也可能256×256）。
4. **尺度级数不固定**：根据源尺寸和目标尺寸按实际所需决定，下文的"1024→...→16七级"和"[200,150,...]步数分配"仅是当时讨论的示例，不是方案定义。
5. **评价标准**：对源图忠实度、形状可读性、边缘干净度、调色板协调性。

---

## 一、核心问题与思路演进

### 1.1 初始问题
- **目标**: 生成16×16像素画
- **方法**: SD-πXL (SDS蒸馏 + 调色板约束)
- **痛点**: 极慢（3.5小时/张）

### 1.2 调色板实验
- **4色 (slowly.hex)**: 形状无法表达，退化成几何块
- **32色 (DawnBringer32)**: 训练中，预计能表达曲线

---

## 二、多尺度渐进式架构（用户提出）

### 2.1 核心思路
> 注：下图以1024→16为例。实际源/目标分辨率均任意，级数按需决定（见文档顶部"用户修正"）。
```
1024×1024 (初始)
    ↓ 常规算法缩小
512×512 (粗糙)
    ↓ 深度学习refinement (SDS)
512×512 (精细)
    ↓ 常规算法缩小
256×256 (粗糙)
    ↓ 深度学习refinement
...
16×16 (最终)
```

### 2.2 关键技术问题
**问题**: 如何在refinement阶段接入teaching model产生梯度？

**答案**: Score Distillation Sampling (SDS)
```python
# 伪代码
x = x_coarse.requires_grad_(True)

for step in range(refinement_steps):
    # 1. 加噪声
    x_noisy = x + sigma * noise
    
    # 2. SDXL预测噪声
    noise_pred = sdxl(x_noisy, timestep, prompt)
    
    # 3. SDS梯度
    grad = (noise_pred - noise) / sigma
    
    # 4. 梯度下降
    x = x - lr * grad
    
    # 5. 投影到调色板
    x = project_to_palette(x, palette)
```

### 2.3 与现有工作的关系

#### Magic3D (CVPR 2023, NVIDIA)
- **领域**: 3D生成 (NeRF → DMTet)
- **策略**: 两阶段从粗到精
  - Stage 1: 低分辨率NeRF + 64×64扩散模型 (5000步)
  - Stage 2: 高分辨率Mesh + 512×512扩散模型 (3000步)
- **效果**: 比DreamFusion快2×，分辨率高8×
- **与用户方案的关系**: 
  - ✅ 核心思想相同（粗到精 + SDS + 多尺度）
  - ✅ 优化流程相同（渐进式梯度下降）
  - ❌ 应用不同（3D vs 2D）
  - **用户方案优势**: 2D更简单，无渲染成本，可用更多阶段

#### Cascaded Diffusion Models (JMLR 2022, Google Brain — 非ICLR/OpenAI)
- 多个扩散模型串联，每个模型专注一个分辨率（原论文为class-conditional ImageNet: 32→64→256；"64→256→1024"是后来Imagen的配置，早期版本混淆了两者）
- 区别: 生成流程 vs 优化流程

#### VAR (Visual Autoregressive, NeurIPS 2024 Best Paper)
- 多尺度token预测：在VQ-VAE latent空间预测逐级增大的token map（1×1→2×2→...→16×16网格，对应最终256×256图像），并非像素分辨率16×16→...→256×256
- 区别: autoregressive vs SDS优化

---

## 三、PixelDiT (CVPR 2026 Best Paper Finalist)

### 3.1 核心创新
**完全移除VAE，直接在像素空间扩散！**

传统: 文本 → 扩散模型 → Latent → VAE → 像素
PixelDiT: 文本 → PixelDiT → 像素 (端到端)

### 3.2 双层级架构
- **Patch-level DiT**: 捕获全局语义（粗粒度）
- **Pixel-level DiT**: 精细化纹理（细粒度）
- **Pixel Token Compaction**: 降低计算成本（避免百万级self-attention）

### 3.3 与用户方案的关系
- **不是专门为像素画设计**: 文生图部分做1024×1024，同样核心的是ImageNet类条件生成（256²FID 1.61 / 512²FID 1.81），"主攻1024"的说法以偏概全
- **潜在结合方向**: PixelDiT的像素空间架构 + SD-πXL的调色板约束
- **优势**: 可能比SD-πXL快10倍+（生成模型 vs 优化）

---

## 四、SD-πXL详解

### 4.1 核心机制（非训练，优化）

**误区澄清**:
- ❌ "不需要训练" ≠ 不需要梯度
- ✅ "不需要训练" = 不训练神经网络权重
- ✅ 本质: 用梯度优化像素图（不是优化网络参数）

**SDS (Score Distillation Sampling)**:
```
SDXL = 老师（frozen，参数不变）
像素图 x = 学生（可学习，被优化）

每步:
1. x加噪声 → x_noisy
2. SDXL看x_noisy，预测噪声 → noise_pred
3. 比较noise_pred和真实noise → 差值 = 梯度方向
4. 用梯度更新x（不更新SDXL）
```

### 4.2 可微调色板约束
> 术语勘误：SD-πXL论文的实际术语是 **Gumbel-softmax重参数化 / differentiable image generator**，论文中并未使用"soft rasterization"一词（该词是可微网格渲染领域的术语，系早期笔记误植）。下面的softmax加权原理性解释仍然成立。
```python
# 传统hard rasterization (不可微)
color_idx = argmin(||pixel - palette||)  # ❌ argmin不可微
output = palette[color_idx]

# SD-πXL的可微调色板投影 (Gumbel-softmax思想，简化示意)
weights = softmax(-||pixel - palette||² / temperature)  # ✅ softmax可微
output = Σ weights[i] * palette[i]  # 加权和
```

**为什么可微？**
- softmax平滑，有梯度
- 梯度能反向传播到像素值
- SDS能指导像素往"更接近调色板"方向优化

### 4.3 完整流程
```
1. 生成目标图 (SDXL直接生成高分辨率参考)
2. 初始化16×16随机像素
3. For N步（论文默认约6000步≈1.5h @RTX 4090；本项目实验用10000步≈3.5h）:
   a. 加噪声
   b. SDXL预测噪声
   c. 计算SDS梯度
   d. 更新像素
   e. Soft rasterization投影到调色板
4. Hard rasterization (最终输出)
```

---

## 五、技术对比

| 方法 | 速度 | 调色板 | 分辨率 | 核心思想 |
|------|------|--------|--------|---------|
| **SD-πXL** | 慢(小时级；实测3.5h) | ✅精确 | 任意 | SDS优化+Gumbel-softmax可微调色板 |
| **PixelDiT** | 中(分钟级) | ❌无 | 1024² | 像素空间扩散，双层级DiT |
| **Magic3D** | 快(40min) | ❌无 | 512² | 两阶段粗到精(3D) |
| **用户方案** | — (以质量优先) | ✅精确 | 任意(16²~256²) | 多尺度SDS+调色板，大图→像素画 |

---

## 六、创新点与优势

### 用户提出的多尺度方案
| 特性 | 现有工作 | 用户方案 |
|------|---------|----------|
| 多尺度 | Magic3D(2阶段) | 级数自适应（按源/目标尺寸决定），逐级更平滑 |
| SDS优化 | SD-πXL(单尺度) | 多尺度渐进 |
| 调色板约束 | SD-πXL | ✅保留 |
| 像素艺术 | ❌ | ✅专门优化 |
| 任务形态 | SD-πXL偏文本生成（有可选输入图像模式） | **大图→像素画压缩**（源/目标分辨率均任意） |

**为什么效果更好？（首要动机）**
1. 一步到位的降采样+量化会毁掉结构；逐级降采样，每级用SDS把形状/边缘修整好再往下降，信息损失逐级被修复
2. 高分辨率阶段梯度更准确，先建立好形状，低分辨率只需微调
3. 每级继承上一级结果，不从头开始
4. （副产品）总步数远少于单尺度10000步，但每级步数应由"修到什么程度算好"决定，不预设总预算

---

## 七、实现建议

### 7.1 基于SD-πXL改造
```python
scales = [1024, 512, 256, 128, 64, 32, 16]
steps_per_scale = [200, 150, 100, 100, 50, 50, 50]

x = random_init(1024)

for size, steps in zip(scales, steps_per_scale):
    # Refine at current scale
    x = sds_optimize(x, prompt, steps, resolution=size)
    
    # Downsample to next scale
    if size > 16:
        x = F.interpolate(x, next_size, mode='bilinear')
        x = project_to_palette(x, palette)
```

### 7.2 参考Magic3D的超参数
- **学习率**: 高分辨率大(0.05) → 低分辨率小(0.01)
- **时间步**: 高分辨率高噪声(600-900) → 低分辨率低噪声(200-500)
- **初始化**: 每级从上一级downsample开始

---

## 八、待验证假设

1. **多尺度真的更快吗？** 
   - 理论: 700步 vs 10000步 (14×加速)
   - 需要实验验证

2. **16×16从1024×1024缩小会更好吗？**
   - 理论: 高分辨率梯度更准确
   - 风险: 降采样可能丢失细节

3. **每个尺度最优步数？**
   - Magic3D: Stage1=5000, Stage2=3000
   - 你的方案: 待调优

---

## 九、下一步行动

### 选项A: 实现多尺度SD-πXL
- 基于现有SD-πXL代码改造
- GPU1测试（不影响GPU0训练）
- 对比速度和质量

### 选项B: 等DawnBringer32完成
- 今晚22:30结果
- 验证32色是否足够
- 再决定是否需要多尺度

### 选项C: 研究PixelDiT + 调色板约束
- 长期方向
- 可能做出更强的方法
- 需要几周时间

---

## 十、参考文献

### 核心论文
1. **SD-πXL** (SIGGRAPH Asia 2024): 推理时调色板约束
2. **Magic3D** (CVPR 2023): 两阶段3D生成
3. **PixelDiT** (CVPR 2026): 像素空间扩散Transformer
4. **DreamFusion** (2022): SDS原始论文
5. **ProlificDreamer** (NeurIPS 2023): VSD改进
6. **Cascaded Diffusion Models** (JMLR 2022, Google): 多尺度生成

### 数据集
- GameTileNet: 语义标注游戏tile
- LPC 4-View: 4方向sprite
- DawnBringer32: 32色经典调色板

---

**生成时间**: 2026-08-16 12:45:49
**讨论轮次**: 约30轮
**涉及概念**: SDS, 扩散模型, 调色板约束, 多尺度优化, Gumbel-softmax可微调色板投影
