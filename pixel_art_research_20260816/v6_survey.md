# V6 调研备忘录：原生精灵空间条件生成（2026-08-20，三路并行调研汇总）

> 背景：v1-v5"压缩范式"达到天花板（32级7-8分,16级6-7分）,根因=微特征必须"画"而非"压"。本调研为 v6（条件生成范式）提供决策依据。已排除 SDS/SD-πXL 方向。

## 一、架构证据（最强→最弱）

1. **像素空间小 UNet 扩散,原生 16-32px 训练** —— 2022-2025 被独立验证多次：
   - PixDiff-PIG（2025）: TinyUNet DDPM + OKLab + 调色板/索引图/亮度条件,FID 19.6,击败 StyleGAN2(41.7) 和 LDM(28.4)。**必对标 baseline**,未开源
   - 教学级证明：16×16 + ~89k 数据 + 几 M 参数即可出干净样本（DeepLearning.AI 课程数据可直接用）
2. **离散路线单点证据全部为正,组合为空白**：
   - 逐像素粒度表征有效（Pixel VQ-VAE 的 1×1 PixelSight,168k 宝可梦 64px,开源）
   - **每像素 palette one-hot 输出头,100 张数据就能训**（Five-Dollar Model, AIIDE 2023）
   - 离散 AR 可行（PixelCNN-on-codes / PixelBytes）
   - **"离散扩散 × 调色板索引 × 精灵"无任何已发表工作 —— 明确的可发表空白**
3. GAN：仅在成对姿态翻译上有性价比,生成任务已被淘汰（mode collapse+颜色噪声）
4. **商业反向印证**（Retro Diffusion 创始人访谈）：最强产品=FLUX 微调+专有降采样+量化后处理,**不是原生网格**;其公开承认的痛点（色数失控、网格错位）恰是原生索引输出天然解决的——差异化成立

## 二、忠实度条件方案（排序,均有文献支撑）

1. **解耦 cross-attention 注入图像 embedding**（IP-Adapter 模式,机制与模型规模无关,~22M 参数级;建议 CLIP 与 DINOv2 都试）——身份通道
2. **结构化 caption 双通道**（VLM 抽"主体/姿态/配色词"）——属性通道;文献实证：颜色/风格属性经文本可靠传递,**身份不能**（arXiv:2605.24624）,故双通道分工明确
3. **显式调色板条件**（照片 k-means 8-16 色,OKLab 空间注入;ColorPeel 证明精确色可 embedding 化）
4. 分层注入调优（InstantStyle 经验：低分辨率层给形态、高分辨率层给颜色）
5. 不推荐：人脸 ID 路线（不适用任意主体）、纯文本条件（身份传不过去）
- **重要负结果**（Coutinho）：调色板索引作【输出表示】在 GAN 里过拟合——但 Five-Dollar 的 one-hot 头 + 扩散/AR 无此报告;输出端量化 vs 模型内量化的消融文献支持**模型内**（可微调色板量化 2025、PixDiff-PIG 的 OKLab 消融 FID 19.6→23.7）
- 工业先例：**Apple Genmoji** = "照片身份→极小分辨率风格化"唯一产品化案例,做法是风格域由训练数据钉死、身份弱注入

## 三、数据（不是瓶颈）

| 桶 | 内容 | 规模（清洗后估计） |
|---|---|---|
| 绿(CC0) | OGA-CC0 仓切割 + Nouns 32px | 5-15 万 |
| 绿(署名) | OGA CC-BY/OGA-BY + GameTileNet | +2-3 倍 |
| 黄(CC-BY-SA) | LPC 生成器**受控合成**（带完整属性标签,天然适合条件训练） | 10-50 万 |
| 灰 | 宝可梦系（学界先例多,不再分发、不商用） | +20 万,只作域泛化测试 |
| 红线 | itch.io 批量抓取（明文禁AI）、老游戏截取、Lospec gallery | 不碰 |
- 合计可发表训练集 **30 万-100 万张**,32px 级别属大规模
- 清洗工具链现成：pixeldetector/pixel-scale（原生网格检测）、alpha 连通域切割、pHash+SigLIP 去重
- **配对数据（忠实度通道）三路**：(a) 我们的 v5 链批量像素化照片=合成对（**v5 转世为数据引擎**）(b) depixelize 真精灵反向构造 (c) 前沿闭源模型造对（OmniConsistency 范式）

## 四、评估

- FID 低分辨率域外+对 resize 极敏感：用 clean-fid + 最近邻整数放大,同时报 KID/CMMD
- 专用指标：PCS（调色板一致性,PixDiff-PIG）、网格对齐分（pixeldetector 通过率）、唯一色数分布、条件一致性（CLIP score/属性分类器）
- 用户研究：2AFC 成对强制选择 + "可直接进游戏引擎"画师评审

## 五、V6 架构定稿建议

**双模型并行**：
- **Model A（锚点,低风险）**：TinyUNet DDPM @32×32,OKLab,条件=调色板+caption+图像embedding（解耦cross-attn）→ 复现并超越 PixDiff-PIG（FID~20）
- **Model B（新颖性,论文主角）**：**masked/uniform 离散扩散 over 调色板索引网格**,每像素 one-hot 头（Five-Dollar 证据）+ 1×1 逐像素粒度（PixelSight 证据）+ 同套三条件——填补已确认的文献空白
- 忠实度：双通道条件,配对数据先用 v5 合成对大规模预训,再用闭源模型精修对对齐
- 算力：TinyUNet 级,8×A100 单卡数小时/run,完全无压力
- 论文定位："原生离散精灵生成+参考条件化" vs Retro Diffusion 自认痛点;110 个 v1-v5 实验成为 motivation（压缩范式系统性失败的证据地图）

**风险**：离散扩散无前人超参参考（Model A 兜底）;"像具体这只"的忠实度上限未知（Genmoji 先例说明弱注入可行,强忠实待验证）;PixDiff-PIG 未开源需自行复现 baseline。
