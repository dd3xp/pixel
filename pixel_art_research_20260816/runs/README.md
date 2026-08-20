# runs/ — 每实验一个目录的完整过程链

每个目录内：
- `00_source_*.png` 源图
- `config.yaml` 该 run 的完整配置
- `level_<i>_<size>.png` 金字塔各层级结果（按分辨率从大到小）
- `zz_final.png` 最终像素画（256px 最近邻放大便于查看）

单尺度实验（petal_16_db32_full 等）无 level 文件。逐步中间图（每50步）仍在服务器 workdir 对应 run 的 png_logs/ 下。
以后每次从服务器收割结果都按此结构入库。

特殊目录：`_early_samples/`(P0-P2d 时期的散图) `_strips/`(各版本并排对比条,看效果先看这里) `_stylized_sources/`(风格桥接产物)。
