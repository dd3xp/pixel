V6 自主推进一轮(你是无人值守的定时任务,每30分钟运行一次,每次是全新会话,所有状态只能从文件和服务器获取;不要提问,直接做事)。
项目: 原生精灵空间条件生成;最终目标=text→16×16 RGBA Minecraft item材质。本地仓库 c:\Codes\pixel;服务器 `ssh emnlp`(免密),项目路径 /mnt/data/kw/RoundSquisheen/pixel/pixel,conda 环境 /mnt/data/kw/anaconda3/envs/SD-piXL,HF 需 HF_ENDPOINT=https://hf-mirror.com。
流程:
1. 读 pixel_art_research_20260816/experiment_log.md 末段 40 行获取最新状态与下一步计划(log 是唯一的跨轮记忆,务必先读)。
2. ssh emnlp 检查训练(当前主线 v6d: workdir/v6d_items16, logs/v6d_train.log, 60k 步; GPU0)。用 scp 拉最新样本格到 pixel_art_research_20260816/runs/v6d_items16/,用 Read 工具看图,按 8 条评估 prompt(金柄剑/红苹果/粉花瓣/水壶/蓝药水/金币/木盾/红帽蘑菇)逐格**诚实打分**写入 log;崩溃则修复重启(nohup,日志到 logs/)。
3. v6d 收敛(60k)后的阶梯(按 log 里记录的进度接着做,一轮只推进一步): ①若花瓣/苹果类仍糊→数据二次清洗(踢掉满格 tile 类,只留镂空 item 形素材 = alpha 填充率<0.85 且边缘透明)统计余量,重训为 v6d2 ②扩充自由 prompt 测试(MC 典型 item 16 连测)拉图 ③clean-FID/KID 跑分(vs oga_clean 留出集)。
4. GPU 纪律: 只许用 GPU0/GPU1,其余卡一律不碰。不动服务器上其他人的任何进程/目录。
5. 每轮末尾: 追加 log(带时间戳的一行,写清本轮做了什么、打分、下一步),git add -A && git commit && git push(失败忽略,下轮补)。
6. 若训练正常且无新样本,写一行"无新进展"即可退出,不要做多余动作。
