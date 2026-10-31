# Alvin 与缓存清理、训练监测记录

2026-10-31。按用户要求检查 Alvin 和用户缓存，完成两次定点清理，合计观测释放 **20.21 GiB**，完成时磁盘可用约 **29.59 GiB**。期间 free-x 队列正常完成全部四组 400 轮训练。

| 清理内容 | 数量 | 释放空间 |
| --- | ---: | ---: |
| uv 失败下载／构建留下的临时目录 | 2 个 | 1.58 GiB |
| WAM 原版 Wan2.2-TI2V-5B 下载分片 | 3 个 | 18.63 GiB |

uv 清理目标是 `/home/ziyaowang/.cache/uv/.tmpcf4Gdw` 和 `builds-v0/.tmpcoEp1b`，文件最近修改在 8 月。整个 uv 缓存表观占用约 9.06 GiB，大部分与已安装环境共用硬链接；本次保留注册缓存，仅清理两个过期临时目录。Alvin 与用户 `.local` 内未发现指向这些 uv 缓存的外部符号链接。

WAM 删除的是 `WAM/FastWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors` 至 `00003-of-00003` 三个原版下载分片，非用户训练产物。保留原下载元数据 `.msc`、`.mv`；ModelScope 来源为 `Wan-AI/Wan2.2-TI2V-5B`，本地记录版本 `c2272aadfec2d95cea869a4e748d5e28275621c6`。每个文件的大小、SHA256 与原路径已保存于 [恢复清单](assets/storage_maintenance_20261031/plan.json)。**后续 WAM 若使用默认的原版 DiT 初始化，需要重新下载这些分片**；没有承诺存在本地二进制备份。

删除前在宿主机核对当前用户的 WAM／包安装进程及候选文件占用，未发现冲突。缓存目录位于工作区外，清理按平台审批后执行。保留了所有项目代码、虚拟环境、训练数据、自训 LoRA／AgentVLA 权重、FastWAM 已发布的 `libero_uncond_2cam224.pt` 及 VAE。Hugging Face 中的 lerobot 本地数据不当作可随意重建的包缓存删除。

AudioLab 的四份 1400 轮完整续训断点和两套当前受控训练数据，在两次清理前后逐文件 SHA256 一致。当前运行中的新权重没有作为清理候选。三份用户未提交文档也未修改。没有删除其他用户的共享 `/tmp` 内容；扫描遇到的无权访问目录未进一步操作。

## 训练监测结果

监督器正常退出，A-W0、A-W1、S-W0、S-W1 均为 `completed`，每组 400 轮／12,800 次更新；这是 32 条构造样本的受控试验。GPU 0/1 结束后使用约 131/16 MiB，利用率均为 0%。没有把中间 8 条样本的满分当作最终泛化结论。

最终未入训随机组合起点 F1@25ms：A-W0/W1 为 0.5142/0.5809，S-W0/W1 为 0.6036/0.6272；联合排布仍弱。本轮模型不替换产品权重。完整对照见 [第 4 阶段报告](windowed_ar_stage4_20261031.md)。第 5 阶段尚未执行。

清理逐项结果：[缓存](assets/storage_maintenance_20261031/cache_result.json)、[原版下载分片](assets/storage_maintenance_20261031/result.json)；进程核对：[缓存核对](assets/storage_maintenance_20261031/cache_process_check.json)、[权重核对](assets/storage_maintenance_20261031/process_check.json)。空间按两次清理前后差值分别统计，期间训练会新增 checkpoint，因此不将目录表观大小或一次总差值冒充释放量。
