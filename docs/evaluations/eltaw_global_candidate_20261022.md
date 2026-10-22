# Eltaw：global 4K 单曲试玩候选（2026-10-22）

按用户要求，从既有网格匹配好的模型中选择 `4k/global/seed42`，重新执行一次完整音频推理并打包；暂不扩建独立歌曲测试集，先改善训练曲表现。本次未训练新模型，未修改生成代码、产品默认权重或用户原有两个谱包。

- 文件：`../AudioLab_test_phira/Eltaw-4K-global-seed42.phira.zip`，1825445 字节（约 1.74 MiB）。
- SHA-256：`4cfa33d2e4dd964eef2752bfc569e1e25cb4f23b6859a4d2f84a23ef0c266109`。
- 模型：global cross-attention，scratch 10 epochs，soft onset＋历史状态；源训练 checkpoint SHA `cef1095fa0ca1441968e41d02ed17e9fd0bce5e086d33299dc0305237d800b57`。临时运行 checkpoint 只重新封装同一 decoder 与其冻结 encoder，参数未改；推理结束后已清理该临时副本。
- 音频 SHA `253c107a59aae0c7e027511ddb66025d6c831649d2ce584edde8ef56ea166708`，与用户旧包音频完全一致。
- BPM=193（沿用参考谱／benchmark 时基）、offset=0、seed=42、temperature=1、top_p=.95；用户旧包自动 BPM=129.19921875，因此新旧包同时包含权重与时基差异，不能归因于单一因素。
- 实际完整推理耗时 30.52 秒。进程峰 RSS 1.632 GiB；torch CUDA 峰 allocated 0.113 GiB、reserved 0.156 GiB。它们不是显卡整体占用／其他硬件性能保证。

原始生成的每个音符 tick／轨道／类型／时长与既有 benchmark 对应输出逐项相等。最终包有 1321 个音符：Tap 1189、Drag 12、Hold 111、Flick 9。保持现有 static-v1 打包路径，19 个 Hold 被原有同轨后续起点／曲尾规则截短，0 个转 Tap；起点、轨道、数量保持。原始谱、最终谱和逐音修改记录分别保存在 `out/eltaw_global_candidate_20261022/job/4k.raw.json`、`4k.json`、`4k.repairs.json`。

本曲原始唯一起点 F1@50ms=0.5403，网格平均奖励=1.000；是训练曲开发诊断。首个音符 1.244 秒，最后一个 127.306 秒，音频长 135.551 秒。Tap 占比与尾段遗漏仍需用户听看；网格满分未解决音乐对应或乐句模式问题。

谱包通过全部成员 CRC／SHA、完整音频解码、静态 PGR 格式、时间与位置校验。未执行本机 Phira／触屏试玩。包内包括音乐、谱面、封面和元信息，可直接传到手机／平板导入。难度未定级。

新问题 ELTAW-04（前奏重复／模进未体现于键型模式）已记录到 [用户反馈](eltaw_user_feedback_20261022.md)。接下来关注乐句间节奏／键型动机的一致性，以及模进的相对位置变化；不把多生成某种键型或机械复制图案当作改善证据。

复现来源与全部检查：本地 `out/eltaw_global_candidate_20261022/preparation.json`、`delivery.json`、`benchmark.json`、`job/manifest.json`；已有源训练权重仍保留。音频、权重及谱包不提交 Git。
