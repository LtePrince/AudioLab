# S2 音频理解首版实现

本阶段把音频证据从玩法编排中分离，提供音频 → 连续特征／事件概率／帧内时间精修 → 候选事件的可运行链路。复用 `TranscriberNet.features()`；新声学训练使用全局事件头，不预测轨道、键型、Hold 数或难度。

## 交付与验收范围

| 检查 | 本阶段要求 |
| --- | --- |
| 时间 | 统一音频绝对秒；短音频、奇数帧、窗口交界与末尾无静默截断；转换谱面 tick 时只减一次 offset |
| 输出 | 版本化 JSON、连续特征与未筛选概率 NPZ；保留音频／权重／特征／推理参数与来源指纹 |
| 监督 | 已审核区域才参与 loss；不确定区间屏蔽；声学、伪标签、谱面取音三类不混报 |
| 训练 | 默认冻结编码器、重置事件头；可显式微调；来源分组与字节哈希防止跨集泄漏 |
| 评价 | 唯一起点 P/R/F1@25/50ms、误报／分钟；独立 100ms 匹配时间误差；按歌曲宏平均与重采样区间 |
| 接入 | 同一结果可作为 4K、free-x 的候选骨架；音频哈希和时长不匹配时拒绝 |
| 资源 | 整曲 mel、分段 attention；解码前检查 RAM 估算；单曲缓存；受限 checkpoint 保存 |

**这是功能交付，不是新的声学质量验收。** 现有 `timing_v1/transcriber` 和 `rhythm_global_v2` 权重来自谱面监督，输出明确标为 `chart_onset`。本轮未找到独立审核的声学标注，未开展真实声学训练或把游戏谱面伪装成声学真值。合成数据上的训练测试只检查梯度、冻结、序列与文件链路。

beat、downbeat、持续、音高与音乐段落均返回 `unavailable`；`tempo_map` 为 `null`。它们需要相应独立标签和后续任务头。没有推断 BPM，也不会把声学延音当作游戏 Hold。README 中的声学候选门槛仍未冻结。

## 推理与结果

在包含依赖的项目环境中运行；独立 worktree 可以直接使用原工程 `.venv/bin/python`，并通过绝对路径读取原数据和权重，无需复制。

```bash
python -m script.understand_audio \
  --audio /path/song.ogg \
  --checkpoint /path/transcriber_best.pt \
  --output out/understanding/song.json --device cpu
```

也支持新入口训练的 `audio_best.pt` 和已有的全局节奏 checkpoint。默认参数为 1024 个 mel 帧的 core、左右各 128 帧上下文、阈值 0.7、35ms 置信度 NMS。**这些阈值是手动基线，`calibration=manual_unvalidated`，不是新声学头的校准结果。** 阈值／间隔应只在训练或开发集选择，之后固定在独立测试与下游对照中。不得逐曲优化或平移时间。

JSON 包含候选事件 ID、音频秒、置信度、`label_source`、支持能力、配置、来源指纹、NPZ 校验和。零候选明确标记 `status=no_candidates`，不等于成功成谱。NPZ 保留：

- `features`：编码器时间序列，形状 `[T_encoder,D]`；`feature_times_seconds` 为帧锚点。
- `event_probability`、`frame_times_seconds`、`time_offset_seconds`、`event_times_seconds`：筛选前的概率与精修时间。
- `valid_event_time`：精修点是否落在真实音频长度内；末尾 padding 不能变成音频外事件。

特征采用原 `AudioGPUprocessor` 的 22050Hz、2048 FFT、512 hop、128-bin 归一化 log-mel 实现，在 CPU 提取整曲特征再分窗，避免每窗独立 dB 归一化。不能混用旧 `AudioCPUprocessor` 的 librosa 特征缓存。

默认音频长度上限 600 秒、最多 8 声道、192kHz 输入采样率；超过上限明确失败。短于 STFT reflect padding 要求的输入先补零，但输出仍以真实时长截断。支持格式取决于当前 soundfile 与 torchaudio 解码器；本轮实测 WAV、OGG，不宣称所有平台 MP3 已验证。全曲波形／mel 仍在 RAM 中，本实现不是无限长度流式解码。

输出必须使用新的 `.json`/`.npz` 路径；JSON 最后落盘，作为完成标记。推理中断可能留下没有 JSON 的 NPZ；不得把这个文件当作完成任务。数据与产物在 `.gitignore` 下，不进入 Git。

## 声学标注与训练

JSONL 每行一个音频标注记录；下面为了阅读展开，实际每行须是完整 JSON：

```json
{
  "schema": "audiolab-audio-events-v1",
  "id": "song-a",
  "source_group": "original-composition-a",
  "source_start_seconds": 0,
  "audio": "audio/song-a.wav",
  "duration_seconds": 120.0,
  "label_source": "reviewed_acoustic",
  "provenance": "review protocol v1; annotator/reviewer IDs and revision",
  "coverage": [[0.0, 30.0], [60.0, 90.0]],
  "uncertain": [[10.0, 10.5]],
  "events_seconds": [0.125, 2.51, 62.0]
}
```

时间与覆盖区间都相对该音频的 0 秒，区间左闭右开。`duration_seconds` 必须与实际解码时长相符；示例数值不能直接作为真实文件标注。`source_start_seconds` 将剪辑映射到原曲时间轴，仅用于歌曲级评估合并。来源分组需要人工维护，同曲不同难度、重编码、剪辑必须同组；字节哈希只能发现完全相同文件，不能替代近重复音频审计。未审核区不能填入负例，不确定区间中的事件不训练、不计分。

同刻事件按 1 微秒精度去重。一个 23.22ms bin 内如果有多个不同事件，当前单事件头无法全部表达：训练保留存在性标签，屏蔽该 bin 的偏移 loss，记录碰撞数；不把多个事件平均成一个虚构起点。这是表示分辨率限制。

```bash
python -m src.train_audio_understanding \
  --train-manifest data/acoustic/train.jsonl \
  --val-manifest data/acoustic/dev.jsonl \
  --encoder-ckpt /path/transcriber_best.pt \
  --run-dir checkpoints/acoustic_v1 --device cpu
```

默认只训练新事件头。`--encoder-lr 0.00001` 显式启用编码器微调。不复用旧谱面事件头；`chart_onset` 标签不能进入此训练入口；伪标签需 `--allow-pseudo-labels`，并以单独实验与来源标识训练、评价。默认 8 epochs、batch 2、1024 帧训练窗；不改现有模型或生成默认值。

遍历所有包含已审核帧的窗口，包括末尾，按歌曲随机排列、再随机排列该曲窗口，单曲仅解码一次／epoch，最多缓存一首训练曲和一首验证曲。窗口内未审核及 padding 帧屏蔽 loss。训练窗口上下文与推理 halo 不完全相同，质量必须实测。保存 `audio_best.pt` 与 `audio_last.pt` 两份固定命名权重（不无限累积），以及清单／音频／编码器哈希、代码版本、配置、碰撞数和逐轮 loss；目前新入口不提供断点续训，应使用新的 run 目录。

资源参数沿用 `ResourceBudget`，默认磁盘和可用 RAM 各保留 8GiB，RSS 上限 10GiB；GPU 内存比例 0.65。训练、推理和评估也已登记到 `script.train_models --worker`，可通过原资源调度入口运行。

## 独立评价

先对每首歌按同一权重、阈值和窗口参数推理，再建立 ID 到结果路径的 JSON 索引，例如 `{"song-a": "results/song-a.json"}`，路径相对索引文件。

```bash
python -m script.evaluate_audio_understanding \
  --manifest data/acoustic/test.jsonl \
  --predictions out/predictions.json --output out/acoustic-evaluation.json
```

要求所有标注 ID 均有结果，不静默跳过失败曲；检查音频字节哈希、时长和冻结配置。声学、伪标签和谱面标注必须分别报分。协议为 `unique-us-max-cardinality-min-absolute-error-v1`：先最大化一对一匹配数，再最小化绝对误差总和；与旧脚本的贪心最近匹配口径不同，不能拼接历史分数。

评估包括开头，不忽略前 5 秒。100ms 匹配单独报告有符号平均误差、绝对误差中位数/P95，以及参考与预测的未匹配比例。无匹配时误差为 `null`，不报告为 0。无参考且无预测时 P/R/F1 定义为 1；有参考无预测时为 0；无参考有预测时 recall=1、precision/F1=0。未审核区域的预测不计 FP，但单独统计忽略数。

同一原曲的不同片段先在原曲时间轴合并不重叠的已审核区域，再匹配和计算歌曲指标，避免切片方式改变 F1。相邻片段允许跨接缝匹配，重叠区域拒绝重复计分。之后对歌曲宏平均并以歌曲为单位 bootstrap 2000 次，seed=42；单曲不伪造区间。保留逐记录、逐歌曲结果和所有指纹。当前不提供曲风自动分类或自动通过产品门槛。

## 下游接入

```bash
python -m src.test_decoder \
  --audio /path/song.ogg --audio-events out/understanding/song.json \
  --ckpt /path/decoder_best.pt --bpm 120 --offset 0 \
  --output out/song-4k.json --device cpu
```

换成 free-x 的 decoder checkpoint 即使用同一音频结果生成无轨布局。`--audio-events` 不与其他骨架输入或重新筛选参数混用；当前 CLI 接入为单曲模式。下游编码器／布局权重保持原有流程，尚未把新连续特征直接替换到已训练 decoder；此处验证的是共用候选事件与时间接口。难度取音、联合训练、真实播放器与玩法验收属于后续 K/F 阶段。

## 首轮功能证据（2026-10-17）

合成测试覆盖训练→推理→评价、冻结编码器不变、两个 decoder 的事件接入、短/奇数/尾帧、完整窗口所有权、部分标注、碰撞 loss、歌曲切片不变性、重叠拒绝、泄漏拒绝与 RAM 前置保护。匹配器与小规模穷举最优解对照；末尾精度用 0.1/0.3/1.1/2.3 秒回归。合成输入不能证明真实声学质量。

完整 `python -m pytest -q test`：**209 passed**，其中新增音频理解测试 29 项。历史依赖的音频／谱面 fixture 通过只读用途的软链接复用，生成 fixture 与缓存保留在本 worktree／临时目录；不复制或覆盖原数据。9 条 warning 来自原测试的依赖弃用提示、四轨映射碰撞及禁用 rate augmentation 提示。

真实 `Eltaw.ogg` 时长 135.550839 秒，5838 mel 帧。既有转录权重、手动阈值 0.7/NMS35、CPU 两线程下，默认分段流程运行约 2.79 秒，RSS 快照约 0.80GiB，输出 530 候选；连续特征尾帧锚点 135.511655 秒，覆盖真实末尾。这是单次本机功能观测，不是 P50/P95 或产品速度承诺。

同曲整段 attention 得到 557 候选，分段得到 530，二者 25ms 匹配 501 个；概率平均绝对差约 0.04475。窗口边界左右各 200ms 内，整段 14、分段 12、匹配 12。**分段与整段不等价，旧全曲 F1 和校准参数不能直接迁移。** 这里只测一致性，不能判断哪一个更接近真实声学标签。报告保存在本地 `out/audio_understanding_smoke/chunk_comparison.json`。

两首真实音频的数据读取探针遍历 13 个窗口，每首解码一次，总耗时约 2.07 秒、进程峰值 RSS 约 1.11GiB；保存于 `out/audio_understanding_smoke/io_probe.json`。该探针不训练，不建立声学标签，也不代表 600 秒／8 声道边界或 GPU 峰值已实测。

同一份 `eltaw_v1.json` 接入 `timing_v1` 真实权重，4K 输出 630 音符、free-x 输出 624 音符；两者均完整保留 530 个唯一候选起点，四种键型均有输出，时间范围／Hold 结束／位置／时长硬错误均为 0。最大同刻音符均为 3，作为多指诊断保留。原静态后处理在无轨上截短 9 个 Hold，定轨未截短；不将该修复当成模型学习收益。分别约 18.5/20.4 秒，为两任务同时运行时的单次观测，不能作为标准速度比较。

记录与谱面保存在 `out/audio_understanding_smoke/downstream_audit.json`、`eltaw_4k.json`、`eltaw_freex.json`，含音频／权重／谱面指纹；未做目标播放器导入或人工试玩。单曲参考谱取音评估也已走新入口，`label_source=chart_onset`，F1@25/50ms 均约 0.68447，仅作为 `chart_onset_evaluation.json` 的流程证据，不是独立声学测试成绩。

尚需独立审核标注和来源去重审计、真实声学训练与阈值校准、冻结测试，以及同一配置下两种玩法的完整质量／资源对照。当前分支完成工程功能验证后推送，不据此升级 `main`。
