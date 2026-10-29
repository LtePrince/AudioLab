# E20：Phira 社区 10 首新歌推理评估

2026-10-29，按用户要求从 Phira 社区选择参考难度 13–15 的常规谱，检查当前 AR 模型在新歌曲上的表现。本实验不训练，不调采样参数，不使用谱面起点作为生成输入，不更新发布权重。

## 结果：40 组已完成推理，泛化仍不足

| 候选 | 起点 F1，25 ms | 起点 F1，50 ms | 起点 F1，100 ms | 网格加分 | 自然结束 | 尾部 10 秒召回 |
| --- | --- | --- | --- | --- | --- | --- |
| 4K A / 1400 | 0.3590 | 0.4745 | 0.5508 | 0.1763 | 3/10 | 0.3216 |
| 4K S / 1400 | 0.3009 | 0.3881 | 0.4584 | 0.1746 | 3/10 | 0.2419 |
| free-x A / 1300 | 0.3252 | 0.4567 | 0.5267 | 0.1720 | 3/10 | 0.2862 |
| free-x S / 1400 | 0.3319 | 0.4770 | 0.5613 | 0.1728 | 4/10 | 0.2529 |

全部数值为十首逐曲宏平均，含未结束的原始前缀；参考谱本身的平均网格分为 **0.9121**。时间＋键型 F1 分别为 0.2207/0.1909/0.2246/0.2178；未结束计零后分别为 0.1141/0.1038/0.0890/0.0786。它们与旧的含位置联合 F1 口径不同，不能直接拿来与 0.99 比较。[逐曲完整结果](assets/phira_generalization10_20261029/summary.json)。

当前四套候选在这批新歌上都出现明显的节奏对齐、采音覆盖与结束问题。4K A 本次的起点对应优于 S；free-x S 起点 F1 略高且多一首自然结束，但没有形成明显的网格优势，而且 A/S 轮次不同，不能据此断言额外起音通道的因果收益。正常结束也不保证尾段有效，例如 4K A 的 Constant Moderato 虽然自然结束，最后十秒参考起点召回为零。

为排除测试管线变化，在相同 GPU 前端重新编码训练曲 Eltaw，并对四套候选复跑：两种路线的音频特征及软起音与旧缓存 **逐元素一致**，四份结束状态、起点 F1、原含位置联合 F1 与网格分均复现对应历史结果。该控制曲不计入新歌均值。[一致性对照](assets/phira_generalization10_20261029/rollout_control.json)。CPU 编码的少量浮点差异另存于 `frontend_control.json`，实际新歌评测使用 GPU。

由此可以确认，训练曲的高拟合尚未转化为本批新歌的稳定表现；目前更应针对泛化、时间偏差与自主停止设计下一轮训练，而非仅依据训练集损失继续追加相同训练。这里仍是十首开发评估，没有完成真人试听/触屏试玩。

40 份预测均保留了可预览音符，并生成本地 Phira 包；失败输出沿用明确的截断提示，未在统计里改成成功。预览总入口为 `out/phira_generalization10_20261029/index.html`。完成后磁盘可用约 **8.29 GiB**，没有复制权重或生成重复的合集大 ZIP。

## 名单与来源

从 [Phira 公开谱面接口](https://api.phira.cn/chart?tags=regular&order=-rating&reviewed=true) 的评分排序前六页初筛，再读取下载谱包。下表为推理前冻结的名单；难度来自社区元数据，包含 15.x。所有入选谱均已审核并有 `regular` 标签，实际谱面为恒定 BPM，判定键无低透明度或超大尺寸；保留常规多押。装饰假键从参考事件中排除。未执行手机触屏试玩，标签与数据检查不构成手感认证。

| 社区 ID | 曲目 | 参考难度 | BPM | 时长（秒） |
| --- | --- | --- | --- | --- |
| [40717](https://api.phira.cn/chart/40717) | 天使にふれたよ! | 13.0 | 117 | 185.05 |
| [37396](https://api.phira.cn/chart/37396) | リテラチュア（翻自 上田麗奈） | 13.1 | 118 | 89.40 |
| [6990](https://api.phira.cn/chart/6990) | Constant Moderato | 13.3 | 112 | 140.84 |
| [30114](https://api.phira.cn/chart/30114) | Rubia | 13.0 | 136 | 194.56 |
| [23968](https://api.phira.cn/chart/23968) | Bunny Bunny Carrot Carrot | 14.4 | 170 | 115.70 |
| [51352](https://api.phira.cn/chart/51352) | トリックハート | 14.6 | 150 | 157.00 |
| [64888](https://api.phira.cn/chart/64888) | Yuri on ICE | 14.5 | 80 | 222.02 |
| [51818](https://api.phira.cn/chart/51818) | emp | 15.0 | 180 | 132.00 |
| [52417](https://api.phira.cn/chart/52417) | 匙ノ咒 | 15.5 | 174 | 269.91 |
| [52244](https://api.phira.cn/chart/52244) | あのね | 15.0 | 246 | 199.02 |

归一化曲名和原音频 SHA 均未命中现有 300 首解码器训练数据。尚未做跨转码/变速的声学指纹搜索，冻结前端早期训练数据的重合情况也未建立。完整来源、谱师、音频/谱面哈希、筛选数据见 [冻结清单](assets/phira_generalization10_20261029/selection.json)。未因模型表现更换曲目。多数谱的最大同时判定数为 2–4；《匙ノ咒》有一组 11 个同时 Drag，其余最大为三押，保留这类扫键，不用双指上限筛除常规谱。

只提取必要的原音频、谱面和元数据，删除本次下载的临时 ZIP 及其中未使用的图片、视频、故事板。原音频下混并重采样为 22050 Hz 单声道浮点 WAV，保留时间长度与解码后波形，包括少量超出 ±1 的解码峰值；推理与预览共用同一 WAV。清理记录位于 `out/phira_generalization10_20261029/download_cleanup.json`。

## 冻结模型与评价口径

| 运行 | 权重轮次 | 条件 |
| --- | --- | --- |
| AL-AR4K v0.5.1 / E14-A | 1400 | 纯音频，无额外起音通道 |
| AL-AR4K v0.5.1 / E14-S | 1400 | 同音频前端，增加软起音 |
| AL-FreeX v0.4.0 / E18-A | 1300 | 纯音频，采用上一阶段训练曲最佳轮次 |
| AL-FreeX v0.4.0 / E18-S | 1400 | 增加软起音 |

四套候选均在测试前根据训练曲结果选定。保留原前端、全局音频条件与完整自主生成；贪心、seed 42、12000 token 上限不变。BPM 取参考谱；参考音符和 offset 仅进入评分，未送入生成器。原始预测不做网格吸附或排布修复。模型输出难度未定级，不能因参考谱为 13–15 级就认定生成谱具有相同难度。

- 复用原唯一事件匹配器，分别报告 25/50/100 ms 起点 F1；它衡量与该作者采音的对应，不是唯一正确编排的判定。
- 复用原 `0/4/8/16/24/32` tick 网格加分；使用参考 BPM 与原 offset，报告参考谱本身的网格分，不拟合新的相位、不吸附输出。
- 辅助报告“时间＋键型 F1”、音符数比、尾部 10 秒召回、四种键型与排布统计。这里没有位置匹配，**不能与训练曲的“时间＋位置 bin＋键型联合 F1”直接比较**。社区谱包含动态判定线，本实验不把它的空间排布强行当成静态目标。
- 自然结束单独报告；原始指标包含失败前缀，另列未结束计零的时间＋键型分。正常结束不等于完整覆盖。导出沿用现有截断预览逻辑，保留原始失败状态与警告。

RPE 类型映射（1 Tap、2 Hold、3 Flick、4 Drag）与时间约定依据 [Phira RPE 解析器](https://github.com/TeamFlos/phira/blob/750579396cbc5dc61e6379fbfb2aa1adb70751a6/prpr/src/parse/rpe.rs) 和 [游戏时钟](https://github.com/TeamFlos/phira/blob/750579396cbc5dc61e6379fbfb2aa1adb70751a6/prpr/src/scene/game.rs)：`audio_seconds = beat × 60/BPM + META.offset/1000 + info.offset`，设备校准 offset 取零。读取内容已与该版本的 Git blob 校验匹配，来源记录见 `phira_source.json`。三项自动测试已覆盖四键型映射、两个 offset 相加、假键排除、变 BPM/隐藏判定键拒绝、完美匹配、明显时间平移与失败前缀计分。

## 运行与产物

复现脚本：[benchmark.py](assets/phira_generalization10_20261029/benchmark.py)。两个 GPU 分别跑 4K 与 free-x，每个进程顺序推理两套权重，共 40 份整谱。资源保护保持磁盘 8 GiB、RAM 8 GiB、RSS 4 GiB、GPU 分配 65%。

```bash
.venv/bin/python docs/evaluations/assets/phira_generalization10_20261029/benchmark.py evaluate --mode 4k --device cuda:0
.venv/bin/python docs/evaluations/assets/phira_generalization10_20261029/benchmark.py evaluate --mode freex --device cuda:1
.venv/bin/python docs/evaluations/assets/phira_generalization10_20261029/benchmark.py summarize
```

本地产物：`out/phira_generalization10_20261029/<社区ID>/<运行ID>/`，包括原始预测、评分、可预览部分、谱包与来源报告。进度为根目录 `progress-4k.json` 和 `progress-freex.json`。谱包仅留本地，没有上传 GitHub Release 或社区。
