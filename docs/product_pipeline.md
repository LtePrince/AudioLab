# 本地快速成谱工程入口

本入口把现有权重串成可检查的完整歌曲流程：共享音频候选 → 定轨 4K／静态 free-x → 原始与修复输出 → 音频同步预览 → JSON 导出。它交付工程闭环，不代表声学质量、难度控制、真实玩家手感或目标播放器兼容性已验收。

## 运行

使用项目现有 Python 环境，无需安装新的前端依赖。在仓库根目录执行：

```bash
python -m script.generate_product \
  --audio /path/song.ogg --bpm 160 --mode both \
  --model-root /path/AudioLab/checkpoints/timing_v1 \
  --output out/song-job --device cpu
```

已有 GPU 调度入口也可使用：

```bash
python -m script.train_models --gpu 1 --worker script.generate_product -- \
  --audio /path/song.ogg --bpm 160 --mode both \
  --model-root /path/AudioLab/checkpoints/timing_v1 \
  --output out/song-job --device cuda:0
```

`--bpm` 可选：提供时使用显式时基；留空时使用后述未校准速度估计。为约束 token 时间跨度，工程入口接受 1–1000 BPM；这不是可玩难度范围。`--mode` 可选 `4k`、`freex` 或 `both`；后者顺序生成两种玩法。当前是基线谱，没有未经训练验证的难度档位。仅支持本轮已验证的 WAV／OGG 输入，工程上限 600 秒、128 MiB 文件；超过范围明确失败，不截断音频。

输出目录必须不存在，避免混合不同权重和配置的结果。参数与权重指纹记录在结果中。单任务默认 CPU 两线程、RSS 限额 4 GiB、可用 RAM／磁盘各保留 8 GiB；GPU 内存比例默认 0.65。900 秒超时是工程退出条件，不是产品速度承诺。公开服务队列和多用户并发不在本入口中。

## 结果与预览

仅当 `manifest.json` 存在且 `status=completed` 时，整项任务完成。`job.json` 记录当前阶段、解码进度和失败原因；任何一路失败，整个双模式任务不生成完成标记。

打开 `preview.html`，浏览器直接读取同目录音频，离线播放／暂停／拖动音频时间／回到开头。两种模式可切换，四种键型用不同颜色与形状表示。预览唯一时钟是 `audio.currentTime`；音符使用已经包含一次 offset 的绝对音频秒，暂停和跳转时不累加虚拟时钟。Hold 显示持续到尾端。此页面只做静态视觉预览，没有键盘／触屏击打判定，也不等于 Phigros 官方渲染器。

下载按钮导出实际生成的谱面、内部事件和报告。整个输出目录可移动保存，音频副本使它不依赖原输入的路径；不依赖 CDN、HTTP 服务、远程字体或在线脚本。保留同目录全部文件，浏览器仍需支持输入音频编码。

| 文件 | 用途 |
| --- | --- |
| `audio.wav`／`audio.ogg` | 本次任务的音频快照 |
| `audio_events.json`／`.npz` | 共享候选、连续特征、未筛选概率和指纹；旧权重明确标 `chart_onset` |
| `tempo.json` | 自动估计或人工时基的来源、参数、候选和不确定状态 |
| `4k.raw.json`、`freex.raw.json` | 模型原始静态输出，仅作诊断，Hold 可能越过音频末尾 |
| `4k.json`、`freex.json` | 通过格式、位置、时间及候选覆盖检查的最终 Phigros v3 JSON |
| `*.events.json` | 稳定音符 ID、来源序号、绝对音频时间、键型和屏幕横坐标 |
| `*.repairs.json` | 静态修复的数量与逐音改动；不归为模型收益 |
| `preview.html` | 使用最终事件的离线同步预览 |
| `manifest.json` | 完成标记、运行配置、权重／产物哈希、逐模式检查、耗时与资源 |
| `job.json` | 运行中进度／失败／取消状态；不是完成凭据 |

4K 保留四个固定位置及 Tap／Drag／Hold／Flick。free-x 保留自由横坐标和多指，没有隐式双指或押数上限。静态修复保留音符与起点数量，按现有策略处理横坐标及 Hold 截短；不强行补齐键型配额。原始与最终事件沿用来源序号以跟踪修改。

## 完整性、取消与限制

解码保留整曲 AR 状态；**布局编码默认沿用历史整曲注意力**，运行前保守估计注意力矩阵、工作空间和解码 KV 缓存，并按 RAM／显存余量拒绝超预算曲目。600 秒只是输入上限，不保证所有设备都可处理。不会因资源不足自动更换编码语义。

可显式指定 `--layout-encoder chunked` 实验：1024 mel 帧的 core 与左右 128 帧 context。单曲同权重／候选／seed 对照已观察到布局和键型分布明显变化，尚未校准，因此不作为默认。共享音频候选仍使用已版本化的 1024+128 分块基线；两个配置分别记录。

完整覆盖所有选中的量化候选是链路完整性检查，不是听觉质量指标。事件量化到最近合法 tick；末尾若最近 tick 在音频外，改用音频内最后 tick，避免静默丢失末候选。报告合并数、边界钳制数与最大时间误差；边界误差可超过普通舍入的半 tick，但小于一 tick。不宣称选中了全部音乐事件。

解码最多 12000 个外层步骤，可显式修改 `--max-decoder-steps`；必须自然生成 EOS 才允许完成。预算耗尽时即使所有候选都已有音符，也可能截断末和弦或 Hold，仍明确失败；不接受历史解码器强补 EOS 的结果。空候选、非法 BPM、资源不足、输入不支持、某一路异常或超时同样明确失败。

运行时创建 `OUTPUT/CANCEL` 文件可请求取消，也可 Ctrl+C／发送 SIGTERM。推理窗口之间、每 32 个解码步骤及音频复制阶段检查取消和预算。保留部分文件供诊断，但无 `manifest.json`。取消不能中断一个正在执行的 GPU kernel；完成该小步后退出。

真实播放器导入、触屏判定、难度条件学习、自动节拍、独立声学标注、动态多判定线演出和后续编辑各自另行验收。现有谱面监督权重不会因为换了产品入口而变成声学真值模型。

## 本地交互与自动时基（N1）

```bash
python -m script.serve_product \
  --model-root /path/AudioLab/checkpoints/timing_v1 \
  --output out/local-product --device cpu --port 7860
```

打开 `http://127.0.0.1:7860`，选择音频和玩法后生成；手动 BPM 位于可选设置中。支持进度、取消、失败后重试、最近任务、预览与导出。服务只绑定回环地址、检查本地来源，不提供公开网络部署或账户功能。

一次最多一个上传／生成任务，忙时拒绝新任务。每个生成任务使用独立子进程，取消后等待其退出才释放槽位；重启时拒绝覆盖仍在运行的旧任务。上传按块落盘并保留 8 GiB 磁盘余量，不在内存中积累整个请求；输入最多 128 MiB。工作完成后清理上传临时副本，生成目录保留自己的音频。任务目录必须完整保留，HTTP 音频支持单段 byte range 以便跳转。仅服务完成任务清单里的产物，不支持任意文件路径。

自动时基复用已提取的 22050 Hz／512 hop log-mel：`librosa.onset.onset_strength`（`center=False`）→ 8 秒自相关 tempogram → `librosa.feature.tempo`。采用该库明确记录的 120 BPM 初始先验、std=1、上限 320 BPM；120 是先验，不是无信号时返回的观测值。估计值直接作为导出和 AR 时基，不按已有谱面的 BPM 分布事后挑选，也不把音符吸附到拍点。[起音强度文档](https://librosa.org/doc/0.11.0/generated/librosa.onset.onset_strength.html)、[速度估计文档](https://librosa.org/doc/0.11.0/generated/librosa.feature.tempo.html)。

log-mel 沿用权重的 2048 点居中 STFT，帧 `i` 的中心是音频绝对时间 `i×512/22050` 秒；`onset_strength(center=False)` 不再追加半窗移位。原始 WAV 点击及 1／4 秒前导静音夹具覆盖这条加载链路，未删除前导静音。居中窗会提前响应冲激，测试中 flux 峰早约 1–2 帧；这里只利用周期，不将其峰值当成模型音符的起音时间。

`tempo.json` 与报告分别保存 `timebase_bpm`、未审核的 `musical_tempo_estimate`、方法／版本／参数／分析范围／音频哈希、半／倍速候选、自相关强度和局部估计分布。相关强度不是校准概率；局部跳变可能是半／倍速切换，不是经过验证的真实变速。所有自动结果标为 `estimated_unconfirmed`，预览中提示可手动调整。没有信号、少于四个显著起音峰或短于四秒时，返回 `needs_tempo`，不默默填常数；这些是版本化工程启发式，尚未按真实音乐校准。

**已知可控样例限制：** 默认先验对 180 BPM 均匀点击输出约 89.10 BPM，表现为半速；在 90→180 BPM 的两段点击中，局部估计还可能一直保持约 89.10 BPM，不能据此判断没有变速。120 BPM、前导静音／弱起样例可在一个分析帧的周期误差内恢复；120→160 样例可触发局部变化提示。这些功能样例不构成真实音乐 BPM 准确率、可靠 tempo map 或学习式 beat 模型已完成的证据。
