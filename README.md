# AudioLab

音频条件的 Phigros 谱面生成项目，包含 4K 扩散模型、音频转录器、4K AR / 混合生成，以及 free-x 无轨玩法生成。无轨多判定线演出仍在后续开发计划中。

## 分支管理

| 分支 | 更新范围 |
| --- | --- |
| `main` | 全部模型的稳定集成版本 |
| `develop` | 全部模型的开发集成版本 |
| `feature/variable-length-inference` | 4K 扩散路线：VAE、DiT、音频条件编码器 |
| `feature/transcription-baseline` | 音频转录器 |
| `feature/4k-generation` | 4K AR 解码器、定轨混合生成与约束 |
| `feature/freeform-gameplay` | free-x 无轨玩法、挖掘数据、无轨约束与演出 |

模型更新在各自分支开发、验证和推送，再合入 `develop`；完成集成测试、生成谱面
验证及验证集质量评估后，再发布到 `main`。时间约定和训练资源管理等共用修复单独
提交，合入 `develop` 后同步到受影响的模型分支。分支区分更新归属，工程可以保留
全部模型代码。具体工作约定见 [AGENTS.md](AGENTS.md)。

2026-10-16 整理：原有模型分支此前是逐层叠加开发的，保留已发布历史，以
`9184b7f`（112 项测试通过的时间修复和资源受限训练实现）作为共同开发基线，
新增 `develop` 和 `feature/4k-generation`。这次基线同步包含全部现有模型；之后
独立模型更新分别提交。`main` 本次仍保留原稳定提交，待新一轮模型验证完成后集成。

训练流程与资源限制见 [设计文档](docs/freeform_design.md#7-有资源限制的双路线训练)。

## 生成输出约束

`src.test_decoder` 默认修复 4K 同轨长条冲突：Hold 不越过下一同轨音符或音频
结尾，不足 1 tick 时转 Tap。起点、轨道、音符数保留，日志报告修复数量。
`--raw-output` 保留原始采样结果，用于区分模型表现与后处理效果。
音频截止时间使用波形实际长度，不把 mel 补齐部分计入可生成范围。

在 timing_v1 的 30 首验证输出上，30,847 个音符中有 40 条 Hold 需要截短，
修复后同轨长条重叠为 0，无 Hold 转 Tap；音符起点与轨道完全不变。
这项修复用于消除物理冲突；Hold 是否合适仍需结合对应音频判断，不设数量配额。

free-x 默认在同一时刻的音符之间保留至少 0.04 屏宽间距（`--free-min-gap`），
以最小平方位移调整位置并保持左右次序，再在邻近下一音符前收尾 Hold。
不删除音符；若极端和弦无法满足间距，明确报错，可缩小间距或使用 `--raw-output`。
这仅适用于当前生成的静态单线谱，不代表已经解决手数、连续手速或多线演出。

timing_v1 的 30 首 free-x 验证输出中，104/32,618 个音符被移动（最大屏宽
0.01984），335 条 Hold 截短，0 条转 Tap。距离 <0.03 的近距离和弦 31→0、
Hold 期间邻近音符 627→0；位置 F1 @92.9ms/0.05 从 0.18958 到 0.18965。
起点和音符数保留，onset F1 不变。可复用旧生成结果做 CPU 修复，无须再次采样：

```bash
.venv/bin/python -m script.repair_generated_charts --mode freex \
  --gen-dir checkpoints/timing_v1/eval_decoder_freex --list data/val.txt \
  --out-dir out/playable_freex
```

4K 使用 `--mode 4k` 和对应输入目录。输出目录必须是新目录；原谱、权重保留，
同级的 `.report.json` 记录逐曲修复和冲突计数。

实验性采样开关 `--sampling-policy type_preserving` 对每种键型的**位置条件分布**
分别执行 top-p，保留模型原始的 Tap/Drag/Hold/Flick 概率质量；其他动作另成一组。
它没有手工提高 Hold 权重，也不保证连续生成时恢复训练集的键型比例。
默认仍为 `flat`。诊断脚本同样接受该开关，需先比较真实生成质量再考虑推广。

## 音频节奏实验

`--onset-threshold 0.7 --onset-min-interval-ms 35` 可启用按置信度去重的转录骨架。
参数在 32 首训练曲校准，独立 30 首验证曲的唯一起点 F1 @25ms 为 0.6920
（原始阈值模式 0.6626）。这项指标把同刻和弦计为一个事件，与整谱音符 F1 不同。

`src.train_rhythm` 可在已有音频编码器上训练独立的全局节奏头：以 mel 帧
（约 23.2ms）为单位预测 onset 和时间偏移，训练标签不包含位置或键型配额。
默认冻结编码器，仅训练 33,412 个参数，保留原转录器的 32 通道输出接口：

```bash
.venv/bin/python -m script.train_models --worker src.train_rhythm -- \
  --epochs 8 --run-dir checkpoints/rhythm_global_v1
```

输出 `rhythm_best.pt`、`rhythm_last.pt`、`rhythm_final.pt`（含完整编码器配置与
权重），`--resume` 从 last 继续并检查实验配置。新 checkpoint 可作为生成入口的
`--skeleton-ckpt`，但其概率需要重新校准；不能沿用四轨联合概率的 0.7 阈值。
先用 `script.calibrate_skeleton --phase fit --ckpt ...` 在训练列表校准，再以
相同 checkpoint、冻结的 `--calibration` 报告和独立列表执行 `--phase evaluate`。
本实验的接受条件是音频起点和完整生成质量，不是 Hold 占比。

第二阶段可从已有节奏头继续微调声学骨干，使用独立 run 目录，例如
`--encoder-ckpt checkpoints/rhythm_global_v1/rhythm_best.pt --lr 0.0001
--encoder-lr 0.00001 --run-dir checkpoints/rhythm_global_v2`。四轨分类头保持冻结，
该 checkpoint 应通过全局节奏骨架接口使用；不将它视为重新校准过的四轨转录器。
默认 `--encoder-lr 0` 仍只训练节奏头，旧 checkpoint 和训练入口行为保留。

生成入口和诊断入口支持 `--skeleton-calibration <训练集 fit 报告>`，自动读取
已选阈值与最小间隔，并检查报告已完成、来自 fit 阶段且 checkpoint 路径一致。
使用它时不再同时传手工阈值/间隔，避免混用不同模型的概率标度。例如：

```bash
.venv/bin/python -m script.train_models --worker src.test_decoder -- \
  --ckpt checkpoints/timing_v1/decoder/decoder_best.pt \
  --skeleton-ckpt checkpoints/rhythm_global_v2/rhythm_best.pt \
  --skeleton-calibration out/global_rhythm_v2_refined_train32/report.json \
  --list data/val.txt --out-dir out/global_rhythm_4k --seed 42 --device cuda:0
```

free-x 使用 `decoder_freex/decoder_best.pt` 并换一个输出目录。上述仍是开发实验
组合；详见 [节奏实验结果](docs/freeform_design.md#11-独立全局节奏模型2026-10-17)。
校准后不要覆盖原 checkpoint；权重、校准原始报告和生成谱保留在本地，不提交 Git。

### 完整验证集对照

三套冻结配置为 timing_v1 原始骨架、原转录器 0.7/35ms 去重骨架，以及全局
v2 的 0.25/25ms 骨架。固定 seed=42、flat top-p=0.95，两种 AR 权重不变。
下面两个 GPU 任务依次执行；已完成的 6 曲子集可以复用：

```bash
.venv/bin/python -m script.train_models --worker script.diagnose_decoder -- \
  --data-list data/val.txt --samples 30 --top-ps 0.95 \
  --onset-threshold 0.7 --onset-min-interval-ms 35 \
  --reuse-report out/decoder_nms35_val6/report.json --output out/decoder_nms35_val30

.venv/bin/python -m script.train_models --worker script.diagnose_decoder -- \
  --data-list data/val.txt --samples 30 --top-ps 0.95 \
  --skeleton-ckpt checkpoints/rhythm_global_v2/rhythm_best.pt \
  --skeleton-calibration out/global_rhythm_v2_refined_train32/report.json \
  --reuse-report out/decoder_global_v2_refined_val6/report.json \
  --output out/decoder_global_v2_refined_val30

.venv/bin/python -m script.compare_full_charts
```

没有子集时省略 `--reuse-report`。中断恢复时保持同一参数和输出目录，将
`--reuse-report ...` 换为 `--resume`。复用时检查生成设置、权重与文件指纹；旧版
报告没有历史指纹，只能检查设置和 checkpoint 修改时间，此限制写入新报告。

CPU 对照要求每套配置都有完整 30 曲，保留原始输出，并对三套方案应用相同的
静态线修复。`out/full_chart_comparison_val30/report.json` 包含逐曲指标、配对
bootstrap 区间、退化曲目、时间/格式检查与修复量；`playable/` 下保存修复后的
三套定轨/无轨谱面。指标沿用整谱多音符匹配，不与唯一起点 F1 混用。

2026-10-17 的完整 30 曲结果已经完成：定轨原始/NMS35/全局 v2 的 F1@25ms
为 **0.5917/0.5990/0.5990**，无轨为 **0.5849/0.5891/0.5939**。新全局模型
相对 NMS35 的配对区间跨 0，未能确认稳定额外收益。定轨后续采用 NMS35 作对照
基线，无轨保留两个候选并优先处理位置与同刻多押；通用默认参数和 main 不变。
全部 180 份谱面已按同一规则检查与修复，静态碰撞归零，但不代表人工可玩性验收。
完整逐曲表、区间和退化分析见 [定轨报告](docs/evaluations/4k_val30.md) 与
[无轨报告](docs/evaluations/freex_val30.md)。

---

## 4K 扩散路线的原始设计

- 输入：音频文件（.ogg / .mp3）
- 输出：Phigros 格式谱面（固定 4 轨道、固定下落方向、固定判定线位置）
- 模型架构：1D DiT（Diffusion Transformer）+ 潜空间 VAE + 多尺度音频条件编码器

---

## 数据格式约定

### Phigros note 类型映射

| Phigros type | 键型    | 原始标签 | `note_type` 存储值（归一化） | 解码阈值          |
|:------------:|:-----:|:----:|:--------------------:|:-------------:|
| 1            | Tap   | 1    | 0.25                 | ≤ 0.375       |
| 2            | Drag  | 2    | 0.50                 | 0.375 ~ 0.625 |
| 3            | Hold  | 3    | 0.75                 | 0.625 ~ 0.875 |
| 4            | Flick | 4    | 1.00                 | > 0.875       |

> `note_type` 通道存储归一化值（原始标签 ÷ 4），使其与其他 [0,1] 通道量级一致；  
> 对外接口仍使用原始标签 1/2/3/4。

### `positionX` → 轨道量化

以 2.5 为段宽，量化边界为 -2.5 / 0 / 2.5：

| 轨道  | positionX 范围  | 重建中心值 |
|:---:|:-------------:|:-----:|
| 0   | `(-∞, -2.5)`  | -3.75 |
| 1   | `[-2.5,  0)`  | -1.25 |
| 2   | `[  0,  2.5)` | 1.25  |
| 3   | `[ 2.5, +∞)`  | 3.75  |

### `note_array` 通道布局（共 20 通道）

每个轨道 k（k = 0,1,2,3）占 5 个通道：

```
[k +  0]  is_start   ∈ {0,1}        — 此帧该轨是否有音符起始
[k +  4]  start_off  ∈ [0,1]        — 起始时刻亚帧偏移
[k +  8]  is_holding ∈ {0,1}        — 此帧该轨是否处于 Hold 延伸中
[k + 12]  end_offset ∈ [0,1]        — Hold 结束时刻的亚帧偏移
[k + 16]  note_type  ∈ {0.25,0.5,0.75,1.0}  — 0.25=Tap / 0.5=Drag / 0.75=Hold / 1.0=Flick
                                     （原始标签 1/2/3/4 ÷ 4 归一化存储，is_start=0 时填 0）
```

形状：`(20, T_note)`，`T_note = 4096`

### 时序参数对齐

```
sr = 22050,  n_fft = 2048,  hop_length = 512
mel 帧时长       = hop / sr                      ≈ 23.22 ms / 帧
音符帧时长       = frame_ms = hop/sr/4×8×1000    ≈ 46.44 ms / 帧（= 2 个 mel 帧）
最大音符帧数     = 4096  →  最大时长 ≈ 190 s（约 3 分 10 秒）
最大 mel 帧数    = 8192（与谱面轴等时长；chart_frames_to_mel_frames() 换算，
                   切勿按音符帧数填充 mel——那只覆盖半首歌且时间轴错位 2 倍）
VAE 压缩比       = 16×  →  latent 长度 = 4096 / 16 = 256
音频 token 数    = 8192 / 16 = 512  →  _align_time() 线性插值到 256 与 latent 对齐
DiT token 数     = 256（无 patchify，与 latent 等长）
Phigros tick→ms = tick / 32 × (60000 / bpm)
```

---

## 项目目录结构

```
AudioLab/
├── README.md
├── main.py
├── pyproject.toml
├── configs/
│   ├── autoencoder.yaml          # VAE 超参
│   └── chart_diffusion.yaml      # 扩散模型超参
├── data/
│   ├── data.txt                  # 训练集路径列表（每行一条 Phigros JSON 路径）
│   ├── audio/                    # 音频文件 & 预计算 mel .npy
│   ├── chart/                    # Phigros JSON 谱面文件
│   └── json/                     # （旧）备用
├── src/
│   ├── data/
│   │   ├── audio2mel.py          # ✅ 已有：CPU/GPU Mel 提取
│   │   ├── phigros_convertor.py  # 📌 Phigros JSON ↔ note_array 转换
│   │   └── phigros_dataset.py    # 📌 Dataset + 数据增强
│   └── models/
│       ├── chart_vae.py          # 📌 1D VAE（Encoder / Decoder / KL）
│       ├── audio_encoder.py      # 📌 多尺度 Mel 条件编码器
│       ├── dit_1d.py             # 📌 1D DiT 主干（带 Cross-Attention）
│       └── diffusion.py          # 📌 扩散训练框架（DDPM / DDIM）
└── scripts/
    ├── generate.py               # 📌 推理 Pipeline
    └── evaluate.py               # 📌 评估与可视化
```

---

## TODO List

### Phase 0 — 数据格式设计

- [ ] **P0-1** 确定 4 轨道的 `positionX` 量化策略：以 2.5 为段宽，边界为 -2.5/0/2.5，重建中心值为 -3.75/-1.25/1.25/3.75
- [ ] **P0-2** 设计 `note_array` 的 20 通道布局（见上方"数据格式约定"）
- [ ] **P0-3** 确定并记录时序帧参数（见上方"时序参数对齐"）

---

### Phase 1 — 数据层（`src/data/`）

**`phigros_convertor.py`**（参考 `Mug-Diffusion/mug/data/convertor.py`）

- [ ] **P1-1** 定义 `PhigrosMeta` 数据类（存储 `bpm`、`audio_path`、`json_path`、`version`、原始 `judgeLineList`）
- [ ] **P1-2** 实现 `parse_phigros_file(json_path, convertor_params)` → 合并所有 judgeLine 的 notes，返回扁平化 note 列表和 `PhigrosMeta`
- [ ] **P1-3** 实现 `Phigros4kConvertor.read_time(tick)` → `(time_ms, frame_index, sub_frame_offset)`
- [ ] **P1-4** 实现 `Phigros4kConvertor.objects_to_array(notes, meta)` → `(note_array: ndarray[20, T], valid_flag: ndarray[T])`
  - positionX → lane 量化
  - Tap(1)：填 `is_start`, `start_off`, `note_type=0.25`
  - Drag(2)：填 `is_start`, `start_off`, `note_type=0.50`
  - Hold(3)：填 `is_start`, `start_off`, `note_type=0.75`；逐帧填 `is_holding`；末帧填 `end_offset`
  - Flick(4)：填 `is_start`, `start_off`, `note_type=1.00`
- [ ] **P1-5** 实现 `Phigros4kConvertor.array_to_objects(note_array, meta)` → Phigros JSON note 列表
  - 阈值解码 `is_start`（>0.5）、`note_type` 分类（0.375 / 0.625 / 0.875 为分界）→ 还原为原始标签 1/2/3/4
  - 逆量化 lane → `positionX` 中心值（-3.75 / -1.25 / 1.25 / 3.75）
  - frame_index + sub_offset → tick
  - Hold：根据 `is_holding` 计算 `holdTime`
- [ ] **P1-6** 实现 `save_phigros_file(meta, note_array, output_path)`：重建合法 Phigros JSON（`formatVersion`, `judgeLineList`，固定 speed/rotation events）

**`phigros_dataset.py`**（参考 `Mug-Diffusion/mug/data/dataset.py`）

- [ ] **P1-7** 实现 `PhigrosDataset(Dataset)`：`__getitem__` 返回 `{audio, note, valid_flag, meta}`
  - 音频：`AudioGPUprocessor.forward()` → `(128, T_audio)` log-mel，支持磁盘 `.npy` 缓存
  - 谱面：`parse_phigros_file` + `objects_to_array` → `(20, T_note)`
- [ ] **P1-8** 实现数据增强：
  - 镜像（lane 3↔0, 2↔1，同步翻转 positionX）
  - 随机列置换（仅限 Tap/Flick，不跨 Hold 连续段）
  - 时间 rate 变速（0.9~1.1），同步缩放 mel 和 note_array
  - Mel 频率掩码（SpecAugment）
- [ ] **P1-9** 编写路径扫描脚本，生成 `data/data.txt`（每行一条 Phigros JSON 路径）

---

### Phase 2 — First Stage VAE（`src/models/chart_vae.py`）

参考 `Mug-Diffusion/mug/firststage/autoencoder.py` 和 `mug/model/models.py`

- [ ] **P2-1** 移植 `Encoder`（1D ResNet）：输入 `(B, 20, 4096)` → 输出 `(B, 32, 512)`（8× 时间压缩，`z_channels=16`）
- [ ] **P2-2** 移植 `Decoder`（1D ResNet 上采样）：输入 `(B, 16, 512)` → 输出 `(B, 20, 4096)`
- [ ] **P2-3** 实现 `DiagonalGaussianDistribution`（`.sample()`, `.mode()`, `.kl()`）
- [ ] **P2-4** 实现 `PhigrosReconstructLoss`（参考 `ManiaReconstructLoss`）：
  - `is_start` / `is_holding`：BCE loss
  - `start_off` / `end_offset`：L1 loss（仅对有效位置加权）
  - `note_type`：smooth L1 loss（仅在 `is_start=1` 的位置）
  - `valid_flag` 用于 mask 无效尾部帧
- [ ] **P2-5** 实现 `AutoencoderKL`（`encode`, `decode`, `forward`, Lightning `training_step`）
- [ ] **P2-6** 编写 `configs/autoencoder.yaml`（`x_channels=20, z_channels=16, channel_mult=[1,2,4,4]`）
- [ ] **P2-7** 训练 VAE，验证重建谱面 F1 分数（音符位置命中率达到可接受基线）

---

### Phase 3 — 音频条件编码器（`src/models/audio_encoder.py`）

参考 `Mug-Diffusion/mug/cond/wave.py` 的 `MelspectrogramScaleEncoder1D`

- [ ] **P3-1** 移植 `MelspectrogramScaleEncoder1D`：输入 `(B, 128, 32768)` → 输出 10 尺度特征列表
  - 配置：`middle_channels=128, channel_mult=[1,1,1,1,2,2,2,4,4,4]`
- [ ] **P3-2** 确认各尺度输出形状 `(B, C_i, T_i)`，与 DiT 各层 context 对齐
- [ ] **P3-3** 实现 `AudioContextProjector`：将多尺度列表中选定尺度的 `(B, C_i, T_i)` 插值并投影到统一 `context_dim`（如 `(B, T_ctx, 256)`）
- [ ] **P3-4** 单元测试：随机 mel 输入，验证各尺度输出形状正确

---

### Phase 4 — 1D DiT 主干（`src/models/dit_1d.py`）

参考 `DiT/models.py` + `Mug-Diffusion/mug/model/attention.py`

- [ ] **P4-1** 实现 `PatchEmbed1D`：`Conv1d(z_channels=16, hidden_size, kernel_size=4, stride=4)` → `(B, 128, D)`

- [ ] **P4-2** 实现 `get_1d_sincos_pos_embed(embed_dim, length)`（改写 DiT 的 2D 版为 1D）

- [ ] **P4-3** 直接复用 `TimestepEmbedder`（`DiT/models.py`）

- [ ] **P4-4** 实现 `DiTBlock1D`（在 `DiTBlock` 基础上添加 Cross-Attention 分支）：
  
  ```
  x → Self-Attn (adaLN-Zero) → Cross-Attn (context=audio_ctx) → FFN (adaLN-Zero)
  ```
  
  Cross-Attn 参考 `mug/model/attention.py` 的 `ContextualTransformer`

- [ ] **P4-5** 实现 `FinalLayer1D`：输出 `(B, 128, patch_size × z_channels)` → unpatchify → `(B, 16, 512)`

- [ ] **P4-6** 实现完整 `ChartDiT1D`，`forward(z_noisy, t, audio_ctx)` → `z_pred`

- [ ] **P4-7** （可选升级）参考 `Open-Sora/opensora/models/mmdit/layers.py` 的 `DoubleStreamBlock` 实现双流变体（音频 token 与谱面 token 互相 attend）

- [ ] **P4-8** 单元测试：`(B=2, 16, 512)` + timestep + audio ctx → 输出 `(B, 16, 512)` ✓

---

### Phase 5 — 扩散训练框架

参考 `Mug-Diffusion/mug/diffusion/diffusion.py`

- [x] **P5-1** `src/diffusion/schedule.py`：`make_beta_schedule`（linear / cosine）+ `NoiseSchedule`（含 `q_sample`、`snr`、posterior buffers）
- [x] **P5-2** `src/train.py`：训练主循环（纯 PyTorch）
  - 冻结 VAE + Wave encoder，只训练 DiT
  - CFG Dropout（`--cfg-drop`，概率性将 `audio_c` 置零）
  - AdamW + CosineAnnealingLR
  - 梯度累积（`--grad-accum`）+ 梯度裁剪
  - 定期 checkpoint 保存 + 续训支持（`--dit-ckpt`）
- [ ] **P5-3** 编写 `configs/chart_diffusion.yaml`（训练超参配置文件）

---

### Phase 6 — 推理 Pipeline（`scripts/generate.py`）

参考 `Mug-Diffusion/mug/diffusion/ddim.py`

- [ ] **P6-1** 移植 `DDIMSampler`（50~100 步加速采样）

- [ ] **P6-2** 实现 `generate_chart(audio_path, model, cfg_scale=3.0, ddim_steps=50)`：
  
  ```
  audio → mel → audio_encoder → ctx
  z_T ~ N(0,I) → DDIM 去噪 → z_0
  z_0 → VAE decode → note_array → array_to_objects → save_phigros_file
  ```

- [ ] **P6-3** 实现 CFG 推理：`ε_cfg = ε_uncond + scale × (ε_cond − ε_uncond)`

- [ ] **P6-4** 实现后处理：阈值过滤低置信度音符（`is_start < 0.5`），同帧同轨冲突去重

---

### Phase 7 — 验证与评估（`scripts/evaluate.py`）

- [ ] **P7-1** 实现谱面 piano-roll 可视化：时间轴 × 4 列，Tap/Drag/Hold/Flick 用不同颜色
- [ ] **P7-2** 实现对齐评估指标：
  - **F1@10ms**：以 10 ms 窗口判断音符位置命中
  - **音符密度–音频能量 Pearson 相关系数**（逐轨及全局）
- [ ] **P7-3** 实现 VAE 重建基线（encode→decode，不经扩散，验证性能上界）
- [ ] **P7-4** 音频对齐可视化：将生成谱面音符时刻叠加到 mel 图上，人工判断节拍对齐效果

---

### 未来计划 — PyTorch Lightning 训练框架

> 当前 `src/train.py` 使用纯 PyTorch 实现，逻辑透明、易于调试。
> 随着实验规模扩大（多 GPU、混合精度、自动 checkpoint 管理），可迁移至 **PyTorch Lightning**。

迁移后的架构：

```
src/
├── lit_module.py      # LightningModule：封装冻结 VAE + Wave + 可训练 DiT
│     training_step()  # q_sample → DiT forward → MSE loss
│     validation_step()# DDIM 采样 → 解码 → F1 评估
│     configure_optimizers()  # AdamW + CosineAnnealingLR
└── lit_train.py       # Trainer 入口：多 GPU / bf16 / 自动 checkpoint
```

Lightning 相比纯 PyTorch 的优势：

| 功能                      | 纯 PyTorch（当前）   | Lightning（未来）        |
| ----------------------- |:---------------:|:--------------------:|
| 多 GPU（DDP）              | 需手动实现           | 一行配置                 |
| 混合精度（bf16/fp16）         | 需手动 scaler      | 自动                   |
| Checkpoint 管理           | 手动 `torch.save` | `ModelCheckpoint` 回调 |
| 日志（TensorBoard / WandB） | 手动              | 插件式 Logger           |
| 验证循环                    | 需自行实现           | `validation_step`    |

迁移成本低（核心数学逻辑不变），可在模型结构稳定后按需实施。

---

### 未来计划 — Mapperatorinator 参考特性

> 以下特性来自 [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator)（osuT5 + osu-diffusion 双阶段管线），
> 当前 AudioLab 尚未实现，可在模型稳定后逐步引入。

#### 一、多维条件控制（Conditioning）

| 特性                                    | Mapperatorinator 实现                                                   | AudioLab 现状 | 预期收益        |
| ------------------------------------- | --------------------------------------------------------------------- | ----------- | ----------- |
| **难度条件**                              | StarRating → 离散难度类别嵌入，独立 CFG dropout (`diff_dropout_prob=0.2`)        | ❌ 无难度控制     | 可生成指定难度谱面   |
| **风格 / Mapper 条件**                    | mapper_id → style embedding；`num_mappers=3731`                        | ❌ 无风格控制     | 可模仿特定制作者风格  |
| **描述符 / Tag 条件**                      | 文字描述符（"stream"/"jump"/"tech"）→ 多热编码类别嵌入                               | ❌ 无描述控制     | 可定向生成特定图型风格 |
| **多条件独立 CFG**                         | 每个条件维度（音频 / 难度 / Mapper / 描述符）分别 dropout + 独立引导尺度                     | 仅音频 CFG     | 更细粒度的生成控制   |
| **Hold note ratio / Scroll speed 条件** | 专用前缀 Token：`add_hold_note_ratio_token`、`add_scroll_speed_ratio_token` | ❌           | 控制长按比例与滚动速度 |

实现要点：

- 在 `RhythmDiT` 中新增 `CondEmbedder`，将难度 / 风格标量映射为向量，拼接到 timestep embedding 后
- 在 `train.py` 中对每个条件独立执行 dropout，推理时每个维度有独立的 `cfg_scale`

#### 二、训练改进

| 特性                               | Mapperatorinator 实现                                                              | 备注                                 |
| -------------------------------- | -------------------------------------------------------------------------------- | ---------------------------------- |
| **节奏权重损失**                       | `rhythm_weight=3.0`：对节奏 Token 区间 `[rhythm_token_start, rhythm_token_end]` 放大损失权重 | 强制模型优先对齐节拍，避免音符位置漂移                |
| **Double-time 数据增强**             | `double_time_prob=0.5`：以 50% 概率将谱面 2× 加速（等价于 0.5× 时间缩放）                          | 与现有 rate stretch 互补，扩展速度分布         |
| **结构化训练日志 (WandB)**              | `log_with='wandb'`，每 10 步记录 loss / lr / grad norm                                | 当前仅 print，无法在线监控或对比实验              |
| **HuggingFace Accelerate 多 GPU** | 通过 `accelerate` 管理分布式训练与混合精度                                                     | 与 Lightning 方案二选一，Accelerate 侵入性更低 |

#### 三、推理改进

| 特性                      | Mapperatorinator 实现                                 | 备注                                 |
| ----------------------- | --------------------------------------------------- | ---------------------------------- |
| **滑动窗口推理**              | `lookback` / `lookahead` 控制上下文重叠窗口；跨窗口传递历史 Token    | 突破当前 4096 帧（≈190 s）上限，支持任意时长       |
| **二阶段精化模型**             | osu-diffusion 在 osuT5 生成序列后再做坐标精化（`refine_iters` 次） | AudioLab 若后续扩展 2D 坐标输出，可考虑同样的两阶段架构 |
| **Beat snapping Token** | `EventRange(EventType.SNAPPING, 0, 16)` 量化节拍细分信息    | 辅助模型学习节奏规律，减少非节拍帧上的噪声预测            |

#### 四、评估改进

| 特性                                   | Mapperatorinator 实现                                                    | 备注                    |
| ------------------------------------ | ---------------------------------------------------------------------- | --------------------- |
| **FID 评估**                           | 训练专用 `OsuClassifier`，用其 feature 层计算生成集与真实集的 Fréchet Inception Distance | 比 F1@10ms 更全面反映谱面分布质量 |
| **Drain time / BPM / StarRating 统计** | `calc_fid.py` 中同时输出谱面的客观难度统计量，与真实分布做对比                                 | 辅助验证生成谱面的难度分布是否符合预期   |

> **优先级建议**（从高到低）：
> 
> 1. 节奏权重损失 — 一行改动，立竿见影
> 2. WandB 日志 — 实验管理必备
> 3. 难度条件控制 — 使模型实用化的关键
> 4. 滑动窗口推理 — 解除时长限制
> 5. FID 评估 — 需额外训练分类器，工程量较大
> 6. 风格 / Mapper / 描述符条件 — 需大规模数据支撑

---

## 模块依赖关系

```
P0（格式设计）
  └─► P1（数据层）
        ├─► P2（VAE）─────────────────────────────┐
        └─► P3（音频编码器）                        │
              └─► P4（DiT 主干）                   │
                    └─► P5（扩散训练）◄────────────┘
                          └─► P6（推理）
                                └─► P7（评估）
```

> **最小可运行里程碑**：P0 → P1 → P2（VAE 能重建谱面）→ P3 → P4 → P5（能完成训练），再实现 P6 进行推理验证。P4-7（MMDiT 双流）和 P7（评估脚本）为后续迭代项。

---

## 参考项目

| 项目              | 主要参考模块                                            |
| --------------- | ------------------------------------------------- |
| `Mug-Diffusion` | 谱面转换、Dataset、VAE、音频编码器、UNet、扩散框架                  |
| `DiT`           | DiTBlock、TimestepEmbedder、FinalLayer、sin-cos 位置编码 |
| `Open-Sora`     | MMDiT DoubleStreamBlock、RoPE 位置编码                 |
| `AudioLDM2`     | CLAP 全局音频嵌入（可选）、DDIM 采样                           |

```
AudioLab/src/
├── data/
│   ├── audio2mel.py          ✅ 已有
│   └── chart_convertor.py    📌 移植自 Mug-Diffusion/mug/data/convertor.py
├── models/
│   ├── chart_vae.py          📌 移植自 mug/firststage/autoencoder.py
│   ├── audio_encoder.py      📌 移植自 mug/cond/wave.py (MelspectrogramScaleEncoder1D)
│   ├── dit_1d.py             📌 改写自 DiT/models.py (1D + cross-attn)
│   └── diffusion.py          📌 移植自 mug/diffusion/diffusion.py
└── dataset/
    └── mania_dataset.py      📌 移植自 mug/data/dataset.py
```
