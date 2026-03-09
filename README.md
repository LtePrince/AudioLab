# AudioLab

基于 DiT 架构的 Phigros 4k 定轨下落谱面生成项目。将音频作为条件，通过扩散模型生成包含 Tap / Drag / Hold / Flick 四种键型的谱面。

---

## 项目目标

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
sr = 22050,  n_fft = 512,  hop_length = 128
音频帧时长       = hop / sr              ≈  5.8 ms / 帧
音符帧时长       = audio_note_window_ratio(8) × 音频帧  ≈ 46.4 ms / 帧
最大音频帧数     = 32768  →  最大时长 ≈ 190 s（约 3 分 10 秒）
最大音符帧数     = 32768 / 8 = 4096
VAE 压缩比       = 8×  →  latent 长度 = 512
DiT token 数     = 512 / patch_size(4) = 128
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

| 功能 | 纯 PyTorch（当前）| Lightning（未来）|
|------|:-----------------:|:----------------:|
| 多 GPU（DDP）| 需手动实现 | 一行配置 |
| 混合精度（bf16/fp16）| 需手动 scaler | 自动 |
| Checkpoint 管理 | 手动 `torch.save` | `ModelCheckpoint` 回调 |
| 日志（TensorBoard / WandB）| 手动 | 插件式 Logger |
| 验证循环 | 需自行实现 | `validation_step` |

迁移成本低（核心数学逻辑不变），可在模型结构稳定后按需实施。

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
