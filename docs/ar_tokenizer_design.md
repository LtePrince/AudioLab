# AR 谱面 Token 词表设计（v1）

> 目标：把 Phigros 4k 谱面编码为离散 token 序列，供自回归 decoder
> （现有转录编码器 + 交叉注意力 decoder，osuT5 结构）逐 token 生成。
> 设计原则：**tick 原生（零量化损失）、音乐时间不变性（与 BPM 无关）、
> 序列最短、语法可约束**。

## 0. 数据依据（80 首实测）

| 统计 | 值 |
|---|---|
| onset 间隔 (ticks, 1 tick = 1/32 拍) | p50=16 · p90=32 · p99=128 · max=1632；**≤32 占 96%** |
| Hold 时长 (ticks) | p50=16 · p90=59 · p99=224 · max=1015；**100% 整数** |
| 每首音符数 | 均值 ~890，最大 ~1800 |

时间与时长在原始数据里就是整数 tick——token 化**无任何量化损失**
（对比 osuT5 的 10ms 网格：既有损、又不随 BPM 泛化）。

## 1. 词表（共 129 token）

| 类别 | token | 数量 | 说明 |
|---|---|---|---|
| 特殊 | `PAD` `BOS` `EOS` | 3 | |
| 时间步进·拍 | `DTB_1 … DTB_16` | 16 | 前进 n×32 ticks；>16 拍链式连发 |
| 时间步进·tick | `DTT_1 … DTT_31` | 31 | 前进 n ticks（拍内余数） |
| 音符 | `NOTE_L{0-3}_{TAP,DRAG,HOLD,FLICK}` | 16 | 轨道×键型融合，1 token/音符 |
| Hold 时长·拍 | `DURB_1 … DURB_16` | 16 | n×32 ticks；超长链式 |
| Hold 时长·tick | `DURT_1 … DURT_31` | 31 | 拍内余数 |
| 条件预留 | `COND_0 … COND_15` | 16 | 难度/密度/风格控制位，v1 不用 |

融合 NOTE token（而非 LANE+TYPE 两枚）：小数据下**序列长度**
比词表大小金贵；16 个组合词表毫无压力。

## 2. 规范编码规则（保证唯一编码，teacher forcing 一致性）

1. 音符按 onset tick 升序；**同 tick（和弦/双押）内按轨道升序**。
2. 相邻 onset 组之间发时间步进：`gap = 32·b + r` →
   先发 `DTB` 链（每枚 ≤16 拍），再发 `DTT_r`（r>0 时）。
   gap=0（同和弦成员）**不发任何时间 token**。
3. `NOTE_Lk_HOLD` 后**必须立刻**跟时长：`dur = 32·b + r` →
   `DURB` 链 + `DURT_r`；dur ≥ 1 tick（holdTime=0 的退化 Hold 已在数据侧转为 Tap）。
4. 完整序列：`BOS [COND…] { 时间步进 | 音符[+时长] }* EOS`。

## 3. 解码期语法约束（logits mask，全部可判定）

- 同一和弦内轨道不得重复（NOTE 后屏蔽同 lane 的 NOTE，直到出现时间步进）；
- `NOTE_*_HOLD` 之后只允许 `DURB/DURT`；时长未完成前禁止 NOTE/DT/EOS；
- `DURT`/`DTT` 之后不得再接同类 tick token（余数只有一枚）；
- 累计 tick 折算秒数超过音频长度 → 强制 `EOS`；
- `EOS` 仅允许出现在完整音符（含时长）之后。

## 4. 与音频的对齐（架构备注，非词表）

decoder 每个 token 的输入嵌入额外注入**当前绝对秒数**的正弦嵌入
（`秒 = 累计ticks × 60/(32·bpm)`，由前缀确定性可算，训练/推理一致），
交叉注意力据此在编码器的音频帧上定位。rate 增强时按拉伸后的等效
bpm 折算即可兼容。

## 5. 序列长度预算

去重 onset ~700 × ~1.1 枚时间 token + ~890 枚 NOTE + Hold 时长 ~130 枚
+ 首尾 ≈ **2.1k~2.6k token/首**（最密 ~4.5k）。裁剪窗训练无压力。

## 6. 为什么 AR 化后键型塌缩自然消解

teacher forcing 的 CE 按**序列联合分布**训练；推理用温度采样而非
逐帧 argmax——边际分布保真是采样的自然属性，不再需要类权重补丁；
Drag 串/Hold+Flick 组合等设计语义由输出侧自回归上下文承载。

## 7. 工整示例（アンビバレンス 开头，真实数据）

```
tick 128: Tap@L0 + Hold@L3(25 ticks)   tick 160: Tap@L0 + Flick@L3   tick 192: Tap@L0 + Tap@L2

BOS  DTB_4  NOTE_L0_TAP  NOTE_L3_HOLD DURT_25   ← 4拍空白后的开场双押
     DTB_1  NOTE_L0_TAP  NOTE_L3_FLICK          ← 1拍后 Tap+Flick
     DTB_1  NOTE_L0_TAP  NOTE_L2_TAP  …
```

## 8. 待裁决项

1. `DTB` 上限 16 拍（超长静默链式连发，p99 仅 4 拍）——是否够用？
2. `COND` 预留 16 个槽位是否保留（难度/密度控制的将来接口）；
3. 小节锚点 `BAR` token 先不加（可由累计 tick 推导），需要再议。
