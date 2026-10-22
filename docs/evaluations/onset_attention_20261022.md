# 起音、音频条件与全局时间注意力（2026-10-22）

本轮源于用户对“简单三音能否识别”“第二阶段是否充分训练”“全局音频随歌曲时间 t 分配注意力”的质疑。先定位，再做训练对照；工程测试、推理消融和训练后质量分别记录。原 full300 H 产品权重保留。

## 实测起音与静音

输入取自 [Tonejs 的 Salamander 钢琴采样](https://github.com/Tonejs/audio/tree/master/salamander)，作者 Alexander Holm，CC BY 3.0。下载 C4、Fs4、A4 共 208,085 字节。固定波形幅度标准裁去开头空白，加入 2ms attack 和 100ms release 淡变，单音保留 1 秒；拼接为已知起点的音频。它们是受控采样测试，不能充当自然歌曲泛化准确率。

测试使用交付的 `full300_history/transcriber/transcriber_best.pt`，阈值 0.7、35ms NMS 沿用原配置，未根据测试调整。

| 输入 | 真实起音 | 识别起音 | F1@25ms |
|---|---:|---:|---:|
| 10 秒，1/4/7 秒三个不同钢琴音 | 3 | 3 | 1.000 |
| 中间音振幅降至 3% | 3 | 2 | 0.800 |
| 三次同音重复 | 3 | 3 | 1.000 |
| 32 秒，1/24/29 秒起音，跨推理分窗 | 3 | 3 | 1.000 |
| 10 秒纯静音 | 0 | 0 | 无误报 |

普通三音预测时间为 0.990865、3.984611、6.993959 秒。弱音附近概率约 0.538568，低于报告候选阈值，但仍原样进入解码器的 dense soft-onset 输入。没有提高阈值，也没有把弱音标成负样本。

## 第二阶段的失败

对相同音频分别使用预测 soft-onset、理想起音构造的 soft-onset、全零 soft 条件；音频特征、时基 120 BPM、种子 42、温度 1、top-p .95 相同。另有贪心解码对照，无后处理。理想条件分布与训练预测值不同，因此它只是定位手段，不代表质量上界。

- 4K 在三音上三种条件均输出 `BOS → DTB_16 → DTB_16 → EOS`，0 个音符；音乐时钟到 16 秒。greedy 和 sampling 一致。
- free-x 三音采样得到 34 个音符、29 个唯一起点；预测、理想和零 soft 条件的 token 序列完全相同。20 个唯一起点位于已知静音区域。
- free-x 在纯静音上采样得到 27 个音符、22 个唯一起点。greedy 输出空谱。

前端能识别普通三音，不能保证解码器依证据落键。这是第二阶段的能力缺口；单个分布外短片段不足以把原因唯一归为参数量或训练轮数。

“清零音频特征后，相对原有声谱的损失增加”说明正确音频有帮助，不是需要压低的差距。它与“真实静音音频对应空谱”不同。后者应有空谱监督并独立验收。soft-onset 消融变化很小，也可能存在音频特征信息冗余，不能单凭这一项证明通路无效。

## 音频如何进入 cross-attention

当前没有 VQ/离散音频词表。22,050Hz 波形经过 128-bin log-mel（512 hop，约 23.22ms），冻结的编码器每两帧输出一个 256 维连续向量，约 46.44ms 一个 memory token。布局编码器读取整曲；独立起音前端采用 1024 mel 帧 core、128 帧上下文分窗。这两种窗口不是同一件事。

memory 是音频向量投影、音频帧时间嵌入与每帧概率/偏移投影之和，经 LayerNorm。解码 query 来自已有谱面 token、音乐时钟、历史状态和因果 self-attention。

旧解码器的 `cross_window=64` 屏蔽离当前音乐时钟超过约 2.97 秒的 memory。编码器读过整曲不等于解码器能直接访问所有位置的细节。

新可选 `cross_time_scale=64, cross_window=None` 使用全局时间偏置：

```text
score(t,s) = Q(t)·K(s)/sqrt(d) - log(1 + ((s-t)/τ)^2)
attention(t,:) = softmax(score(t,:))
τ = 64 memory frames ≈ 2.97 seconds
```

所有有效音频帧可见，只有 padding 为 -∞。时间距离是软先验，内容相关性仍由网络学习。scale 本轮固定用于受控比较，不宣称已找到最优值。这里 t 是歌曲时间；diffusion 的去噪步数是另一变量。

cross-attention 能表达这种需求，问题不是它要求局部窗口。框架参考 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)；本轮时间偏置为项目实验设计，并非该论文的质量结论。

## 区分推理消融、迁移训练与从零训练

1. **推理消融已完成**：同一旧权重切换 local/global/time_bias，对 SURREALISM、ABYSS MOTION、Eltaw 记录真实历史下的分项 NLL、最后一层 32 个 query 的注意力质量分布，以及 greedy/top-p 两种整曲输出。这个实验不能比较各架构训练后的能力。
2. **迁移训练**：最初完成了 4K local 控制组的 300 首一轮微调。用户质疑旧权重的局部窗口偏向后，改为从随机初始化比较。没有另一匹配迁移组，不据此得出架构结论。
3. **从零训练对照**：4K 与 free-x 分开，每种玩法三组，各自的三组具有字节相同的随机初始参数。音频编码器冻结，解码器全部随机初始化。相同 300 首、相同顺序、10 epochs、AdamW、初始学习率 3e-4、5% warmup 与 cosine 衰减、seed 42、完整上下文与 gradient checkpointing。

每 10 首音乐额外训练一段数字静音，长度轮换 5/13/26/41 秒，目标 `BOS EOS`。输入通过真实 mel/起音/布局编码器，而非把中间特征强行置零；10 秒测试静音不参与梯度。每组共 3,000 次音乐更新、300 次静音更新。每轮保存学习曲线和一个最新模型。

首次会话中断时，两种玩法的 local 组都已训练 10 轮；4K 评价完成，free-x 评价中断。恢复流程跳过完成的 4K local，free-x local 仅重跑评价，未重训。新增 checkpoint 保存 optimizer、随机数和数据顺序状态，可从完整 epoch 边界继续；旧的已完成模型也可只恢复评价。恢复后采用独立后台进程，PID 与日志见 `out/onset_attention_20261022/background_jobs.json`。

最终六组均完成 10 轮和评价，同一玩法的三组初始化 hash 相同。未参与训练的 10 秒纯静音在六组的 greedy 和 sampling 下均为 0 音符；普通三音仅 4K local 的 sampling 产生 5 音符、唯一起点 F1@50ms=0.25，静音区有 2 个错误起点，其余组合全部空谱。静音测试改善，但稀疏有声输入六组均不合格，因此不替换产品权重。

| 玩法 | local 采样 F1@50ms | global 采样 F1@50ms | time_bias 采样 F1@50ms |
|---|---:|---:|---:|
| 4K | 0.4656 | 0.6037 | 0.4244 |
| free-x | 0.4984 | 0.4971 | 0.5021 |

这是三首已训练曲、固定 seed=42 的宏平均。4K 的 greedy 均值分别为 0.5489/0.6960/0.7250，排序与 sampling 不同；free-x 为 0.4040/0.1127/0.1975。不能据单种解码、单种子与训练内 F1 宣称某架构稳定优胜。逐曲记录分别见 [4K](4k_attention_scratch_20261022.md) 和 [free-x](freex_attention_scratch_20261022.md)。

新的数据设计风险需要先排除：当前静音负样本较短，有声样本主要是完整长曲，长度与标签相关；且单 EOS 静音和长音乐分别按各自 token 平均损失更新，可能放大序列起点处的静音梯度。尚未做这两项的单因素对照，不能认定为唯一原因。下一轮优先匹配有声/静音样本长度并联合归一化 loss，再检查三音正样本、静音负样本及完整音乐；不只增加负样本，也不只增加模型大小。

所有 300 首均参与训练，三首音乐对照也是训练内诊断。没有独立自然歌曲测试集，不能发布泛化成绩。10 轮是首轮固定预算，是否充分要看曲线，不能把同轮数等同于各架构都收敛。静音改善还需检查有声三音与音乐是否一起退化成空谱。

推理对照使用缓存参考时基，Eltaw 为原参考谱时基；与用户浏览器自动估算的 129.2 BPM 不是同一生成配置。音符数不能直接与用户谱包相减来归因。

## Diffusion 路线现状

代码和权重都保留，长期分支为 `feature/variable-length-inference`，不是当前 AR 谱包的推理路径：

```text
音频 → AudioWaveEncoder
噪声谱面 latent + 音频条件 → RhythmDiT / DDIM → ChartVAE.decode → 4K 事件数组
```

入口 `src/train.py`、`src/test.py`；模型在 `src/models/dit.py`、`src/encoder/encoder.py`、`src/condition/wave.py`、`src/diffusion/`。本地仍有 `checkpoints/dit/dit_best.pt`、`dit_round1_frozen/dit_best.pt`、`vae/`、`wave/`。当前最佳 DiT 文件记录 epoch 297，ε-MSE 0.175566；这不是 onset F1。最后一条训练日志的音频条件收益约 -2.73%，仅说明该次 loss 对照未显示正收益，不能替代完整生成评价。

旧 DiT 权重没有现行 `audio-offset-post-token-v1` 时间版本，不能据其旧结果宣称已通过当前时间和产品验收。后续应先验证 VAE 重建的起音/持续时间保真、重建后的时间合同，再在相同音频与玩法评估协议下训练 DiT。它继续作为独立 4K 候选，不静默替换 AR。

## 复现与证据

- `script.probe_isolated_onsets`：构造与评分起音测试。
- `script.audit_decoder_alignment fixtures|music|adapt`：原权重的条件/采样消融及迁移控制。
- `script.train_alignment_scratch --mode 4k|freex --epochs 10`：配对从零训练及训练后评价。
- 完整音频、dense 概率、token、训练曲线、checkpoint 在 `out/onset_attention_20261022/`；不入 Git。
- [数值证据](onset_attention_20261022.json) 保留起音、静音、整曲推理消融及六组完成后的分母与分项分数。全部原始 token 和 dense 输入保留在本地输出目录。

工程验证：43 项解码/历史/运行合同/梯度重算测试通过；从零初始化、真实静音编码、理想亚帧时钟、断点后下一次参数更新与样本顺序一致等补充测试也通过。全局注意力验证覆盖远处音频可影响输出、padding 不影响输出、因果性、整段前向与 KV cache 一致。使用实际交付权重、Eltaw 的前 64 个真实历史位置另做前向/逐步解码对照，两种玩法三种注意力配置的 logits 最大绝对差均小于 9e-6，记录在 `out/onset_attention_20261022/forward_step_parity.json`。
